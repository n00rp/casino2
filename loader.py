"""
Data Loader — Multi-instrument M5/M30 pipeline för HRL-workers.

Ansvar:
  1. Läs MT5-CSV (tabb-separerad) → pandas DataFrame med DatetimeIndex
  2. Cache → parquet (snabb återladdning)
  3. Resample M5 → M15 med look-ahead-säker labeling
  4. Train/Val/Test split enligt HRL_PLAN.md
  5. Multi-instrument join för parallell träning

Användning:
    from casino2.loader import DataLoader

    dl = DataLoader()
    ustec_m5_train = dl.get('USTEC', 'M5', split='train')
    all_m5_train   = dl.get_all('M5', split='train')  # dict[instrument] = df

Förberedelse (körs en gång):
    python -X utf8 -m casino2.loader --prepare
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd


# ── Konfiguration ───────────────────────────────────────────────────────────

# Paths relativa till Casino2/
BASE_DIR = Path(__file__).resolve().parent            # casino2/
REPO_ROOT = BASE_DIR.parent                           # OpenClaw_RL/

# Källdata ligger kvar på den gamla platsen (flyttades inte vid re-org)
SRC_DIR = REPO_ROOT / "data" / "april2026" / "2017-2026"

# Parquet-cache under casino2/
CACHE_DIR = BASE_DIR / "data" / "parquet"

INSTRUMENTS = ("USTEC", "US500", "DE40")
TIMEFRAMES = ("M5", "M30")  # M15 genereras via resample

# CSV-filnamn-prefix (exakt match i SRC_DIR)
CSV_PREFIX: Dict[tuple, str] = {
    ("USTEC", "M5"):  "USTEC_M5_",
    ("USTEC", "M30"): "USTEC_M30_",
    ("US500", "M5"):  "US500_M5_",
    ("US500", "M30"): "US500_M30_",
    ("DE40", "M5"):   "DE40_M5_",
    ("DE40", "M30"):  "DE40_M30_",
}

# Train / Val / Test / Forward-gränser enligt HRL_PLAN.md (UTC)
SPLIT_DATES = {
    "train_end":   pd.Timestamp("2024-12-31 23:59:59", tz="UTC"),
    "val_end":     pd.Timestamp("2025-10-31 23:59:59", tz="UTC"),
    "test_end":    pd.Timestamp("2026-03-31 23:59:59", tz="UTC"),
    # Allt efter test_end räknas som "april"-forward-window
}

Split = Literal["train", "val", "test", "april", "all"]


# ── CSV-parsing ─────────────────────────────────────────────────────────────

def _find_csv(instrument: str, tf: str) -> Path:
    """Hitta senaste CSV som matchar prefix (sorterar efter namn)."""
    prefix = CSV_PREFIX[(instrument, tf)]
    matches = sorted(SRC_DIR.glob(f"{prefix}*.csv"))
    if not matches:
        raise FileNotFoundError(
            f"Ingen CSV hittad för {instrument} {tf} i {SRC_DIR} "
            f"(prefix='{prefix}')"
        )
    return matches[-1]  # senaste (längst datum-range)


def _parse_csv(path: Path) -> pd.DataFrame:
    """
    Läs MT5-format (tabb-separerad, <DATE>\t<TIME>\t<OPEN>...).

    Returnerar DataFrame med:
      - UTC DatetimeIndex (från DATE+TIME)
      - Kolumner: open, high, low, close, volume (= TICKVOL)
    """
    df = pd.read_csv(
        path,
        sep="\t",
        header=0,
        dtype={
            "<DATE>":    str,
            "<TIME>":    str,
            "<OPEN>":    np.float32,
            "<HIGH>":    np.float32,
            "<LOW>":     np.float32,
            "<CLOSE>":   np.float32,
            "<TICKVOL>": np.int32,
            "<VOL>":     np.int32,
            "<SPREAD>":  np.int32,
        },
    )

    df["timestamp"] = pd.to_datetime(
        df["<DATE>"] + " " + df["<TIME>"],
        format="%Y.%m.%d %H:%M:%S",
        utc=True,
    )
    df = df.set_index("timestamp").sort_index()

    df = df.rename(columns={
        "<OPEN>":    "open",
        "<HIGH>":    "high",
        "<LOW>":     "low",
        "<CLOSE>":   "close",
        "<TICKVOL>": "volume",
    })[["open", "high", "low", "close", "volume"]]

    df = df[~df.index.duplicated(keep="first")]

    return df


# ── Parquet-cache ───────────────────────────────────────────────────────────

def _cache_path(instrument: str, tf: str) -> Path:
    return CACHE_DIR / f"{instrument.lower()}_{tf.lower()}.parquet"


def prepare_parquet(force: bool = False, verbose: bool = True) -> None:
    """Konvertera alla CSV:er till parquet i casino2/data/parquet/."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for instrument in INSTRUMENTS:
        for tf in TIMEFRAMES:
            out = _cache_path(instrument, tf)
            if out.exists() and not force:
                if verbose:
                    print(f"  [skip] {out.name} finns redan")
                continue

            src = _find_csv(instrument, tf)
            if verbose:
                print(f"  parsing {src.name} ...", flush=True)

            df = _parse_csv(src)
            df.to_parquet(out, engine="pyarrow", compression="snappy")

            if verbose:
                print(
                    f"  [ok]   {out.name}: {len(df):,} bars "
                    f"({df.index[0].date()} → {df.index[-1].date()})"
                )


# ── Split-logik ─────────────────────────────────────────────────────────────

def _split_df(df: pd.DataFrame, split: Split) -> pd.DataFrame:
    if split == "all":
        return df

    train_end = SPLIT_DATES["train_end"]
    val_end   = SPLIT_DATES["val_end"]
    test_end  = SPLIT_DATES["test_end"]

    if split == "train":
        return df.loc[:train_end]
    if split == "val":
        return df.loc[train_end + pd.Timedelta(seconds=1): val_end]
    if split == "test":
        return df.loc[val_end + pd.Timedelta(seconds=1): test_end]
    if split == "april":
        return df.loc[test_end + pd.Timedelta(seconds=1):]

    raise ValueError(f"Okänd split: {split}")


# ── Resampling (look-ahead-säkert) ──────────────────────────────────────────

def resample_m5_to_m15(m5: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregera M5 → M15 med label='right', closed='right'.

    Vid tidpunkt t ser vi bara M15-bar som stängde ≤ t−1 om shift(1) tillämpas
    (se join_higher_tf).
    """
    agg = {
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }
    out = m5.resample("15min", label="right", closed="right").agg(agg).dropna()
    return out


def join_higher_tf(
    low_tf: pd.DataFrame,
    high_tf: pd.DataFrame,
    prefix: str = "htf_",
) -> pd.DataFrame:
    """Join high-TF på low-TF med shift(1) för look-ahead-säkerhet."""
    shifted = high_tf.shift(1).add_prefix(prefix)
    return low_tf.join(shifted.reindex(low_tf.index, method="ffill"))


# ── Publikt API ─────────────────────────────────────────────────────────────

def load_instrument(
    instrument: str,
    tf: str,
    split: Split = "all",
) -> pd.DataFrame:
    """Ladda ett instrument+TF och skär ut angivet split."""
    path = _cache_path(instrument, tf)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} saknas. Kör först: python -X utf8 -m casino2.loader --prepare"
        )
    df = pd.read_parquet(path)
    return _split_df(df, split)


def load_all(
    tf: str,
    split: Split = "all",
) -> Dict[str, pd.DataFrame]:
    return {inst: load_instrument(inst, tf, split) for inst in INSTRUMENTS}


class DataLoader:
    """Thin stateful wrapper som håller laddad data i minnet."""

    def __init__(self, tfs: tuple = ("M5", "M30")):
        self.tfs = tfs
        self._cache: Dict[tuple, pd.DataFrame] = {}

    def get(self, instrument: str, tf: str, split: Split = "all") -> pd.DataFrame:
        key = (instrument, tf, split)
        if key not in self._cache:
            self._cache[key] = load_instrument(instrument, tf, split)
        return self._cache[key]

    def get_all(self, tf: str, split: Split = "all") -> Dict[str, pd.DataFrame]:
        return {inst: self.get(inst, tf, split) for inst in INSTRUMENTS}

    def clear(self) -> None:
        self._cache.clear()


# ── CLI ─────────────────────────────────────────────────────────────────────

def _main():
    import argparse

    ap = argparse.ArgumentParser(description="Casino2 data loader")
    ap.add_argument("--prepare", action="store_true",
                    help="Konvertera CSV → parquet i casino2/data/parquet/")
    ap.add_argument("--force", action="store_true",
                    help="Skriv över existerande parquet")
    ap.add_argument("--check", action="store_true",
                    help="Visa summary för alla parquet")
    args = ap.parse_args()

    if args.prepare:
        print(f"Förbereder parquet i {CACHE_DIR} ...")
        prepare_parquet(force=args.force, verbose=True)
        print("Klart.")
        return

    if args.check:
        print(f"Parquet-cache i {CACHE_DIR}:\n")
        for inst in INSTRUMENTS:
            for tf in TIMEFRAMES:
                p = _cache_path(inst, tf)
                if not p.exists():
                    print(f"  [saknas] {p.name}")
                    continue
                df = pd.read_parquet(p)
                print(
                    f"  {p.name:30s} {len(df):>9,} bars  "
                    f"{df.index[0].date()} → {df.index[-1].date()}"
                )
        print("\nSplits:")
        for k, v in SPLIT_DATES.items():
            print(f"  {k:12s} {v}")
        return

    ap.print_help()


if __name__ == "__main__":
    _main()
