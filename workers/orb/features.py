"""
ORB Feature Engineering
========================

13 features för Opening Range Breakout på M5.

Session-öppningar (UTC):
  USTEC / US500: 13:30  (NYSE cash open)
  DE40         : 07:00  (Xetra open)

Opening Range = första 3 × M5-bars (15 min) av session.
Aktiv handels-window = 120 min från session-öppning (24 M5-bars).

Features:
  time_of_day_sin, time_of_day_cos         — cyklisk tid (0-24h)
  minutes_since_open                       — normaliserad [0, 1] inom session-window
  dist_to_or_high                          — (close - OR_high) / ATR
  dist_to_or_low                           — (OR_low - close) / ATR
  or_width_atr                             — (OR_high - OR_low) / ATR
  rvol_m5                                  — volume / SMA(20) vol
  atr_pct                                  — ATR(14) / close
  ret_lag_1 .. ret_lag_5                   — log-returns, shifted

Gates (används av env, ej policy):
  in_session                               — 1 om inom 120-min fönster, annars 0
  or_formed                                — 1 om OR färdigformad (>= 3 bars in i session)
  minutes_to_close                         — minuter till session-slut (för force-exit)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta


# ── Konstanter ──────────────────────────────────────────────────────────────

# Session-öppningar i UTC (timmar, minuter)
SESSION_OPEN_UTC = {
    "USTEC": (13, 30),
    "US500": (13, 30),
    "DE40":  (7, 0),
}

OR_BARS = 3              # 15 min opening range (3 × M5)
SESSION_WINDOW_BARS = 24  # 120 min handels-window
ATR_PERIOD = 14
VOL_SMA_PERIOD = 20

FEATURE_COLS = [
    "time_of_day_sin",
    "time_of_day_cos",
    "minutes_since_open",
    "dist_to_or_high",
    "dist_to_or_low",
    "or_width_atr",
    "rvol_m5",
    "atr_pct",
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "ret_lag_4",
    "ret_lag_5",
]

GATE_COLS = [
    "in_session",
    "or_formed",
    "minutes_to_close",
    "or_high",
    "or_low",
]


# ── Session-logik ───────────────────────────────────────────────────────────

def _tag_sessions(df: pd.DataFrame, instrument: str) -> pd.DataFrame:
    """
    Lägg till kolumner:
      session_id      — unikt id per handelsdag (ökar vid varje session-öppning)
      bars_in_session — 0 vid session-öppning, 1, 2, ..., n; -1 utanför
      in_session      — 1 om 0 <= bars_in_session < SESSION_WINDOW_BARS
    """
    if instrument not in SESSION_OPEN_UTC:
        raise ValueError(f"Okänt instrument: {instrument}")

    open_h, open_m = SESSION_OPEN_UTC[instrument]
    idx = df.index

    # Matcha session-start: UTC-bar vars timestamp ligger på (open_h:open_m) eller närmast efter
    is_open_bar = (idx.hour == open_h) & (idx.minute == open_m)

    # Om exakt match saknas (data-lucka), approximera: första bar inom 30 min efter open-tiden
    # Men vanligtvis är M5-data synkad på 5-min-gränser, så 13:30 finns exakt.
    session_mark = is_open_bar.astype(int)
    session_id = session_mark.cumsum() - 1  # 0-indexerat från första matchning

    # bars_in_session: för varje session-id, räkna bars från och med session_mark
    df = df.copy()
    df["session_id"] = session_id
    df["is_open_bar"] = is_open_bar

    # Inom varje session_id, räkna från 0
    df["bars_in_session"] = df.groupby("session_id").cumcount()

    # Markera bars utanför window som -1
    in_window = (
        (df["session_id"] >= 0)
        & (df["bars_in_session"] < SESSION_WINDOW_BARS)
    )
    df.loc[~in_window, "bars_in_session"] = -1

    return df


def _compute_or(df: pd.DataFrame) -> pd.DataFrame:
    """
    Beräkna OR_high / OR_low per session (max/min över första OR_BARS bars).

    OR_high/OR_low är NaN innan OR_BARS:e bar är stängd (look-ahead-säker).
    Från och med bar OR_BARS och framåt är de konstanta för resten av sessionen.
    """
    df = df.copy()
    df["or_high"] = np.nan
    df["or_low"] = np.nan

    # För varje session_id, hitta first OR_BARS bars
    for sess_id, grp in df[df["session_id"] >= 0].groupby("session_id"):
        if len(grp) < OR_BARS:
            continue
        or_bars = grp.iloc[:OR_BARS]
        or_high = or_bars["high"].max()
        or_low = or_bars["low"].min()

        # OR är känd från och med slutet av OR_BARS:te baren
        # Markera från och med bar-index OR_BARS (0-indexerat)
        active = grp.iloc[OR_BARS:]
        df.loc[active.index, "or_high"] = or_high
        df.loc[active.index, "or_low"] = or_low

    df["or_formed"] = df["or_high"].notna().astype(int)
    return df


# ── Huvudfunktion ───────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame, instrument: str, dropna: bool = True) -> pd.DataFrame:
    """
    Beräkna alla 13 features + gates för ORB.

    Args:
        df: M5 OHLCV med UTC DatetimeIndex.
        instrument: 'USTEC' | 'US500' | 'DE40'.
        dropna: om True, trimma rader där features är ogiltiga.

    Returns:
        DataFrame med features, gates, original OHLCV + session-kolumner.
    """
    out = df.copy()
    close = out["close"].astype(np.float64)
    high = out["high"].astype(np.float64)
    low = out["low"].astype(np.float64)

    # ── Session-taggning + OR ───────────────────────────────────────────
    out = _tag_sessions(out, instrument)
    out = _compute_or(out)

    # ── Time-of-day (cyklisk) ───────────────────────────────────────────
    minutes_of_day = out.index.hour * 60 + out.index.minute
    theta = 2 * np.pi * minutes_of_day / (24 * 60)
    out["time_of_day_sin"] = np.sin(theta).astype(np.float32)
    out["time_of_day_cos"] = np.cos(theta).astype(np.float32)

    # ── Minuter sedan session-öppning (normaliserad [0, 1]) ─────────────
    # När utanför session → 0
    bars_since_open = out["bars_in_session"].clip(lower=0)
    out["minutes_since_open"] = (
        (bars_since_open * 5.0) / (SESSION_WINDOW_BARS * 5.0)
    ).astype(np.float32)

    # minutes_to_close (gate): minuter kvar till session-slut
    out["minutes_to_close"] = (
        (SESSION_WINDOW_BARS - bars_since_open - 1).clip(lower=0) * 5.0
    ).astype(np.float32)

    # in_session-flagga
    out["in_session"] = (out["bars_in_session"] >= 0).astype(np.int8)

    # ── ATR ─────────────────────────────────────────────────────────────
    atr = ta.atr(high, low, close, length=ATR_PERIOD)
    out["atr_pct"] = (atr / close).astype(np.float32)

    # ── OR-distances ────────────────────────────────────────────────────
    or_high = out["or_high"]
    or_low = out["or_low"]
    atr_safe = atr.replace(0, np.nan)

    out["dist_to_or_high"] = ((close - or_high) / atr_safe).astype(np.float32)
    out["dist_to_or_low"] = ((or_low - close) / atr_safe).astype(np.float32)
    out["or_width_atr"] = ((or_high - or_low) / atr_safe).astype(np.float32)

    # Fill NaN (OR ej formad eller utanför session) med 0
    for col in ["dist_to_or_high", "dist_to_or_low", "or_width_atr"]:
        out[col] = out[col].fillna(0.0).astype(np.float32)

    # ── Relative volume ────────────────────────────────────────────────
    vol = out["volume"].astype(np.float64).clip(lower=1.0)
    vol_sma = vol.rolling(VOL_SMA_PERIOD, min_periods=VOL_SMA_PERIOD).mean()
    out["rvol_m5"] = (vol / vol_sma).astype(np.float32)

    # ── Lagged returns (1..5) ───────────────────────────────────────────
    ret = np.log(close / close.shift(1))
    for k in range(1, 6):
        out[f"ret_lag_{k}"] = ret.shift(k - 1).astype(np.float32)

    # ── Cleanup ────────────────────────────────────────────────────────
    if dropna:
        # Kräv att alla features är icke-NaN OCH att ATR är beräknad
        need = FEATURE_COLS + ["atr_pct"]
        out = out.dropna(subset=need).copy()

    return out


# ── CLI smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from casino2.loader import load_instrument

    for inst in ["USTEC", "US500", "DE40"]:
        print(f"\n{inst} M5 (val):")
        df = load_instrument(inst, "M5", split="val")
        feats = compute_features(df, instrument=inst)
        print(f"  rows: {len(feats):,}  cols: {len(feats.columns)}")

        n_in_session = int(feats["in_session"].sum())
        n_or_formed = int(feats["or_formed"].sum())
        n_sessions = int(feats["session_id"].max()) + 1 if len(feats) > 0 else 0
        print(f"  sessions: {n_sessions}  "
              f"bars in-session: {n_in_session:,}  "
              f"bars with OR: {n_or_formed:,}")

        in_sess = feats[feats["in_session"] == 1]
        if len(in_sess) > 0:
            print(f"  feature stats (in-session only):")
            print(in_sess[FEATURE_COLS].describe().T[["mean", "std", "min", "max"]].round(3))
