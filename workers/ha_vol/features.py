"""
M30 HA-Supertrend — Features
==============================

Strategi: Trend-följning med Heikin Ashi Supertrend på M30.

HA används RÄTT här: som trend-filter och exit-signal (ej reversal-picker).

Signal:  ha_st_signal (+1/-1) gatad av ADX > ADX_THRESHOLD
Entry:   vid riktningsbyte (ha_st_flip = 1) under session
Exit:    vid nästa riktningsbyte (ha_st_flip i motsatt riktning)

M30 data tas direkt från load_instrument('M30') eller resamplas från M5.

Features (M30):
  1. ha_st_signal    HA-ST riktning  (-1 / +1)
  2. ha_st_conf      |ha_close - ST-linje| / (mult×ATR),  klippt 0-1
  3. ha_st_flip      Riktningsbyte från föregående bar (0/1)
  4. m30_adx_norm    M30 ADX / 100
  5. m30_atr_pct     M30 ATR / close,  klippt 0-0.05
  6. bar_return      log-avkastning per M30-bar,  klippt ±2%
  7. ha_bull_streak  konsekutiva HA-bull-bars / 10,  klippt 0-1
  8. session         EU/US aktiv session  (0/1)

Gate-kolumner (beslutsstöd för baseline/env):
  ha_st_signal, adx_trending, ha_st_flip, session

Look-ahead-audit:
  HA           : sekventiell loop, bar i beror enbart på i-1  ✓
  HA-ST        : beror på ha_close[i] + fu/fl[i-1]            ✓
  ADX          : pandas ewm bakåt                             ✓
  ha_st_flip   : direction[i] != direction[i-1] = shift(1)    ✓
  Session      : timestamp-baserat, ingen databeror           ✓
  M30 via m30  : data tas direkt (ingen shift nödvändig)      ✓
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from casino2.loader import load_instrument


# ── Parametrar ───────────────────────────────────────────────────────────────

ATR_LEN       = 10     # Wilders ATR-period för HA-ST (validerat i casino/)
ATR_MULT      = 3.0    # Supertrend-multiplikator
ADX_LEN       = 14     # ADX-period
ADX_THRESHOLD = 20.0   # ADX > 20 = trendande regime (lägre än 25 för fler signals)

# M30 bars/år: ~636k M5 / 6 / 9 år ≈ 11 800
BARS_PER_YEAR_M30 = 11_778


# ── Feature- och gate-kolumner ───────────────────────────────────────────────

FEATURE_COLS = [
    "ha_st_signal",
    "ha_st_conf",
    "ha_st_flip",
    "m30_adx_norm",
    "m30_atr_pct",
    "bar_return",
    "ha_bull_streak",
    "session",
]

GATE_COLS = [
    "ha_st_signal",
    "adx_trending",   # ADX > ADX_THRESHOLD
    "ha_st_flip",
    "session",
]


# ── Heikin Ashi OHLC (kausal sekventiell) ───────────────────────────────────

def _compute_ha_ohlc(df: pd.DataFrame) -> tuple:
    n  = len(df)
    op = df["open"].to_numpy(np.float64)
    hi = df["high"].to_numpy(np.float64)
    lo = df["low"].to_numpy(np.float64)
    cl = df["close"].to_numpy(np.float64)

    ha_cl    = (op + hi + lo + cl) / 4.0
    ha_op    = np.empty(n, np.float64)
    ha_op[0] = (op[0] + cl[0]) / 2.0
    for i in range(1, n):
        ha_op[i] = (ha_op[i - 1] + ha_cl[i - 1]) / 2.0

    ha_hi = np.maximum(hi, np.maximum(ha_op, ha_cl))
    ha_lo = np.minimum(lo, np.minimum(ha_op, ha_cl))
    return ha_op, ha_cl, ha_hi, ha_lo


# ── HA Supertrend ────────────────────────────────────────────────────────────

def _compute_ha_supertrend(
    df:       pd.DataFrame,
    atr_len:  int   = ATR_LEN,
    mult:     float = ATR_MULT,
) -> tuple:
    """
    HA-Supertrend på M30.
    Returnerar (direction, confidence, atr_arr).
      direction  : +1 (bull) / -1 (bear)
      confidence : |ha_close - ST| / (mult×ATR), klippt 0-1
    """
    ha_op, ha_cl, ha_hi, ha_lo = _compute_ha_ohlc(df)
    n = len(df)

    # Wilders ATR på HA-OHLC
    atr    = np.empty(n, np.float64)
    atr[0] = ha_hi[0] - ha_lo[0]
    for i in range(1, n):
        tr     = max(ha_hi[i] - ha_lo[i],
                     abs(ha_hi[i] - ha_cl[i - 1]),
                     abs(ha_lo[i] - ha_cl[i - 1]))
        atr[i] = (atr[i - 1] * (atr_len - 1) + tr) / atr_len

    hl2   = (ha_hi + ha_lo) / 2.0
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    fu = np.full(n, np.nan, np.float64)
    fl = np.full(n, np.nan, np.float64)
    d  = np.full(n, -1.0,  np.float64)

    for i in range(1, n):
        ub  = upper[i];  lb  = lower[i]
        pfu = fu[i - 1] if not np.isnan(fu[i - 1]) else ub
        pfl = fl[i - 1] if not np.isnan(fl[i - 1]) else lb
        pc  = ha_cl[i - 1]

        fu[i] = ub if ub < pfu or pc > pfu else pfu
        fl[i] = lb if lb > pfl or pc < pfl else pfl

        cc = ha_cl[i]
        if d[i - 1] == -1.0:
            d[i] = 1.0 if cc > fu[i] else -1.0
        else:
            d[i] = -1.0 if cc < fl[i] else 1.0

    st_line = np.where(d == 1.0, fl, fu)
    dist    = np.abs(ha_cl - st_line)
    safe_a  = np.where(atr > 0, atr, 1.0)
    conf    = np.clip(dist / (safe_a * mult), 0.0, 1.0)

    return d, conf, atr


# ── M30 ADX (Wilder's via EWM) ───────────────────────────────────────────────

def _compute_adx(df: pd.DataFrame, period: int = ADX_LEN) -> np.ndarray:
    hi  = df["high"].to_numpy(np.float64)
    lo  = df["low"].to_numpy(np.float64)
    cl  = df["close"].to_numpy(np.float64)
    n   = len(df)
    alpha = 1.0 / period

    atr14 = np.zeros(n); pdi14 = np.zeros(n); ndi14 = np.zeros(n)
    for i in range(1, n):
        pc   = cl[i - 1]
        tr   = max(hi[i] - lo[i], abs(hi[i] - pc), abs(lo[i] - pc))
        up   = hi[i] - hi[i - 1]
        dn   = lo[i - 1] - lo[i]
        pdm  = up if up > dn and up > 0 else 0.0
        ndm  = dn if dn > up and dn > 0 else 0.0
        atr14[i] = atr14[i-1] * (1 - alpha) + tr  * alpha
        pdi14[i] = pdi14[i-1] * (1 - alpha) + pdm * alpha
        ndi14[i] = ndi14[i-1] * (1 - alpha) + ndm * alpha

    with np.errstate(divide="ignore", invalid="ignore"):
        a14      = np.where(atr14 > 0, atr14, np.nan)
        plus_di  = 100 * pdi14 / a14
        minus_di = 100 * ndi14 / a14
        denom    = plus_di + minus_di
        dx       = np.where(denom > 0,
                            100 * np.abs(plus_di - minus_di) / denom, 0.0)

    adx = np.zeros(n)
    for i in range(1, n):
        adx[i] = adx[i-1] * (1 - alpha) + dx[i] * alpha

    return adx


# ── Session-filter (CET) ─────────────────────────────────────────────────────

def _session_mask(idx: pd.DatetimeIndex) -> np.ndarray:
    """EU 09:00-11:30 CET, US 15:30-18:00 CET → numpy bool array."""
    if idx.tz is None:
        idx_cet = idx.tz_localize("UTC").tz_convert("Europe/Berlin")
    else:
        idx_cet = idx.tz_convert("Europe/Berlin")
    tm = idx_cet.hour * 60 + idx_cet.minute
    return (
        ((tm >= 9 * 60) & (tm < 11 * 60 + 30)) |
        ((tm >= 15 * 60 + 30) & (tm < 18 * 60))
    )


# ── Huvud: compute_features ──────────────────────────────────────────────────

def compute_features(
    m5:         Optional[pd.DataFrame] = None,
    m30:        Optional[pd.DataFrame] = None,
    instrument: str  = "",
    dropna:     bool = True,
) -> pd.DataFrame:
    """
    Bygg M30 HA-Supertrend features.

    Datakälla (prioritetsordning):
      1. m30  — används direkt om längd ≥ 50 bars
      2. m5   — resamplas till M30 som fallback

    Returnerar DataFrame PÅ M30-FREKVENS med FEATURE_COLS + GATE_COLS + OHLCV.
    """
    # ── Välj datakälla ─────────────────────────────────────────────────
    if m30 is not None and len(m30) >= 50:
        data = m30.copy()
    elif m5 is not None and len(m5) >= 50:
        # Fallback: resampla M5 → M30
        agg = {"open": "first", "high": "max",
               "low": "min", "close": "last", "volume": "sum"}
        data = m5.resample("30min", label="right", closed="right").agg(agg).dropna()
    else:
        raise ValueError("Tillhandahåll antingen m30 (≥50 bars) eller m5 (≥50 bars).")

    if len(data) < 100:
        raise ValueError(f"M30 data för kort: {len(data)} bars (min 100)")

    n  = len(data)
    cl = data["close"].to_numpy(np.float64)

    # ── HA Supertrend ──────────────────────────────────────────────────
    ha_st, ha_conf, m30_atr = _compute_ha_supertrend(data)

    # ── HA bull streak ─────────────────────────────────────────────────
    streak = np.zeros(n, np.float64)
    cur    = 0
    for i in range(n):
        cur      = cur + 1 if ha_st[i] > 0 else 0
        streak[i] = cur
    ha_bull_streak = np.clip(streak / 10.0, 0.0, 1.0)

    # ── Riktningsbyte (flip) ───────────────────────────────────────────
    # ha_st[i-1] → ha_st[i] via shift(1): kausal ✓
    d_prev    = np.concatenate([[ha_st[0]], ha_st[:-1]])
    ha_flip   = (ha_st != d_prev).astype(np.float32)

    # ── ADX ────────────────────────────────────────────────────────────
    adx          = _compute_adx(data)
    adx_trending = (adx > ADX_THRESHOLD).astype(np.float32)

    # ── Bar log-avkastning ─────────────────────────────────────────────
    log_ret    = np.zeros(n, np.float64)
    log_ret[1:] = np.log(cl[1:] / np.where(cl[:-1] > 0, cl[:-1], 1.0))
    bar_return = np.clip(log_ret, -0.02, 0.02)

    # ── ATR-pct ────────────────────────────────────────────────────────
    safe_cl    = np.where(cl > 0, cl, 1.0)
    atr_pct    = np.clip(m30_atr / safe_cl, 0.0, 0.05)

    # ── Session ────────────────────────────────────────────────────────
    session = _session_mask(data.index).astype(np.float32)

    # ── Bygg DataFrame ─────────────────────────────────────────────────
    out = pd.DataFrame({
        # OHLCV
        "open":   data["open"],
        "high":   data["high"],
        "low":    data["low"],
        "close":  data["close"],
        "volume": data["volume"],

        # FEATURE_COLS
        "ha_st_signal":  ha_st.astype(np.float32),
        "ha_st_conf":    ha_conf.astype(np.float32),
        "ha_st_flip":    ha_flip,
        "m30_adx_norm":  np.clip(adx / 100.0, 0.0, 1.0).astype(np.float32),
        "m30_atr_pct":   atr_pct.astype(np.float32),
        "bar_return":    bar_return.astype(np.float32),
        "ha_bull_streak": ha_bull_streak.astype(np.float32),
        "session":       session,

        # GATE_COLS
        "adx_trending":  adx_trending,
    }, index=data.index)

    if dropna:
        out = out.dropna(subset=FEATURE_COLS).copy()

    return out


# ── CLI smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    for inst in ["USTEC", "US500", "DE40"]:
        print(f"\n--- {inst} ---")
        m30 = load_instrument(inst, "M30", split="val")
        f   = compute_features(m30=m30, instrument=inst)
        print(f"  M30 rows : {len(f):,}  ({f.index[0].date()} → {f.index[-1].date()})")
        bull = (f["ha_st_signal"] > 0).mean() * 100
        bear = (f["ha_st_signal"] < 0).mean() * 100
        print(f"  HA-ST    : Bull={bull:.1f}%  Bear={bear:.1f}%")
        print(f"  Trending : {f['adx_trending'].mean()*100:.1f}%  "
              f"(ADX > {ADX_THRESHOLD})")
        flips = int(f["ha_st_flip"].sum())
        years = len(f) / BARS_PER_YEAR_M30
        print(f"  Flips    : {flips} totalt  ≈ {flips/years:.0f}/år")
        active = (f["adx_trending"].astype(bool) & f["session"].astype(bool)).mean() * 100
        print(f"  Aktiv (trending+session): {active:.1f}%")
