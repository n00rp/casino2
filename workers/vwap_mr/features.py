"""
VWAP-MR Feature Engineering
============================

Bygger 11 features + ADX-gate + reward-hjälpkolumner för VWAP Mean Reversion-workern.

Alla beräkningar är look-ahead-säkra:
  - VWAP: session-baserad (reset dagligen kl 00:00 UTC), cumulativ
  - Bollinger Bands: rolling, endast historisk data
  - RSI/MACD/ATR: standard pandas_ta-implementationer
  - Lags: shift(k) där k ≥ 1

Output (läggs till df):
  Features (state):
    dist_vwap_pct, bb_upper_dist, bb_lower_dist, rsi_2, macd_hist,
    atr_pct, volume_ratio, ret_lag_1..4

  Gate-flags (används av env, inte policy):
    adx_14              — blockera om > 30
    vwap                — absolutnivå (för reward: touch-bonus)
    bb_upper, bb_lower  — 2σ-band (för reward)
    bb_upper_3s, bb_lower_3s  — 3σ-band (för straff)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_ta as ta


# ── Konstanter ──────────────────────────────────────────────────────────────

BB_PERIOD = 20
BB_STD_MAIN = 2.0
BB_STD_EXTREME = 3.0

RSI_PERIOD = 2          # Fast RSI enligt HRL_PLAN.md
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

ATR_PERIOD = 14
ADX_PERIOD = 14
VOL_SMA_PERIOD = 20

FEATURE_COLS = [
    "dist_vwap_pct",
    "bb_upper_dist",
    "bb_lower_dist",
    "rsi_2",
    "macd_hist",
    "atr_pct",
    "volume_ratio",
    "ret_lag_1",
    "ret_lag_2",
    "ret_lag_3",
    "ret_lag_4",
]

GATE_COLS = [
    "adx_14",
    "vwap",
    "bb_upper",
    "bb_lower",
    "bb_upper_3s",
    "bb_lower_3s",
]


# ── VWAP (session-daglig) ───────────────────────────────────────────────────

def _session_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Daglig cumulativ VWAP som resettar kl 00:00 UTC.

    Look-ahead-säker: vid bar t är VWAP(t) = Σ(typical_price * vol) / Σ(vol)
    över barer (session_open, t] i samma UTC-dag.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(np.float64).clip(lower=1.0)

    day = df.index.floor("D")  # UTC-dag
    tpv = typical * vol

    cum_tpv = tpv.groupby(day).cumsum()
    cum_vol = vol.groupby(day).cumsum()

    vwap = cum_tpv / cum_vol
    return vwap.astype(np.float32)


# ── Huvudfunktion ───────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """
    Beräkna alla features + gates för VWAP-MR-workern.

    Args:
        df: M5 OHLCV med DatetimeIndex (UTC).
        dropna: om True, trimma warmup-perioden.

    Returns:
        DataFrame med alla FEATURE_COLS + GATE_COLS + original OHLCV.
    """
    out = df.copy()
    close = out["close"].astype(np.float64)
    high = out["high"].astype(np.float64)
    low = out["low"].astype(np.float64)

    # ── VWAP ────────────────────────────────────────────────────────────
    vwap = _session_vwap(out)
    out["vwap"] = vwap
    out["dist_vwap_pct"] = ((close - vwap) / vwap).astype(np.float32)

    # ── Bollinger Bands (2σ + 3σ) — manuell för att undvika pandas_ta-bugg ─
    bb_ma = close.rolling(BB_PERIOD, min_periods=BB_PERIOD).mean()
    bb_sd = close.rolling(BB_PERIOD, min_periods=BB_PERIOD).std(ddof=0)
    bb_upper = bb_ma + BB_STD_MAIN * bb_sd
    bb_lower = bb_ma - BB_STD_MAIN * bb_sd
    bb_upper_3s = bb_ma + BB_STD_EXTREME * bb_sd
    bb_lower_3s = bb_ma - BB_STD_EXTREME * bb_sd

    out["bb_upper"] = bb_upper.astype(np.float32)
    out["bb_lower"] = bb_lower.astype(np.float32)
    out["bb_upper_3s"] = bb_upper_3s.astype(np.float32)
    out["bb_lower_3s"] = bb_lower_3s.astype(np.float32)

    out["bb_upper_dist"] = ((close - bb_upper) / close).astype(np.float32)
    out["bb_lower_dist"] = ((bb_lower - close) / close).astype(np.float32)

    # ── RSI(2) ──────────────────────────────────────────────────────────
    rsi = ta.rsi(close, length=RSI_PERIOD)
    # Normalisera till [-1, 1]: (rsi - 50) / 50
    out["rsi_2"] = ((rsi - 50.0) / 50.0).astype(np.float32)

    # ── MACD histogram ──────────────────────────────────────────────────
    macd_df = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    macd_hist = macd_df[f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"]
    # Normalisera mot ATR för skala-invarians
    atr_tmp = ta.atr(high, low, close, length=ATR_PERIOD)
    out["macd_hist"] = (macd_hist / atr_tmp.replace(0, np.nan)).astype(np.float32)

    # ── ATR % ────────────────────────────────────────────────────────────
    out["atr_pct"] = (atr_tmp / close).astype(np.float32)

    # ── ADX (gate) ──────────────────────────────────────────────────────
    adx_df = ta.adx(high, low, close, length=ADX_PERIOD)
    out["adx_14"] = adx_df[f"ADX_{ADX_PERIOD}"].astype(np.float32)

    # ── Volume ratio ────────────────────────────────────────────────────
    vol = out["volume"].astype(np.float64).clip(lower=1.0)
    vol_sma = vol.rolling(VOL_SMA_PERIOD, min_periods=VOL_SMA_PERIOD).mean()
    out["volume_ratio"] = (vol / vol_sma).astype(np.float32)

    # ── Lagged returns (1..4) ───────────────────────────────────────────
    ret = np.log(close / close.shift(1))
    for k in range(1, 5):
        out[f"ret_lag_{k}"] = ret.shift(k - 1).astype(np.float32)
    # ret_lag_1 = ret vid förra baren → motsvarar informations-set vid t

    # ── Trimma warmup ───────────────────────────────────────────────────
    if dropna:
        out = out.dropna(subset=FEATURE_COLS + GATE_COLS).copy()

    return out


# ── CLI-test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from casino2.loader import load_instrument

    for inst in ["USTEC", "US500", "DE40"]:
        print(f"\n{inst} M5:")
        df = load_instrument(inst, "M5", split="val")
        feats = compute_features(df)
        print(f"  rows: {len(feats):,}  cols: {len(feats.columns)}")
        print(f"  feature stats:")
        print(feats[FEATURE_COLS].describe().T[["mean", "std", "min", "max"]].round(4))
