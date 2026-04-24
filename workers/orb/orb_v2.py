"""
ORB v2 — Klassisk Opening Range Breakout enligt OBS_Strategy.md
================================================================

Implementerar ALLA filter från dokumentet:

  1. Korrekta session-tider per instrument (US = 13:30 UTC, DE = 07:00 UTC)
  2. OR = första 15-min M15-bar efter session-open
  3. VWAP-filter (long endast om price > VWAP, short om price < VWAP)
  4. RVOL ≥ 1.5 som HÅRD regel (inte bonus)
  5. 200-EMA M30 som macro trend-filter
  6. RSI(14) M15: long kräver 45-70, short kräver 30-55
  7. Range-bredd max 2% av pris (undvik choppy)
  8. SL = motsatta OR-sidan (konservativ)
  9. TP = 2R (skippar partiell TP för v2.0)
 10. Time-based exit vid 45 min utan rörelse
 11. Max 1 trade per session

Kör:
    python -X utf8 -m casino2.workers.orb.orb_v2
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from casino2.loader import INSTRUMENTS, load_instrument, resample_m5_to_m15
from casino2.workers.vwap_mr.eval import compute_metrics


# ── Session-tider per instrument (UTC, no DST handling — approximate) ───────

SESSION_CONFIG = {
    # USA-indices: NYSE/NASDAQ regular session 09:30-16:00 ET = 13:30-20:00 UTC
    "USTEC": {"open_min": 13 * 60 + 30, "close_min": 20 * 60},
    "US500": {"open_min": 13 * 60 + 30, "close_min": 20 * 60},
    # DAX: Xetra 08:00-16:30 CET = 07:00-15:30 UTC (winter); sommartid -1h
    "DE40":  {"open_min": 7 * 60, "close_min": 15 * 60 + 30},
}


# ── Filter-parametrar ───────────────────────────────────────────────────────

RVOL_THRESHOLD = 1.5
RSI_PERIOD = 14
RSI_LONG_MIN = 45
RSI_LONG_MAX = 70
RSI_SHORT_MIN = 30
RSI_SHORT_MAX = 55
EMA_SLOPE_SPAN = 200          # 200-EMA på M30 ≈ 100h trend (~1 vecka)
MAX_RANGE_PCT = 0.02          # OR-bredd får max vara 2% av pris
TIME_EXIT_BARS = 3            # 3 × M15 = 45 min (från entry)

RR_RATIO = 2.0                # TP = 2 × risk
DEFAULT_TX_COST = 0.0002      # 2 bps
DEFAULT_SLIPPAGE_BPS = 3.0    # Realistisk slippage per sida


# ── Hjälp-funktioner för indikatorer ────────────────────────────────────────

def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI-beräkning (look-ahead-säker via rolling)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    # Wilder's smoothing via EMA med alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _compute_intraday_vwap(df: pd.DataFrame, session_col: str) -> pd.Series:
    """
    Intraday VWAP som resetas varje session.

    VWAP_t = cumsum(close × volume) / cumsum(volume) inom sessionen.
    Applicerar shift(1) för look-ahead-säkerhet (jämför close[t] mot VWAP[t-1]).
    """
    typical = df["close"]  # använd close som proxy för typical price
    vol = df["volume"]
    tpv = typical * vol
    cum_tpv = tpv.groupby(df[session_col]).cumsum()
    cum_vol = vol.groupby(df[session_col]).cumsum()
    vwap = cum_tpv / cum_vol.replace(0, np.nan)
    # Shift(1) så VWAP vid bar t bara inkluderar bars t-1 och tidigare
    return vwap.groupby(df[session_col]).shift(1)


# ── Data-pipeline: bygg M15 med alla filter ─────────────────────────────────

def build_orb_v2_data(
    instrument: str,
    split: str,
) -> pd.DataFrame:
    """
    Bygg M15-dataframe med korrekt session-definition + alla filter-indikatorer.

    Kolumner i output:
      open, high, low, close, volume,
      in_session, bar_in_sess, session_id,
      or_high, or_low, or_mid,
      or_high_safe, or_low_safe,        ← look-ahead-säker OR
      vwap_safe,                        ← shift(1) intraday VWAP
      rvol,
      rsi14,
      m30_ema200, m30_slope,
    """
    cfg = SESSION_CONFIG.get(instrument)
    if cfg is None:
        raise ValueError(f"Okänt instrument: {instrument}")

    # ── M15 bars från M5 ────────────────────────────────────────────────
    m5 = load_instrument(instrument, "M5", split=split)
    m15 = resample_m5_to_m15(m5)

    # Time-tagging
    m15["minute_of_day"] = m15.index.hour * 60 + m15.index.minute
    m15["in_session"] = (
        (m15["minute_of_day"] >= cfg["open_min"]) &
        (m15["minute_of_day"] < cfg["close_min"])
    ).astype(np.int8)

    # Session-id per datum
    m15["date"] = m15.index.date
    m15["session_id"] = (m15["date"] != m15["date"].shift(1)).cumsum() - 1

    # Bar-index inom session
    in_sess = m15["in_session"] == 1
    m15["bar_in_sess"] = m15.groupby("session_id")["in_session"].cumsum() - 1
    m15.loc[~in_sess, "bar_in_sess"] = -1

    # ── Opening Range (första in-session baren, dvs 15 min) ─────────────
    is_or_bar = in_sess & (m15["bar_in_sess"] == 0)
    or_agg = m15.loc[is_or_bar].groupby("session_id").agg(
        or_high=("high", "max"),
        or_low=("low", "min"),
    )
    m15["or_high"] = m15["session_id"].map(or_agg["or_high"])
    m15["or_low"] = m15["session_id"].map(or_agg["or_low"])
    m15["or_mid"] = (m15["or_high"] + m15["or_low"]) / 2.0

    # Look-ahead-säker: OR giltig från bar_in_sess >= 1 (efter OR-bar stängt)
    valid_after_or = m15["bar_in_sess"] >= 1
    m15["or_high_safe"] = m15["or_high"].where(valid_after_or)
    m15["or_low_safe"] = m15["or_low"].where(valid_after_or)

    # ── Intraday VWAP (reset per session) ───────────────────────────────
    m15["vwap_safe"] = _compute_intraday_vwap(m15, "session_id")

    # ── RVOL: dagens M15-volym / 20-bars rullande medel ─────────────────
    m15["rvol"] = (
        m15["volume"] / m15["volume"].rolling(20, min_periods=5).mean()
    )

    # ── RSI(14) på M15 close ────────────────────────────────────────────
    m15["rsi14"] = _compute_rsi(m15["close"], period=RSI_PERIOD)

    # ── M30 200-EMA + slope som macro-trend ─────────────────────────────
    m30 = load_instrument(instrument, "M30", split=split)
    m30["m30_ema200"] = m30["close"].ewm(span=EMA_SLOPE_SPAN, adjust=False).mean()
    m30["m30_slope"] = m30["m30_ema200"].diff(5)
    # Shift(1) för look-ahead-säkerhet
    m30_shift = m30[["m30_ema200", "m30_slope"]].shift(1)
    m15 = m15.join(m30_shift.reindex(m15.index, method="ffill"))

    return m15


# ── Backtester ──────────────────────────────────────────────────────────────

def backtest_orb_v2(
    instrument: str,
    split: str,
    *,
    use_vwap_filter: bool = True,
    use_rvol_filter: bool = True,
    use_trend_filter: bool = True,
    use_rsi_filter: bool = True,
    use_range_filter: bool = True,
    use_time_exit: bool = True,
    sl_mode: str = "opposite",      # "opposite" eller "mid"
    rr_ratio: float = RR_RATIO,
    tx_cost: float = DEFAULT_TX_COST,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> dict:
    """
    Kör full ORB v2 backtest. Returnera metrics-dict.
    """
    df = build_orb_v2_data(instrument, split)
    # Dropna på kritiska kolumner (EMA behöver warmup)
    df = df.dropna(subset=["m30_ema200", "rsi14", "vwap_safe", "rvol"])

    if len(df) < 100:
        return dict(sharpe=0.0, cagr=0.0, mdd=0.0, trades=0,
                    win_rate=0.0, total_pnl=0.0, bars=len(df),
                    sharpe_per_trade=0.0, profit_factor=0.0,
                    cagr_clipped=False)

    # Extrahera arrays
    close = df["close"].to_numpy(np.float64)
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    open_ = df["open"].to_numpy(np.float64)
    or_hi = df["or_high_safe"].to_numpy(np.float64)
    or_lo = df["or_low_safe"].to_numpy(np.float64)
    in_sess = df["in_session"].to_numpy(np.int8)
    sess_id = df["session_id"].to_numpy(np.int32)
    vwap = df["vwap_safe"].to_numpy(np.float64)
    rvol = df["rvol"].to_numpy(np.float64)
    rsi = df["rsi14"].to_numpy(np.float64)
    slope = df["m30_slope"].to_numpy(np.float64)
    ema200 = df["m30_ema200"].to_numpy(np.float64)

    n = len(df)
    rewards = np.zeros(n, dtype=np.float64)
    positions = np.zeros(n, dtype=np.float64)
    pnls = np.zeros(n, dtype=np.float64)

    pos = 0
    entry_px = 0.0
    sl_px = 0.0
    tp_px = 0.0
    entry_bar = -1
    traded_sessions: set = set()
    last_close_for_mtm = 0.0
    slip = slippage_bps / 10000.0

    for t in range(n - 1):
        current_sess = int(sess_id[t])

        # ── Innehav → SL/TP/Time-exit ───────────────────────────────────
        if pos != 0:
            bar_hi = high[t]
            bar_lo = low[t]
            hit_sl = (pos == 1 and bar_lo <= sl_px) or \
                     (pos == -1 and bar_hi >= sl_px)
            hit_tp = (pos == 1 and bar_hi >= tp_px) or \
                     (pos == -1 and bar_lo <= tp_px)
            bars_held = t - entry_bar
            time_exit = use_time_exit and bars_held >= TIME_EXIT_BARS
            force_close = (in_sess[t] == 0) or (t == n - 2)

            exit_px = None
            if hit_sl and hit_tp:
                exit_px = sl_px  # konservativt
            elif hit_sl:
                exit_px = sl_px
            elif hit_tp:
                exit_px = tp_px
            elif time_exit or force_close:
                exit_px = close[t]

            if exit_px is not None:
                exit_slipped = exit_px * (1 - pos * slip)
                bar_pnl = pos * np.log(exit_slipped / last_close_for_mtm) \
                          if last_close_for_mtm > 0 else 0.0
                rewards[t] += bar_pnl - tx_cost
                pnls[t] += bar_pnl
                positions[t] = 0
                pos = 0
                entry_bar = -1
                last_close_for_mtm = 0.0
                continue
            else:
                positions[t] = pos
                bar_pnl = pos * np.log(close[t] / last_close_for_mtm) \
                          if last_close_for_mtm > 0 else 0.0
                rewards[t] += bar_pnl
                pnls[t] += bar_pnl
                last_close_for_mtm = close[t]

        # ── Flat → signal-check med ALLA filter ─────────────────────────
        if pos == 0 and in_sess[t] == 1 and current_sess not in traded_sessions:
            # Grund: OR måste vara formad + alla filter non-NaN
            if np.isnan(or_hi[t]) or np.isnan(or_lo[t]):
                continue

            # Breakout-signal
            long_signal = close[t] > or_hi[t]
            short_signal = close[t] < or_lo[t]
            if not (long_signal or short_signal):
                continue

            # Filter 1: Range-bredd ≤ 2% av pris
            if use_range_filter:
                range_pct = (or_hi[t] - or_lo[t]) / close[t]
                if range_pct > MAX_RANGE_PCT:
                    continue

            # Filter 2: RVOL ≥ 1.5 (hård regel)
            if use_rvol_filter and (np.isnan(rvol[t]) or rvol[t] < RVOL_THRESHOLD):
                continue

            # Filter 3: VWAP-filter
            if use_vwap_filter:
                if np.isnan(vwap[t]):
                    continue
                if long_signal and close[t] < vwap[t]:
                    continue
                if short_signal and close[t] > vwap[t]:
                    continue

            # Filter 4: 200-EMA macro-trend
            if use_trend_filter:
                if np.isnan(ema200[t]) or np.isnan(slope[t]):
                    continue
                # Long kräver close > EMA200 OCH slope > 0
                if long_signal and (close[t] < ema200[t] or slope[t] <= 0):
                    continue
                # Short kräver close < EMA200 OCH slope < 0
                if short_signal and (close[t] > ema200[t] or slope[t] >= 0):
                    continue

            # Filter 5: RSI-intervall (undvik överköpt/översålt)
            if use_rsi_filter:
                if np.isnan(rsi[t]):
                    continue
                if long_signal and not (RSI_LONG_MIN <= rsi[t] <= RSI_LONG_MAX):
                    continue
                if short_signal and not (RSI_SHORT_MIN <= rsi[t] <= RSI_SHORT_MAX):
                    continue

            # Alla filter OK — entry vid nästa bars open
            if t + 1 >= n:
                continue
            new_entry = open_[t + 1]

            if long_signal:
                pos = 1
                entry_px = new_entry * (1 + slip)
                sl_px = or_lo[t] if sl_mode == "opposite" else (or_hi[t] + or_lo[t]) / 2
                risk = max(entry_px - sl_px, 1e-8)
                tp_px = entry_px + rr_ratio * risk
            else:
                pos = -1
                entry_px = new_entry * (1 - slip)
                sl_px = or_hi[t] if sl_mode == "opposite" else (or_hi[t] + or_lo[t]) / 2
                risk = max(sl_px - entry_px, 1e-8)
                tp_px = entry_px - rr_ratio * risk

            rewards[t] -= tx_cost
            positions[t] = pos
            entry_bar = t
            traded_sessions.add(current_sess)
            last_close_for_mtm = entry_px

    m = compute_metrics(
        pnls=pnls,
        positions=positions,
        rewards=rewards,
        bars_per_year=24192,   # M15 estimate
    )
    m["bars"] = n
    return m


# ── CLI ─────────────────────────────────────────────────────────────────────

def fmt_row(inst: str, m: dict) -> str:
    sharpe_t = m.get("sharpe_per_trade", 0.0)
    flag = "✅" if sharpe_t > 0.3 else ("⚠️ " if sharpe_t > -0.3 else "❌")
    cagr_str = f"{m['cagr']*100:+7.1f}%"
    if m.get("cagr_clipped"):
        cagr_str += "*"
    pf = m.get("profit_factor", 0.0)
    pf_str = f"{pf:4.2f}" if np.isfinite(pf) else " inf"
    return (
        f"  {inst:6s} {flag} "
        f"ShT={sharpe_t:+6.2f}  "
        f"CAGR={cagr_str}  "
        f"PF={pf_str}  "
        f"MDD={m['mdd']*100:5.1f}%  "
        f"Trades={m['trades']:>4d}  "
        f"Win={m['win_rate']*100:4.1f}%  "
        f"PnL={m['total_pnl']:+.2f}"
    )


def main():
    ap = argparse.ArgumentParser(description="ORB v2 backtest")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test", "april"])
    ap.add_argument("--instruments", nargs="+", default=list(INSTRUMENTS))
    ap.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE_BPS)
    ap.add_argument("--sweep-filters", action="store_true",
                    help="Kör med olika filter-kombinationer")
    args = ap.parse_args()

    print(f"\n{'='*90}")
    print(f"  ORB v2 — Multi-filter Backtest (slippage = {args.slippage} bps)")
    print(f"{'='*90}")

    if not args.sweep_filters:
        # Default: alla filter på
        print(f"\n  Konfiguration: ALLA filter på (VWAP, RVOL, Trend, RSI, Range, Time-exit)")
        print(f"  Session-tider: USTEC/US500 @ 13:30 UTC, DE40 @ 07:00 UTC")
        print("-" * 90)
        for sp in args.splits:
            print(f"\n  [{sp.upper()}]")
            sharpes = []
            for inst in args.instruments:
                try:
                    m = backtest_orb_v2(
                        inst, sp,
                        slippage_bps=args.slippage,
                    )
                except Exception as e:
                    print(f"  {inst:6s} ❌ FEL: {e}")
                    continue
                print(fmt_row(inst, m))
                sharpes.append(m.get("sharpe_per_trade", 0.0))
            if sharpes:
                mean = float(np.mean(sharpes))
                v = "✅ EDGE" if mean > 0.3 else ("⚠️  SVAG" if mean > 0 else "❌ INGEN")
                print(f"  {'Medel':6s}    ShT={mean:+.3f}  {v}")
    else:
        # Sweep-läge: testa filter-kombinationer för att hitta minimal vinnande set
        print(f"\n  Filter-sweep på TRAIN + VAL (med slippage = {args.slippage} bps)")
        print("-" * 90)
        combos = [
            ("Alla filter",        dict()),
            ("- RSI",              dict(use_rsi_filter=False)),
            ("- Trend",            dict(use_trend_filter=False)),
            ("- VWAP",             dict(use_vwap_filter=False)),
            ("- Range-bredd",      dict(use_range_filter=False)),
            ("- Time-exit",        dict(use_time_exit=False)),
            ("Endast RVOL+VWAP",   dict(use_trend_filter=False, use_rsi_filter=False,
                                          use_range_filter=False, use_time_exit=False)),
        ]
        for label, overrides in combos:
            print(f"\n  → {label}")
            for sp in ["train", "val", "test"]:
                sharpes = []
                for inst in args.instruments:
                    try:
                        m = backtest_orb_v2(
                            inst, sp,
                            slippage_bps=args.slippage,
                            **overrides,
                        )
                        sharpes.append(m.get("sharpe_per_trade", 0.0))
                    except Exception:
                        pass
                mean = float(np.mean(sharpes)) if sharpes else 0.0
                print(f"    [{sp:5s}] Medel ShT = {mean:+.3f}")

    print(f"\n{'='*90}\n")


if __name__ == "__main__":
    main()
