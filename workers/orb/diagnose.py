"""
ORB Diagnostic Tool
====================

Tv\u00e5 analyser f\u00f6r att diagnostisera varf\u00f6r tr\u00e4ningen inte l\u00e4r sig:

  1. Action-distribution hos nuvarande modell per in-session bar.
     \u2192 Om > 95% Flat: agenten har l\u00e5st sig p\u00e5 "do nothing".

  2. Naiv rule-based ORB-baseline.
     \u2192 Long n\u00e4r close > OR_high, Short n\u00e4r close < OR_low, force-close vid session-slut.
     Visar om det finns en edge i r\u00e5a ORB-setups alls.

Anv\u00e4ndning:
    python -X utf8 -m casino2.workers.orb.diagnose --splits val test
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from casino2.loader import INSTRUMENTS, load_instrument, resample_m5_to_m15
from casino2.workers.orb.env import ORBEnv
from casino2.workers.orb.features import compute_features
from casino2.workers.vwap_mr.eval import compute_metrics


# \u2500\u2500 Session-konstanter f\u00f6r europe+us-indices \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
# Samma som i features.py: 07:00-21:30 UTC = handels-sessionen
SESSION_START_MIN = 7 * 60         # 07:00
SESSION_END_MIN = 21 * 60 + 30     # 21:30


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models" / "orb"


# \u2500\u2500 1. Action-distribution \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def analyze_model_actions(
    model: QRDQN,
    vecnorm: VecNormalize,
    instrument: str,
    split: str,
) -> dict:
    raw = load_instrument(instrument, "M5", split=split)
    feats = compute_features(raw, instrument=instrument)

    def _make():
        return Monitor(ORBEnv(feats, max_episode_bars=len(feats) - 1,
                              random_start=False, seed=0))

    vec = DummyVecEnv([_make])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, training=False)
    vec.obs_rms = vecnorm.obs_rms
    vec.ret_rms = vecnorm.ret_rms

    obs = vec.reset()
    in_session_actions = []
    all_actions = []
    done = np.array([False])
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        a = int(action[0])
        obs, r, done, info = vec.step(action)
        all_actions.append(a)
        if info[0].get("in_session", False):
            in_session_actions.append(a)

    total = len(in_session_actions)
    if total == 0:
        return {"instrument": instrument, "split": split, "n_in_session": 0}

    counts = np.bincount(in_session_actions, minlength=3)
    return {
        "instrument": instrument,
        "split": split,
        "n_in_session": total,
        "pct_flat":  counts[0] / total * 100,
        "pct_long":  counts[1] / total * 100,
        "pct_short": counts[2] / total * 100,
    }


# \u2500\u2500 2. Naiv rule-based ORB \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def naive_orb_strategy(
    instrument: str,
    split: str,
    use_rvol_filter: bool = False,
    sl_mode: str = "opposite",
    rr_ratio: float = 2.0,
    tx_cost: float = 0.0002,
    slippage_bps: float = 3.0,   # slippage per sida (entry + exit)
) -> dict:
    """
    Klassisk rule-based ORB med SL/TP:

      Entry (vid CANDLE CLOSE utanf\u00f6r range + n\u00e4sta bars open):
        Long  om close > OR_high  (entry = n\u00e4sta bars open)
        Short om close < OR_low
        Volym-filter: RVOL \u2265 1.5 (om aktiverat)

      SL/TP:
        Long:  SL = OR_low (eller OR_mid)
               TP = entry + rr_ratio \u00d7 (entry - SL)
        Short: speglat

      Exit:
        1. Pris tr\u00e4ffar SL (loss)
        2. Pris tr\u00e4ffar TP (win)
        3. Session-slut (force-close, variabel PnL)

      Max 1 trade per session.
    """
    raw = load_instrument(instrument, "M5", split=split)
    feats = compute_features(raw, instrument=instrument)

    # Extrahera arrays
    close = feats["close"].to_numpy(dtype=np.float64)
    high = feats["high"].to_numpy(dtype=np.float64)
    low = feats["low"].to_numpy(dtype=np.float64)
    open_ = feats["open"].to_numpy(dtype=np.float64)
    in_sess = feats["in_session"].to_numpy(dtype=np.int8)
    or_formed = feats["or_formed"].to_numpy(dtype=np.int8)
    or_hi = feats["or_high"].fillna(0.0).to_numpy(dtype=np.float64)
    or_lo = feats["or_low"].fillna(0.0).to_numpy(dtype=np.float64)
    min_to_close = feats["minutes_to_close"].to_numpy(dtype=np.float32)
    rvol = feats["rvol_m5"].to_numpy(dtype=np.float32)
    sess_id = feats["session_id"].to_numpy(dtype=np.int32)

    n = len(feats)
    rewards = np.zeros(n, dtype=np.float64)
    positions = np.zeros(n, dtype=np.float64)
    pnls = np.zeros(n, dtype=np.float64)

    pos = 0                # -1, 0, +1
    entry_px = 0.0
    sl_px = 0.0
    tp_px = 0.0
    traded_sessions = set()  # sessioner som redan haft en trade
    slip = slippage_bps / 10000.0
    last_close_for_mtm = 0.0  # MTM-referens (s\u00e4tts till entry vid trade-\u00f6ppning)

    for t in range(n - 1):
        can_trade = (in_sess[t] == 1 and or_formed[t] == 1)
        current_sess = int(sess_id[t])

        # \u2500\u2500 Innehar position \u2192 kolla SL/TP p\u00e5 bar [t, t+1] range \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if pos != 0:
            bar_high = high[t]
            bar_low = low[t]
            hit_sl = (pos == 1 and bar_low <= sl_px) or \
                     (pos == -1 and bar_high >= sl_px)
            hit_tp = (pos == 1 and bar_high >= tp_px) or \
                     (pos == -1 and bar_low <= tp_px)
            force_close = min_to_close[t] <= 0 or in_sess[t] == 0

            exit_px = None
            if hit_sl and hit_tp:
                exit_px = sl_px
            elif hit_sl:
                exit_px = sl_px
            elif hit_tp:
                exit_px = tp_px
            elif force_close:
                exit_px = close[t]

            if exit_px is not None:
                exit_px_slipped = exit_px * (1 - pos * slip)
                # PnL p\u00e5 denna bar = fr\u00e5n f\u00f6rra MTM-referenspunkten till exit
                bar_pnl = pos * np.log(exit_px_slipped / last_close_for_mtm) \
                          if last_close_for_mtm > 0 else 0.0
                rewards[t] += bar_pnl - tx_cost
                pnls[t] += bar_pnl
                positions[t] = 0
                pos = 0
                last_close_for_mtm = 0.0
                continue
            else:
                positions[t] = pos
                # MTM fr\u00e5n last_close till close[t] (denna bars r\u00f6relse, ingen \u00f6verlappning)
                bar_pnl = pos * np.log(close[t] / last_close_for_mtm) \
                          if last_close_for_mtm > 0 else 0.0
                rewards[t] += bar_pnl
                pnls[t] += bar_pnl
                last_close_for_mtm = close[t]

        # \u2500\u2500 Flat \u2192 leta efter entry \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if pos == 0 and can_trade and current_sess not in traded_sessions:
            if use_rvol_filter and rvol[t] < 1.5:
                continue

            # Signal-check: close[t] utanf\u00f6r OR?
            long_signal = close[t] > or_hi[t]
            short_signal = close[t] < or_lo[t]

            if not (long_signal or short_signal):
                continue

            # Entry vid n\u00e4sta bars open
            if t + 1 >= n:
                continue
            new_entry = open_[t + 1]

            if long_signal:
                pos = 1
                entry_px = new_entry * (1 + slip)  # slippage mot oss vid entry
                if sl_mode == "opposite":
                    sl_px = or_lo[t]
                else:  # "mid"
                    sl_px = (or_hi[t] + or_lo[t]) / 2.0
                risk = max(entry_px - sl_px, 1e-8)
                tp_px = entry_px + rr_ratio * risk
            else:  # short_signal
                pos = -1
                entry_px = new_entry * (1 - slip)  # slippage mot oss vid entry
                if sl_mode == "opposite":
                    sl_px = or_hi[t]
                else:
                    sl_px = (or_hi[t] + or_lo[t]) / 2.0
                risk = max(sl_px - entry_px, 1e-8)
                tp_px = entry_px - rr_ratio * risk

            rewards[t] -= tx_cost  # entry tx
            positions[t] = pos
            traded_sessions.add(current_sess)
            last_close_for_mtm = entry_px  # MTM-referens = entry

    m = compute_metrics(
        pnls=pnls,
        positions=positions,
        rewards=rewards,
    )
    m["bars"] = n
    return m


# \u2500\u2500 3. M15 ORB med M30 trend-filter \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def _build_m15_with_trend(
    instrument: str,
    split: str,
    ema_span: int = 20,
    or_bars: int = 1,
) -> pd.DataFrame:
    """
    Bygg M15-df med OR-info + M30 EMA-slope-trend (look-ahead-s\u00e4kert).

    or_bars = antal M15-bars som bildar OR:
      1 = 15 min OR (07:00-07:15)
      2 = 30 min OR (07:00-07:30)
      4 = 60 min OR (07:00-08:00)
    """
    # \u2500\u2500 M15 bars fr\u00e5n M5 \u2500\u2500
    m5 = load_instrument(instrument, "M5", split=split)
    m15 = resample_m5_to_m15(m5)

    # Time-info
    m15["minute_of_day"] = m15.index.hour * 60 + m15.index.minute
    m15["in_session"] = (
        (m15["minute_of_day"] >= SESSION_START_MIN) &
        (m15["minute_of_day"] < SESSION_END_MIN)
    ).astype(np.int8)

    # Session-id: en ny session vid varje midnatt
    m15["date"] = m15.index.date
    m15["session_id"] = (m15["date"] != m15["date"].shift(1)).cumsum() - 1

    # Index inom sessionen (0-baserad bar-r\u00e4knare f\u00f6r in-session)
    in_sess_mask = m15["in_session"] == 1
    m15["bar_in_sess"] = m15.groupby(m15["session_id"])["in_session"].cumsum() - 1
    m15.loc[~in_sess_mask, "bar_in_sess"] = -1

    # OR-bildande bars: bar_in_sess i [0, or_bars-1]
    is_or_bar = in_sess_mask & (m15["bar_in_sess"] < or_bars)
    m15["is_or_bar"] = is_or_bar.astype(np.int8)

    # OR = max(high), min(low) \u00f6ver de f\u00f6rsta or_bars bars per session
    or_bar_data = m15.loc[is_or_bar].groupby("session_id").agg(
        or_high=("high", "max"),
        or_low=("low", "min"),
    )
    m15["or_high"] = m15["session_id"].map(or_bar_data["or_high"])
    m15["or_low"] = m15["session_id"].map(or_bar_data["or_low"])
    # Look-ahead-s\u00e4kert: OR g\u00e4ller fr\u00e5n och med bar_in_sess == or_bars
    # (dvs EFTER or_bars f\u00f6rsta bars har st\u00e4ngt)
    valid_after_or = m15["bar_in_sess"] >= or_bars
    m15["or_high_safe"] = m15["or_high"].where(valid_after_or)
    m15["or_low_safe"] = m15["or_low"].where(valid_after_or)

    # RVOL p\u00e5 M15 (20-bar medel)
    m15["rvol"] = m15["volume"] / m15["volume"].rolling(20, min_periods=5).mean()

    # \u2500\u2500 M30 trend (look-ahead-s\u00e4kert via shift(1)) \u2500\u2500
    m30 = load_instrument(instrument, "M30", split=split)
    m30["m30_ema"] = m30["close"].ewm(span=ema_span, adjust=False).mean()
    m30["m30_slope"] = m30["m30_ema"].diff(5)  # 5-bar slope = 2.5h trend
    # Shift(1): vid M15-tid t anv\u00e4nd M30-bar som st\u00e4ngde str\u00e4ngt f\u00f6re t
    m30_shift = m30[["m30_ema", "m30_slope"]].shift(1)
    m15 = m15.join(m30_shift.reindex(m15.index, method="ffill"))

    return m15


def naive_orb_m15_strategy(
    instrument: str,
    split: str,
    use_trend_filter: bool = True,
    sl_mode: str = "opposite",
    rr_ratio: float = 2.0,
    tx_cost: float = 0.0002,
    slippage_bps: float = 3.0,
    ema_span: int = 20,
    or_bars: int = 1,
) -> dict:
    """
    M15 ORB med M30 EMA-slope-trend-filter:
      - OR = f\u00f6rsta M15 bar i sessionen (07:00\u201307:15)
      - Entry efter close > or_high (long) eller close < or_low (short)
      - Long endast om m30_slope > 0
      - Short endast om m30_slope < 0
      - SL = motsatta OR-sidan
      - TP = entry \u00b1 rr_ratio \u00d7 risk
      - Max 1 trade/session
    """
    df = _build_m15_with_trend(instrument, split, ema_span=ema_span,
                                 or_bars=or_bars).copy()
    df = df.dropna(subset=["or_high_safe", "or_low_safe", "m30_slope"])

    if len(df) < 10:
        return dict(sharpe=0.0, cagr=0.0, mdd=0.0, trades=0,
                    win_rate=0.0, total_pnl=0.0, rew=0.0, bars=0)

    close = df["close"].to_numpy(np.float64)
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    open_ = df["open"].to_numpy(np.float64)
    or_hi = df["or_high_safe"].to_numpy(np.float64)
    or_lo = df["or_low_safe"].to_numpy(np.float64)
    in_sess = df["in_session"].to_numpy(np.int8)
    sess_id = df["session_id"].to_numpy(np.int32)
    slope = df["m30_slope"].to_numpy(np.float64)

    n = len(df)
    rewards = np.zeros(n, dtype=np.float64)
    positions = np.zeros(n, dtype=np.float64)
    pnls = np.zeros(n, dtype=np.float64)

    pos = 0
    entry_px = 0.0
    sl_px = 0.0
    tp_px = 0.0
    slip = slippage_bps / 10000.0
    traded_sessions: set = set()

    # Tracker: var f\u00f6rra baren som vi hade position p\u00e5? (f\u00f6r MTM-ber\u00e4kning)
    last_close_for_mtm = 0.0

    for t in range(n - 1):
        current_sess = int(sess_id[t])

        # Innehav \u2192 SL/TP-check
        if pos != 0:
            bar_hi = high[t]
            bar_lo = low[t]
            hit_sl = (pos == 1 and bar_lo <= sl_px) or (pos == -1 and bar_hi >= sl_px)
            hit_tp = (pos == 1 and bar_hi >= tp_px) or (pos == -1 and bar_lo <= tp_px)
            force_close = (in_sess[t] == 0) or (t == n - 2)

            exit_px = None
            if hit_sl and hit_tp:
                exit_px = sl_px
            elif hit_sl:
                exit_px = sl_px
            elif hit_tp:
                exit_px = tp_px
            elif force_close:
                exit_px = close[t]

            if exit_px is not None:
                # Slutlig fill mot oss
                exit_slipped = exit_px * (1 - pos * slip)
                # PnL f\u00f6r denna bar = fr\u00e5n f\u00f6rra MTM-referenspunkten till exit_slipped
                # (d\u00e5 t\u00e4cker summan av pnls[...] hela trade-returen, utan double-counting)
                bar_pnl = pos * np.log(exit_slipped / last_close_for_mtm) \
                          if last_close_for_mtm > 0 else 0.0
                rewards[t] += bar_pnl - tx_cost
                pnls[t] += bar_pnl
                positions[t] = 0
                pos = 0
                last_close_for_mtm = 0.0
                continue
            else:
                positions[t] = pos
                # MTM fr\u00e5n last_close till close[t]  (denna bars r\u00f6relse)
                bar_pnl = pos * np.log(close[t] / last_close_for_mtm) \
                          if last_close_for_mtm > 0 else 0.0
                rewards[t] += bar_pnl
                pnls[t] += bar_pnl
                last_close_for_mtm = close[t]

        # Flat \u2192 signal-check
        if pos == 0 and in_sess[t] == 1 and current_sess not in traded_sessions:
            if np.isnan(or_hi[t]) or np.isnan(or_lo[t]) or np.isnan(slope[t]):
                continue

            long_signal = close[t] > or_hi[t]
            short_signal = close[t] < or_lo[t]

            # Trend-filter
            if use_trend_filter:
                if long_signal and slope[t] <= 0:
                    continue
                if short_signal and slope[t] >= 0:
                    continue

            if not (long_signal or short_signal):
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
            traded_sessions.add(current_sess)
            # S\u00e4tt MTM-referens till entry (s\u00e5 f\u00f6rsta MTM-baren r\u00e4knar fr\u00e5n entry)
            last_close_for_mtm = entry_px

    # M15 har 96 bars/dag \u2192 \u00e5r \u2248 96 * 252 = 24192
    m = compute_metrics(
        pnls=pnls,
        positions=positions,
        rewards=rewards,
        bars_per_year=24192,
    )
    m["bars"] = n
    return m


# \u2500\u2500 Pretty print \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def fmt_metrics(m: dict) -> str:
    """Visa b\u00e5de per-bar Sharpe och per-trade Sharpe samt robusta metrics."""
    sharpe_t = m.get("sharpe_per_trade", 0.0)
    flag = "\u2705" if sharpe_t > 0 else ("\u26a0\ufe0f " if sharpe_t > -0.3 else "\u274c")
    cagr_str = f"{m['cagr']*100:+7.1f}%"
    if m.get("cagr_clipped", False):
        cagr_str += "*"  # markera att v\u00e4rdet var clippat
    pf = m.get("profit_factor", 0.0)
    pf_str = f"{pf:4.2f}" if np.isfinite(pf) else " inf"
    return (
        f"{flag} "
        f"ShT={sharpe_t:+5.2f}  "      # per-trade Sharpe (robustare)
        f"CAGR={cagr_str}  "
        f"PF={pf_str}  "                  # profit factor
        f"MDD={m['mdd']*100:5.1f}%  "
        f"Trades={m['trades']:>5d}  "
        f"Win={m['win_rate']*100:4.1f}%  "
        f"PnL={m['total_pnl']:+.2f}"
    )


# \u2500\u2500 Main \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS_DIR / "best_model.zip"))
    ap.add_argument("--vecnorm", default=str(MODELS_DIR / "vecnormalize.pkl"))
    ap.add_argument("--splits", nargs="+", default=["val", "test"],
                    choices=["train", "val", "test", "april"])
    ap.add_argument("--instruments", nargs="+", default=list(INSTRUMENTS))
    ap.add_argument("--skip-model", action="store_true",
                    help="Hoppa \u00f6ver modell-analysen, k\u00f6r endast naiv baseline")
    args = ap.parse_args()

    print(f"\n{'='*80}")
    print("  ORB Diagnostic")
    print(f"{'='*80}")

    # \u2500\u2500 1. Modellens action-distribution \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if not args.skip_model:
        model_path = Path(args.model)
        vecnorm_path = Path(args.vecnorm)
        if not model_path.exists() or not vecnorm_path.exists():
            print(f"\u26a0\ufe0f  Modell saknas, hoppar \u00f6ver modell-analys")
            args.skip_model = True

    if not args.skip_model:
        print("\n[1] Tr\u00e4nade modellens action-distribution (in-session)")
        print("-" * 80)

        model = QRDQN.load(str(args.model), device="cpu")
        raw = load_instrument(args.instruments[0], "M5", split="val")
        dummy = compute_features(raw, instrument=args.instruments[0]).head(1000)
        dummy_vec = DummyVecEnv([lambda: Monitor(ORBEnv(dummy, random_start=False))])
        vecnorm = VecNormalize.load(str(args.vecnorm), dummy_vec)

        for split in args.splits:
            print(f"\n  [{split.upper()}]")
            for inst in args.instruments:
                r = analyze_model_actions(model, vecnorm, inst, split)
                if r["n_in_session"] == 0:
                    print(f"    {inst:6s} \u26a0\ufe0f  inga in-session bars")
                    continue
                flag = "\u274c DO-NOTHING" if r["pct_flat"] > 95 else (
                    "\u26a0\ufe0f  MYCKET PASSIV" if r["pct_flat"] > 80 else "\u2705 AKTIV"
                )
                print(
                    f"    {inst:6s} n={r['n_in_session']:>5d}  "
                    f"Flat={r['pct_flat']:5.1f}%  "
                    f"Long={r['pct_long']:5.1f}%  "
                    f"Short={r['pct_short']:5.1f}%  {flag}"
                )

    # \u2500\u2500 2. Robusthetstest: slippage-sweep + alla splits \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    print("\n[2] Robusthetstest: R:R=1:2, SL=opposite, slippage-sweep \u00f6ver ALLA splits")
    print("    (om edge f\u00f6rsvinner vid realistisk slippage \u2192 ej tradebar)")
    print("-" * 80)

    # Standard-variant (b\u00e4st fr\u00e5n f\u00f6rra testet)
    fixed = dict(use_rvol_filter=False, sl_mode="opposite", rr_ratio=2.0)
    all_splits = ["train", "val", "test", "april"]
    slippage_levels = [0.0, 2.0, 5.0, 10.0]  # bps

    # Resultat-matris: [split][slip] -> medel Sharpe
    print(f"\n  {'Slippage':<12s}", end="")
    for sp in all_splits:
        print(f"{sp.upper():>10s}", end="")
    print()
    print(f"  {'':<12s}", end="")
    for sp in all_splits:
        print(f"{'------':>10s}", end="")
    print()

    for slip_bps in slippage_levels:
        print(f"  {slip_bps:>3.1f} bps     ", end="")
        for sp in all_splits:
            sharpes = []
            for inst in args.instruments:
                try:
                    m = naive_orb_strategy(
                        inst, sp,
                        slippage_bps=slip_bps,
                        **fixed,
                    )
                    sharpes.append(m["sharpe_per_trade"])
                except Exception:
                    pass
            if sharpes:
                mean = float(np.mean(sharpes))
                print(f"{mean:>+10.3f}", end="")
            else:
                print(f"{'--':>10s}", end="")
        print()

    # \u2500\u2500 3. M15 ORB OR-bredd-sweep (realistisk slippage = 3 bps) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    print(f"\n[3] M15 ORB: OR-bredd-sweep @ 3 bps slippage, R:R=1:2")
    print("    (bredare OR = f\u00e4rre men renare breakouts)")
    print("-" * 80)

    or_widths = [
        (1, "15-min OR (1 bar)"),
        (2, "30-min OR (2 bars)"),
        (4, "60-min OR (4 bars)"),
        (6, "90-min OR (6 bars)"),
    ]

    for tf_on, tf_label in [(False, "Ingen trend-filter"), (True, "Med trend-filter")]:
        print(f"\n  \u2192 {tf_label}")
        print(f"  {'OR-bredd':<22s}", end="")
        for sp in all_splits:
            print(f"{sp.upper():>10s}", end="")
        print()
        for or_bars, label in or_widths:
            print(f"  {label:<22s}", end="")
            for sp in all_splits:
                sharpes = []
                for inst in args.instruments:
                    try:
                        m = naive_orb_m15_strategy(
                            inst, sp,
                            use_trend_filter=tf_on,
                            rr_ratio=2.0,
                            slippage_bps=3.0,
                            or_bars=or_bars,
                        )
                        sharpes.append(m["sharpe_per_trade"])
                    except Exception:
                        pass
                if sharpes:
                    mean = float(np.mean(sharpes))
                    print(f"{mean:>+10.3f}", end="")
                else:
                    print(f"{'--':>10s}", end="")
            print()

    # \u2500\u2500 4. Slippage-sweep f\u00f6r b\u00e4sta M15-variant (15-min OR, no trend) \u2500\u2500\u2500\u2500\u2500\u2500
    print(f"\n\n[4] Slippage-sweep: M15 15-min OR, no trend, R:R=1:2")
    print("    (om edge kvar vid h\u00f6gre slippage \u2192 \u00e4kta signal)")
    print("-" * 80)
    print(f"  {'Slippage':<12s}", end="")
    for sp in all_splits:
        print(f"{sp.upper():>10s}", end="")
    print()
    for slip in [0.0, 2.0, 3.0, 5.0, 8.0, 12.0]:
        print(f"  {slip:>4.1f} bps    ", end="")
        for sp in all_splits:
            sharpes = []
            for inst in args.instruments:
                try:
                    m = naive_orb_m15_strategy(
                        inst, sp,
                        use_trend_filter=False,
                        rr_ratio=2.0,
                        slippage_bps=slip,
                        or_bars=1,
                    )
                    sharpes.append(m["sharpe_per_trade"])
                except Exception:
                    pass
            if sharpes:
                print(f"{float(np.mean(sharpes)):>+10.3f}", end="")
            else:
                print(f"{'--':>10s}", end="")
        print()

    # \u2500\u2500 5. Detalj med riktigt konservativ slippage 5 bps \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    print(f"\n\n[5] Detalj: 15-min OR, no trend @ 5 bps slippage (konservativ)")
    print("    PnL i log-return, CAGR i %, Sharpe annualiserad")
    print("-" * 80)
    for sp in all_splits:
        print(f"\n  [{sp.upper()}]")
        sharpes = []
        for inst in args.instruments:
            try:
                m = naive_orb_m15_strategy(
                    inst, sp,
                    use_trend_filter=False,
                    rr_ratio=2.0,
                    slippage_bps=5.0,
                    or_bars=1,
                )
            except Exception as e:
                print(f"    {inst:6s} \u274c FEL: {e}")
                continue
            print(f"    {inst:6s} {fmt_metrics(m)}")
            sharpes.append(m["sharpe_per_trade"])
        if sharpes:
            mean = float(np.mean(sharpes))
            verdict = "\u2705 EDGE" if mean > 0.3 else (
                "\u26a0\ufe0f  SVAG" if mean > 0 else "\u274c INGEN"
            )
            print(f"    {'':6s}    Medel: {mean:+.3f}  {verdict}")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
