"""
VWAP-MR Evaluation — testa tränad SAC-modell på val/test/april-splits.

Användning:
    python -X utf8 -m casino.workers.vwap_mr.eval
    python -X utf8 -m casino.workers.vwap_mr.eval --splits test april
    python -X utf8 -m casino.workers.vwap_mr.eval --model models/vwap_mr/sac_final.zip

Metrics per (instrument, split):
    Sharpe, CAGR, MaxDD, #trades, win-rate, avg reward

Success criteria (enligt HRL_PLAN.md):
    val  : Sharpe > 0
    test : Sharpe > -0.3
    MaxDD < 5%
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from casino2.loader import INSTRUMENTS, load_instrument
from casino2.workers.vwap_mr.env import VWAPMREnv
from casino2.workers.vwap_mr.features import compute_features


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models" / "vwap_mr"


# ── Metric-beräkningar ──────────────────────────────────────────────────────

def compute_metrics(
    pnls: np.ndarray,
    positions: np.ndarray,
    rewards: np.ndarray | None = None,
    bars_per_year: int = 105_000,
) -> dict:
    """
    pnls      : per-step pnl (position_prev × log_return) — HANDELS-PnL, utan shaping
    positions : per-step position ∈ [-1, 1]
    rewards   : (optional) per-step reward (med shaping) för total-reward rapport
    bars_per_year : M5 24/7 ≈ 105k, M5 handelsdagar ≈ 72k, M15 handelsdagar ≈ 24k

    Returns dict med Sharpe/CAGR/MaxDD beräknat på PnL + per-trade metrics.
    """
    if len(pnls) == 0:
        return {
            "sharpe": 0.0, "cagr": 0.0, "mdd": 0.0, "trades": 0,
            "win_rate": 0.0, "total_pnl": 0.0, "total_reward": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0,
            "sharpe_per_trade": 0.0, "cagr_clipped": False,
        }

    # Equity på log-PnL (cum-sum av log-returns × position)
    equity = np.cumsum(pnls)
    total_log = float(equity[-1])

    # Sharpe (annualiserad på PnL-serie)
    mu = pnls.mean()
    sigma = pnls.std()
    sharpe = (mu / sigma) * np.sqrt(bars_per_year) if sigma > 1e-10 else 0.0

    # CAGR — total log-return skalat till årsbasis
    # Strängare clip (±2 = -86% … +639% CAGR) och en flagga när clip slår till
    years = len(pnls) / bars_per_year
    raw_annual_log = total_log / max(years, 1e-6)
    annual_log = float(np.clip(raw_annual_log, -2.0, 2.0))
    cagr_clipped = bool(abs(raw_annual_log) > 2.0)
    cagr = float(np.exp(annual_log) - 1.0) if years > 0 else 0.0

    # Max Drawdown i log-space → konvertera till procent
    running_max = np.maximum.accumulate(equity)
    dd_log = running_max - equity
    mdd_log = float(dd_log.max())
    mdd_pct = float(1.0 - np.exp(-mdd_log))

    # ── Trade-nivå metrics ──────────────────────────────────────────────
    # Identifiera ENTRIES (0 → ±1) och gruppera varje trade som hela perioden
    # från entry till nästa entry (eller slutet). Det fångar både holding-PnL
    # och exit-fill-PnL i samma "trade".
    pos_sign = np.sign(positions)
    pos_padded = np.concatenate([[0.0], pos_sign])
    is_entry = (pos_padded[:-1] == 0) & (pos_padded[1:] != 0)
    entry_idx = np.where(is_entry)[0]
    trades = int(len(entry_idx))

    win_rate = 0.0
    avg_win = 0.0
    avg_loss = 0.0
    profit_factor = 0.0
    sharpe_per_trade = 0.0

    if trades > 0:
        trade_pnls = []
        # Varje trade = [entry_i, entry_{i+1}) — inkluderar exit-baren samt flat-period
        # efter exit, men flat-period har pnl=0 så det påverkar inte summan.
        for i in range(len(entry_idx) - 1):
            s, e = entry_idx[i], entry_idx[i + 1]
            trade_pnls.append(float(pnls[s:e].sum()))
        # Sista trade: från sista entry till slutet
        if entry_idx[-1] < len(pnls):
            trade_pnls.append(float(pnls[entry_idx[-1]:].sum()))

        if trade_pnls:
            tpnl = np.array(trade_pnls)
            # Filtrera exakta 0-trades (t.ex. entry på sista baren)
            nonzero = tpnl[np.abs(tpnl) > 1e-12]
            if len(nonzero) > 0:
                wins = nonzero[nonzero > 0]
                losses = nonzero[nonzero < 0]
                win_rate = float(len(wins) / len(nonzero))
                avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
                avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
                gross_win = float(wins.sum())
                gross_loss = float(-losses.sum())
                profit_factor = (gross_win / gross_loss) if gross_loss > 1e-12 else np.inf

                # Per-trade Sharpe: annualiserad med sqrt(trades/år)
                trades_per_year = len(nonzero) / max(years, 1e-6)
                tpnl_mu = nonzero.mean()
                tpnl_sigma = nonzero.std()
                if tpnl_sigma > 1e-12:
                    sharpe_per_trade = float(
                        (tpnl_mu / tpnl_sigma) * np.sqrt(trades_per_year)
                    )

    return {
        "sharpe": float(sharpe),
        "sharpe_per_trade": sharpe_per_trade,
        "cagr": cagr,
        "cagr_clipped": cagr_clipped,
        "mdd": mdd_pct,
        "trades": trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "total_pnl": total_log,
        "total_reward": float(rewards.sum()) if rewards is not None else 0.0,
    }


# ── Run-evaluation ──────────────────────────────────────────────────────────

def eval_instrument_split(
    model: SAC,
    vecnorm: VecNormalize,
    instrument: str,
    split: str,
) -> dict:
    """Deterministic rollout på en (instrument, split) och rapportera metrics."""
    raw = load_instrument(instrument, "M5", split=split)
    feats = compute_features(raw)

    def _make():
        env = VWAPMREnv(
            feats,
            max_episode_bars=len(feats) - 1,
            random_start=False,
            seed=0,
        )
        return Monitor(env)

    vec = DummyVecEnv([_make])
    # Återanvänd normalizerings-stats från träningen
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, training=False)
    vec.obs_rms = vecnorm.obs_rms
    vec.ret_rms = vecnorm.ret_rms

    obs = vec.reset()
    rewards, positions, pnls = [], [], []
    done = np.array([False])

    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, info = vec.step(action)
        rewards.append(float(r[0]))
        positions.append(float(info[0].get("position", 0.0)))
        pnls.append(float(info[0].get("pnl", 0.0)))

    metrics = compute_metrics(
        pnls=np.array(pnls),
        positions=np.array(positions),
        rewards=np.array(rewards),
    )
    metrics["bars"] = len(rewards)
    return metrics


# ── CLI ─────────────────────────────────────────────────────────────────────

def format_row(inst: str, m: dict) -> str:
    sharpe = m["sharpe"]
    flag = "✅" if sharpe > 0 else ("⚠️ " if sharpe > -0.3 else "❌")
    return (
        f"  {inst:5s} {flag} "
        f"Sharpe={sharpe:+6.2f}  "
        f"CAGR={m['cagr']*100:+7.1f}%  "
        f"MDD={m['mdd']*100:5.1f}%  "
        f"Trades={m['trades']:>5d}  "
        f"Win={m['win_rate']*100:4.1f}%  "
        f"PnL={m['total_pnl']:+.3f}  "
        f"Rew={m['total_reward']:+7.1f}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default=str(MODELS_DIR / "best_model.zip"),
                    help="SAC-modell att evaluera")
    ap.add_argument("--vecnorm", default=str(MODELS_DIR / "vecnormalize.pkl"),
                    help="VecNormalize-stats")
    ap.add_argument("--splits", nargs="+",
                    default=["val", "test", "april"],
                    choices=["train", "val", "test", "april"])
    ap.add_argument("--instruments", nargs="+", default=list(INSTRUMENTS))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model_path = Path(args.model)
    vecnorm_path = Path(args.vecnorm)

    if not model_path.exists():
        print(f"❌ Modell saknas: {model_path}")
        return
    if not vecnorm_path.exists():
        print(f"❌ VecNormalize saknas: {vecnorm_path}")
        return

    print(f"\n{'='*75}")
    print(f"  VWAP-MR Evaluation")
    print(f"{'='*75}")
    print(f"  Model   : {model_path.name}")
    print(f"  VecNorm : {vecnorm_path.name}")
    print(f"  Device  : {args.device}")

    # Ladda modell + normaliseringsstats
    model = SAC.load(str(model_path), device=args.device)

    # VecNormalize.load kräver en env — bygg en dummy
    raw = load_instrument(args.instruments[0], "M5", split="val")
    dummy_feats = compute_features(raw).head(1000)
    dummy_vec = DummyVecEnv([lambda: Monitor(VWAPMREnv(dummy_feats, random_start=False))])
    vecnorm = VecNormalize.load(str(vecnorm_path), dummy_vec)

    # Evaluera per split
    all_results: Dict[str, Dict[str, dict]] = {}
    for split in args.splits:
        print(f"\n  [{split.upper()}]")
        sharpe_vals = []
        for inst in args.instruments:
            try:
                m = eval_instrument_split(model, vecnorm, inst, split)
            except Exception as e:
                print(f"  {inst:5s} ❌ FEL: {e}")
                continue
            print(format_row(inst, m))
            sharpe_vals.append(m["sharpe"])
            all_results.setdefault(split, {})[inst] = m

        if sharpe_vals:
            mean_sharpe = float(np.mean(sharpe_vals))
            verdict = "✅ OK" if mean_sharpe > 0 else ("⚠️  SVAG" if mean_sharpe > -0.3 else "❌ DÅLIG")
            print(f"  {'':5s}    Medel Sharpe: {mean_sharpe:+6.3f}  {verdict}")

    # Sammanfattning
    print(f"\n{'='*75}")
    print("  Sammanfattning")
    print(f"{'='*75}")
    for split, d in all_results.items():
        sharpes = [m["sharpe"] for m in d.values()]
        if sharpes:
            print(f"  {split:6s}: mean Sharpe = {np.mean(sharpes):+.3f} "
                  f"(min {min(sharpes):+.3f}, max {max(sharpes):+.3f})")
    print()


if __name__ == "__main__":
    main()
