"""
ORB Evaluation — testa tränad QR-DQN på val/test/april.

Användning:
    python -X utf8 -m casino.workers.orb.eval --splits val test april
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from sb3_contrib import QRDQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from casino2.loader import INSTRUMENTS, load_instrument
from casino2.workers.orb.env import ORBEnv
from casino2.workers.orb.features import compute_features
from casino2.workers.vwap_mr.eval import compute_metrics  # återanvänd metric-beräkningen


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models" / "orb"


def eval_instrument_split(
    model: QRDQN,
    vecnorm: VecNormalize,
    instrument: str,
    split: str,
) -> dict:
    raw = load_instrument(instrument, "M5", split=split)
    feats = compute_features(raw, instrument=instrument)

    def _make():
        env = ORBEnv(
            feats,
            max_episode_bars=len(feats) - 1,
            random_start=False,
            seed=0,
        )
        return Monitor(env)

    vec = DummyVecEnv([_make])
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(MODELS_DIR / "best_model.zip"))
    ap.add_argument("--vecnorm", default=str(MODELS_DIR / "vecnormalize.pkl"))
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
    print(f"  ORB Evaluation")
    print(f"{'='*75}")
    print(f"  Model   : {model_path.name}")
    print(f"  VecNorm : {vecnorm_path.name}")

    model = QRDQN.load(str(model_path), device=args.device)

    raw = load_instrument(args.instruments[0], "M5", split="val")
    dummy_feats = compute_features(raw, instrument=args.instruments[0]).head(1000)
    dummy_vec = DummyVecEnv([lambda: Monitor(ORBEnv(dummy_feats, random_start=False))])
    vecnorm = VecNormalize.load(str(vecnorm_path), dummy_vec)

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
