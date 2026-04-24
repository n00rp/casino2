"""
ORB Training — QR-DQN på multi-instrument (USTEC + US500 + DE40).

Off-policy distributional DQN passar ORB eftersom setups är sparse (1-2/dag).

Användning:
    python -X utf8 -m casino.workers.orb.train
    python -X utf8 -m casino.workers.orb.train --timesteps 2000000 --device cuda

Output:
    models/orb/best_model.zip
    models/orb/sac_final.zip         (QR-DQN, sparar som samma zip-format)
    models/orb/vecnormalize.pkl
    logs/hrl/orb/  (TensorBoard)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecNormalize,
)

from casino2.loader import INSTRUMENTS, load_instrument
from casino2.workers.orb.env import ORBEnv
from casino2.workers.orb.features import compute_features


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models" / "orb"
LOG_DIR = BASE_DIR / "logs" / "orb"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Feature caching ─────────────────────────────────────────────────────────

_feature_cache: dict = {}


def get_features(instrument: str, split: str) -> pd.DataFrame:
    key = (instrument, split)
    if key not in _feature_cache:
        raw = load_instrument(instrument, "M5", split=split)
        _feature_cache[key] = compute_features(raw, instrument=instrument)
    return _feature_cache[key]


# ── Env factories ───────────────────────────────────────────────────────────

def make_train_env(instrument: str, seed: int) -> Callable:
    feats = get_features(instrument, "train")
    print(f"  [{instrument}] train bars: {len(feats):,} "
          f"(in-session: {int(feats['in_session'].sum()):,})")

    def _init():
        env = ORBEnv(
            feats,
            max_episode_bars=576,  # ~2 dagar M5 → flera sessioner per episod
            random_start=True,
            seed=seed,
        )
        return Monitor(env)

    return _init


def make_val_env(instrument: str = "USTEC") -> VecNormalize:
    feats = get_features(instrument, "val")
    print(f"  [{instrument}] val bars: {len(feats):,}")

    def _init():
        env = ORBEnv(
            feats,
            max_episode_bars=len(feats) - 1,
            random_start=False,
            seed=123,
        )
        return Monitor(env)

    vec = DummyVecEnv([_init])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, training=False)
    return vec


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="ORB QR-DQN training")
    ap.add_argument("--timesteps", type=int, default=2_000_000)
    ap.add_argument("--device", default="cpu",
                    help="QR-DQN med små nät kör snabbare på CPU (undvik GPU-transfer)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--buffer-size", type=int, default=500_000)
    ap.add_argument("--eval-freq", type=int, default=25_000)
    ap.add_argument("--subproc", action="store_true")
    args = ap.parse_args()

    print(f"\n{'='*65}")
    print("  ORB QR-DQN Training")
    print(f"{'='*65}\n")
    print(f"  Timesteps : {args.timesteps:,}")
    print(f"  Device    : {args.device}")
    print(f"  Instrument: {INSTRUMENTS}")
    print(f"  Seed      : {args.seed}\n")

    # ── Träningsmiljöer ─────────────────────────────────────────────────
    print("Laddar features ...")
    env_fns = [
        make_train_env(inst, seed=args.seed + i)
        for i, inst in enumerate(INSTRUMENTS)
    ]

    VecCls = SubprocVecEnv if args.subproc else DummyVecEnv
    train_env = VecCls(env_fns)
    train_env = VecNormalize(
        train_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
    )

    # ── Val-env ─────────────────────────────────────────────────────────
    print("\nBygger val-env (USTEC) ...")
    val_env = make_val_env("USTEC")
    val_env.obs_rms = train_env.obs_rms
    val_env.ret_rms = train_env.ret_rms

    # ── QR-DQN ─────────────────────────────────────────────────────────
    print("\nInitierar QR-DQN ...")
    model = QRDQN(
        "MlpPolicy",
        train_env,
        learning_rate=1e-4,
        buffer_size=args.buffer_size,
        learning_starts=10_000,
        batch_size=128,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=10_000,
        exploration_fraction=0.4,        # längre utforskning (2x default)
        exploration_initial_eps=1.0,
        exploration_final_eps=0.10,      # högre golv så agenten inte låser sig
        policy_kwargs=dict(net_arch=[64, 32], n_quantiles=50),
        device=args.device,
        tensorboard_log=str(LOG_DIR),
        seed=args.seed,
        verbose=1,
    )

    # ── Callbacks ──────────────────────────────────────────────────────
    eval_cb = EvalCallback(
        val_env,
        best_model_save_path=str(MODELS_DIR),
        log_path=str(LOG_DIR / "eval"),
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=1,
        verbose=1,
    )

    ckpt_cb = CheckpointCallback(
        save_freq=max(args.timesteps // 10, 10_000),
        save_path=str(MODELS_DIR / "ckpt"),
        name_prefix="qrdqn",
    )

    # ── Träning ────────────────────────────────────────────────────────
    print(f"\nTränar i {args.timesteps:,} timesteps ...\n")
    t0 = time.time()
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[eval_cb, ckpt_cb],
            log_interval=10,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[!] KeyboardInterrupt — sparar state innan exit ...")
    dt = time.time() - t0

    # ── Spara ───────────────────────────────────────────────────────────
    final_path = MODELS_DIR / "qrdqn_final.zip"
    model.save(str(final_path))
    train_env.save(str(MODELS_DIR / "vecnormalize.pkl"))

    print(f"\n{'='*65}")
    print(f"  Klart på {dt/60:.1f} min")
    print(f"  Best model : {MODELS_DIR / 'best_model.zip'}")
    print(f"  Final model: {final_path}")
    print(f"  VecNorm    : {MODELS_DIR / 'vecnormalize.pkl'}")
    print(f"  TB logs    : tensorboard --logdir {LOG_DIR}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
