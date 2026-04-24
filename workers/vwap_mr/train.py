"""
VWAP-MR Training — SAC på multi-instrument (USTEC + US500 + DE40).

Användning:
    python -X utf8 -m casino.workers.vwap_mr.train
    python -X utf8 -m casino.workers.vwap_mr.train --timesteps 500000 --device cuda

Output:
    models/vwap_mr/sac_best.zip
    models/vwap_mr/vecnormalize.pkl
    logs/hrl/vwap_mr/  (TensorBoard)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv, SubprocVecEnv, VecNormalize,
)

from casino2.loader import INSTRUMENTS, load_instrument
from casino2.workers.vwap_mr.env import VWAPMREnv
from casino2.workers.vwap_mr.features import compute_features


# ── Paths ───────────────────────────────────────────────────────────────────

# Relativt casino2/: workers/vwap_mr/train.py -> parents[2] = casino2/
BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models" / "vwap_mr"
LOG_DIR = BASE_DIR / "logs" / "vwap_mr"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ── Feature caching ─────────────────────────────────────────────────────────

_feature_cache: dict = {}


def get_features(instrument: str, split: str) -> pd.DataFrame:
    """Cached feature computation per (instrument, split)."""
    key = (instrument, split)
    if key not in _feature_cache:
        raw = load_instrument(instrument, "M5", split=split)
        _feature_cache[key] = compute_features(raw)
    return _feature_cache[key]


# ── Env factories ───────────────────────────────────────────────────────────

def make_train_env(instrument: str, seed: int) -> Callable:
    """Factory som returnerar en lambda för SubprocVecEnv."""
    feats = get_features(instrument, "train")
    print(f"  [{instrument}] train bars: {len(feats):,}")

    def _init():
        env = VWAPMREnv(
            feats,
            max_episode_bars=2000,
            random_start=True,
            seed=seed,
        )
        return Monitor(env)

    return _init


def make_val_env(instrument: str = "USTEC") -> VecNormalize:
    """Single-env VecNormalize för EvalCallback (USTEC val-split)."""
    feats = get_features(instrument, "val")
    print(f"  [{instrument}] val bars: {len(feats):,}")

    def _init():
        env = VWAPMREnv(
            feats,
            max_episode_bars=len(feats) - 1,  # full episod
            random_start=False,
            seed=123,
        )
        return Monitor(env)

    vec = DummyVecEnv([_init])
    vec = VecNormalize(vec, norm_obs=True, norm_reward=False, training=False)
    return vec


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--buffer-size", type=int, default=200_000)
    ap.add_argument("--eval-freq", type=int, default=10_000,
                    help="Evalueringsintervall (per env-steg)")
    ap.add_argument("--subproc", action="store_true",
                    help="Använd SubprocVecEnv (parallella processer)")
    args = ap.parse_args()

    print(f"\n{'='*65}")
    print("  VWAP-MR SAC Training")
    print(f"{'='*65}\n")
    print(f"  Timesteps : {args.timesteps:,}")
    print(f"  Device    : {args.device}")
    print(f"  Instrument: {INSTRUMENTS}")
    print(f"  Seed      : {args.seed}\n")

    # ── Bygg träningsmiljöer (1 per instrument) ────────────────────────
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

    # ── Val-miljö (USTEC) ───────────────────────────────────────────────
    print("\nBygger val-env (USTEC) ...")
    val_env = make_val_env("USTEC")
    # Sync stats från träning
    val_env.obs_rms = train_env.obs_rms
    val_env.ret_rms = train_env.ret_rms

    # ── SAC-modell ──────────────────────────────────────────────────────
    print("\nInitierar SAC ...")
    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=args.buffer_size,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[64, 32]),
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
        name_prefix="sac",
    )

    # ── Träning ────────────────────────────────────────────────────────
    print(f"\nTränar i {args.timesteps:,} timesteps ...\n")
    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=[eval_cb, ckpt_cb],
        log_interval=10,
        progress_bar=True,
    )
    dt = time.time() - t0

    # ── Spara ───────────────────────────────────────────────────────────
    final_path = MODELS_DIR / "sac_final.zip"
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
