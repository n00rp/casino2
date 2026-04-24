"""
ORB Gymnasium Environment
==========================

Diskret-action ORB-env för QR-DQN.

Action: Discrete(3): 0=Flat, 1=Long, 2=Short

Regler:
  - Handel endast tillåten när in_session=1 OCH or_formed=1
  - Utanför session → action ignoreras, position forceras till 0
  - Force-close vid minutes_to_close == 0 (session-slut)
  - Max 1 position i taget (diskret action säkerställer detta)

Reward per step:
    r_t = pnl_t                                       # log-return × pos
        - 0.0002 * |Δposition|                        # transaction cost (2 bps)
        - 0.0005 om entry utan RVOL-bekräftelse         # whipsaw-penalty (5 bps)
        + 0.0003 om entry med RVOL ≥ 1.5               # bonus (3 bps)

Semantik: entry med RVOL-bekräftelse är i praktiken gratis (bonus > tx).
         Entry utan RVOL-bekräftelse kostar 7 bps (2 tx + 5 whipsaw).

Se casino/HRL_PLAN.md för full spec.
"""
from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from casino2.workers.orb.features import FEATURE_COLS


# ── Konstanter ──────────────────────────────────────────────────────────────

# Reward-parametrar (tunade efter första tränings-omgångens "do nothing"-problem)
TX_COST = 0.0002                  # 2 bps per |Δpos| (realistisk futures-cost)
RVOL_ENTRY_BONUS = 0.0003         # 3 bps belohning för entry med RVOL≥1.5
WHIPSAW_ENTRY_PENALTY = 0.0005    # 5 bps straff för entry med RVOL<1.5
RVOL_THRESHOLD = 1.5
OBS_CLIP = 10.0

# Action-mappning
ACT_FLAT = 0
ACT_LONG = 1
ACT_SHORT = 2
ACTION_TO_POS = {ACT_FLAT: 0, ACT_LONG: 1, ACT_SHORT: -1}


class ORBEnv(gym.Env):
    """
    Single-instrument ORB env.

    Args:
        features: DataFrame med FEATURE_COLS + GATE_COLS + OHLCV.
        max_episode_bars: max antal steg per episod (default 288 = en dag).
        random_start: slumpa startposition (rekommenderat för träning).
        seed: reproducerbarhet.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: pd.DataFrame,
        max_episode_bars: int = 288,  # ~1 dag M5
        random_start: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()

        missing = [c for c in FEATURE_COLS if c not in features.columns]
        if missing:
            raise ValueError(f"Features saknar kolumner: {missing}")
        needed = ["close", "in_session", "or_formed", "minutes_to_close",
                  "or_high", "or_low", "rvol_m5"]
        missing_g = [c for c in needed if c not in features.columns]
        if missing_g:
            raise ValueError(f"Features saknar gate-kolumner: {missing_g}")

        self.df = features.reset_index(drop=False).copy()
        self.n_bars = len(self.df)
        self.max_episode_bars = min(max_episode_bars, self.n_bars - 1)
        self.random_start = random_start

        # Pre-extract numpy-arrays
        self._feats = np.nan_to_num(
            self.df[FEATURE_COLS].to_numpy(dtype=np.float32),
            nan=0.0, posinf=0.0, neginf=0.0,
        )
        self._close = self.df["close"].to_numpy(dtype=np.float64)
        self._in_session = self.df["in_session"].to_numpy(dtype=np.int8)
        self._or_formed = self.df["or_formed"].to_numpy(dtype=np.int8)
        self._min_to_close = self.df["minutes_to_close"].to_numpy(dtype=np.float32)
        self._or_high = self.df["or_high"].fillna(0.0).to_numpy(dtype=np.float64)
        self._or_low = self.df["or_low"].fillna(0.0).to_numpy(dtype=np.float64)
        self._rvol = self.df["rvol_m5"].to_numpy(dtype=np.float32)

        # Gym spaces
        self.observation_space = spaces.Box(
            low=-OBS_CLIP, high=OBS_CLIP,
            shape=(len(FEATURE_COLS) + 1,),  # +1 för current position
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # RNG
        self._np_rng = np.random.default_rng(seed)

        # Runtime state
        self._t: int = 0
        self._start: int = 0
        self._end: int = 0
        self._position: int = 0  # -1, 0, +1
        self._entry_bar: int = -1  # index där nuvarande trade öppnades

    # ── Gym API ─────────────────────────────────────────────────────────

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            self._np_rng = np.random.default_rng(seed)

        max_start = max(0, self.n_bars - self.max_episode_bars - 1)
        if self.random_start and max_start > 0:
            self._start = int(self._np_rng.integers(0, max_start))
        else:
            self._start = 0

        self._end = self._start + self.max_episode_bars
        self._t = self._start
        self._position = 0
        self._entry_bar = -1

        return self._obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        a = int(action)
        desired_pos = ACTION_TO_POS[a]

        t_now = self._t
        t_next = self._t + 1

        # ── Gates: forcera flat utanför giltig zone ──────────────────────
        can_trade = (
            self._in_session[t_now] == 1
            and self._or_formed[t_now] == 1
        )
        # Force-close vid session-slut
        is_last_bar = self._min_to_close[t_now] <= 0.0

        if not can_trade or is_last_bar:
            desired_pos = 0

        prev_pos = self._position
        dpos = desired_pos - prev_pos

        # ── PnL ──────────────────────────────────────────────────────────
        px_now = self._close[t_now]
        px_next = self._close[t_next]
        log_ret = float(np.log(px_next / px_now)) if px_now > 0 else 0.0
        pnl = prev_pos * log_ret

        # ── Transaction cost ─────────────────────────────────────────────
        tx = TX_COST * abs(dpos)

        # ── Entry-bonus / whipsaw-penalty ─────────────────────────────
        # Endast vid övergang från flat till aktiv position
        bonus = 0.0
        penalty = 0.0
        if desired_pos != 0 and prev_pos == 0:
            if self._rvol[t_now] >= RVOL_THRESHOLD:
                bonus = RVOL_ENTRY_BONUS
            else:
                penalty = WHIPSAW_ENTRY_PENALTY

        reward = pnl - tx - penalty + bonus

        # ── Uppdatera state ──────────────────────────────────────────────
        if desired_pos != 0 and prev_pos == 0:
            self._entry_bar = t_now
        elif desired_pos == 0 and prev_pos != 0:
            self._entry_bar = -1

        self._position = desired_pos
        self._t = t_next

        terminated = False
        truncated = self._t >= self._end or self._t >= self.n_bars - 1

        info = {
            "pnl": pnl,
            "tx": tx,
            "bonus": bonus,
            "penalty": penalty,
            "position": desired_pos,
            "in_session": bool(can_trade),
            "rvol": float(self._rvol[t_now]),
        }
        return self._obs(), float(reward), terminated, truncated, info

    # ── Helpers ─────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        x = self._feats[self._t]
        obs = np.empty(len(FEATURE_COLS) + 1, dtype=np.float32)
        obs[:-1] = np.clip(x, -OBS_CLIP, OBS_CLIP)
        obs[-1] = float(self._position)
        return obs

    def render(self):
        pass


# ── Smoke-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from casino2.loader import load_instrument
    from casino2.workers.orb.features import compute_features

    df = load_instrument("USTEC", "M5", split="val")
    feats = compute_features(df, instrument="USTEC")
    env = ORBEnv(feats, max_episode_bars=500, random_start=True, seed=42)

    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}  dtype: {obs.dtype}")
    print(f"Action space: {env.action_space}")

    total_r = 0.0
    n_in_sess = 0
    n_trades = 0
    prev_pos = 0
    for i in range(500):
        a = int(env.action_space.sample())
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        n_in_sess += int(info["in_session"])
        if info["position"] != prev_pos and info["position"] != 0:
            n_trades += 1
        prev_pos = info["position"]
        if term or trunc:
            break
    print(f"Random 500 steg: total_reward = {total_r:+.4f}")
    print(f"  In-session bars: {n_in_sess}  trades: {n_trades}")
    print("Env OK.")
