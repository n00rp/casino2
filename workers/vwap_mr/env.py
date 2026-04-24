"""
VWAP-MR Gymnasium Environment
==============================

Single-instrument M5-environment för VWAP Mean Reversion-workern.

Action-rymd: Box(-1, +1, shape=(1,))
  -1.0 = full short
   0.0 = flat
  +1.0 = full long
  kontinuerliga värden = skala-in/ut

State: 11 features enligt FEATURE_COLS i features.py

Reward per step:
    r_t = pnl_t                                   # log-return × position
        - 0.001 * |Δposition|                     # transaction cost
        + 0.5 * vwap_touch_bonus                  # close vid VWAP-återgång
        - 2.0 * breakout_3sigma_penalty           # pris bryter 3σ medan pos öppen

Gate: om adx_14 > 30 → forceras position = 0 (trend-regim)

Se casino/HRL_PLAN.md för full spec.
"""
from __future__ import annotations

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from casino2.workers.vwap_mr.features import FEATURE_COLS


# ── Konstanter ──────────────────────────────────────────────────────────────

ADX_TREND_THRESHOLD = 30.0   # Blockera handel över denna nivå
TX_COST = 0.001              # 10 bp per 100% position-change
VWAP_TOUCH_BONUS = 0.5
BREAKOUT_PENALTY = 2.0
MAX_HOLD_BARS = 40           # 3.3h på M5
VWAP_TOUCH_TOL_PCT = 0.0005  # 5 bp band runt VWAP för "touch"
OBS_CLIP = 10.0              # Clippa features till [-10, 10] för numerisk stabilitet


class VWAPMREnv(gym.Env):
    """
    Single-instrument VWAP Mean Reversion env.

    Args:
        features: DataFrame med alla FEATURE_COLS + GATE_COLS + OHLCV (från compute_features()).
        max_episode_bars: max antal steg per episod (default 2000 ≈ en vecka).
        random_start: slumpa startposition (rekommenderat för träning).
        seed: reproducerbarhet.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: pd.DataFrame,
        max_episode_bars: int = 2000,
        random_start: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()

        # Validera input
        missing = [c for c in FEATURE_COLS if c not in features.columns]
        if missing:
            raise ValueError(f"Features saknar kolumner: {missing}")
        if "close" not in features.columns or "vwap" not in features.columns:
            raise ValueError("Features måste innehålla 'close' och 'vwap'.")

        self.df = features.reset_index(drop=False).copy()  # behåll timestamp
        self.n_bars = len(self.df)
        self.max_episode_bars = min(max_episode_bars, self.n_bars - 1)
        self.random_start = random_start

        # Pre-extract numpy-arrays för snabbhet
        self._feats = self.df[FEATURE_COLS].to_numpy(dtype=np.float32)
        self._close = self.df["close"].to_numpy(dtype=np.float64)
        self._vwap = self.df["vwap"].to_numpy(dtype=np.float64)
        self._adx = self.df["adx_14"].to_numpy(dtype=np.float32)
        self._bb_u3 = self.df["bb_upper_3s"].to_numpy(dtype=np.float64)
        self._bb_l3 = self.df["bb_lower_3s"].to_numpy(dtype=np.float64)

        # Ersätt ev NaN/Inf med 0 i features (säkerhetsnät)
        self._feats = np.nan_to_num(self._feats, nan=0.0, posinf=0.0, neginf=0.0)

        # Gym-spaces
        self.observation_space = spaces.Box(
            low=-OBS_CLIP, high=OBS_CLIP,
            shape=(len(FEATURE_COLS) + 1,),  # +1 för current position
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )

        # RNG
        self._np_rng = np.random.default_rng(seed)

        # Runtime state
        self._t: int = 0
        self._start: int = 0
        self._end: int = 0
        self._position: float = 0.0
        self._bars_in_trade: int = 0

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
        self._position = 0.0
        self._bars_in_trade = 0

        return self._obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # Tolka action
        a = float(np.clip(action[0], -1.0, 1.0))

        # ADX-gate: tvinga flat vid trend-regim
        if self._adx[self._t] > ADX_TREND_THRESHOLD:
            a = 0.0

        # Max-hold gate: stäng efter MAX_HOLD_BARS
        if self._bars_in_trade >= MAX_HOLD_BARS:
            a = 0.0

        prev_pos = self._position
        new_pos = a
        dpos = new_pos - prev_pos

        # Steg fram i tid
        t_now = self._t
        t_next = self._t + 1

        # PnL: log-return från t_now → t_next på PREVIOUS position
        # (position taggad vid bar-stängning, effekten syns nästa bar)
        px_now = self._close[t_now]
        px_next = self._close[t_next]
        log_ret = float(np.log(px_next / px_now)) if px_now > 0 else 0.0
        pnl = prev_pos * log_ret

        # Transaction cost
        tx = TX_COST * abs(dpos)

        # VWAP-touch bonus: om position stängs (|new_pos| < |prev_pos|)
        # och priset är nära VWAP
        bonus = 0.0
        if abs(new_pos) < abs(prev_pos) and abs(prev_pos) > 0.1:
            dist_vwap = abs(px_now - self._vwap[t_now]) / max(self._vwap[t_now], 1e-8)
            if dist_vwap < VWAP_TOUCH_TOL_PCT:
                bonus = VWAP_TOUCH_BONUS * abs(prev_pos - new_pos)

        # Breakout-straff: 3σ-brott i fel riktning med öppen position
        penalty = 0.0
        if prev_pos > 0 and px_next > self._bb_u3[t_next]:
            penalty = BREAKOUT_PENALTY * abs(prev_pos)
        elif prev_pos < 0 and px_next < self._bb_l3[t_next]:
            penalty = BREAKOUT_PENALTY * abs(prev_pos)

        reward = pnl - tx + bonus - penalty

        # Uppdatera trade-counter
        if abs(new_pos) > 0.05 and np.sign(new_pos) == np.sign(prev_pos) and abs(prev_pos) > 0.05:
            self._bars_in_trade += 1
        elif abs(new_pos) > 0.05:
            self._bars_in_trade = 1  # ny trade
        else:
            self._bars_in_trade = 0  # flat

        self._position = new_pos
        self._t = t_next

        terminated = False
        truncated = self._t >= self._end or self._t >= self.n_bars - 1

        info = {
            "pnl": pnl,
            "tx": tx,
            "bonus": bonus,
            "penalty": penalty,
            "position": new_pos,
            "adx": float(self._adx[t_now]),
        }
        return self._obs(), float(reward), terminated, truncated, info

    # ── Helpers ─────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        x = self._feats[self._t]
        obs = np.empty(len(FEATURE_COLS) + 1, dtype=np.float32)
        obs[:-1] = np.clip(x, -OBS_CLIP, OBS_CLIP)
        obs[-1] = self._position
        return obs

    def render(self):
        pass


# ── Smoke-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from casino2.loader import load_instrument
    from casino2.workers.vwap_mr.features import compute_features

    df = load_instrument("USTEC", "M5", split="val")
    feats = compute_features(df)
    env = VWAPMREnv(feats, max_episode_bars=500, random_start=True, seed=42)

    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}  dtype: {obs.dtype}")
    print(f"Action space: {env.action_space}")

    # Kör 100 random steg
    total_r = 0.0
    for i in range(100):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        if term or trunc:
            break
    print(f"Random 100 steg: total_reward = {total_r:+.4f}")
    print(f"Sista obs (range): min={obs.min():.3f} max={obs.max():.3f}")
    print(f"Sista position: {info['position']:+.3f}")
    print("Env OK.")
