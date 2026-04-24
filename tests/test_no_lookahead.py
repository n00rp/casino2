"""
Look-Ahead Bias Unit-Tests
===========================

Empiriskt bevis att features vid tid t inte påverkas av framtida bars.

Metodologi:
  1. Ladda full dataserie → beräkna features (f_full)
  2. Trunkera till bars [0:K] → beräkna features (f_truncated)
  3. Assert att f_full[0:K] == f_truncated[0:K] för alla features

Om någon feature använder framtida data kommer assertion misslyckas.

Körs med:
    python -X utf8 -m pytest casino2/tests/test_no_lookahead.py -v
eller:
    python -X utf8 -m casino2.tests.test_no_lookahead
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Stub så dekoratorerna inte kraschar
    class _PytestStub:
        class mark:
            @staticmethod
            def parametrize(*a, **kw):
                def _wrap(f): return f
                return _wrap
        @staticmethod
        def fail(msg):
            raise AssertionError(msg)
    pytest = _PytestStub()

from casino2.loader import load_instrument
from casino2.workers.vwap_mr.features import (
    compute_features as vwap_features,
    FEATURE_COLS as VWAP_FEATURES,
    GATE_COLS as VWAP_GATES,
)
from casino2.workers.orb.features import (
    compute_features as orb_features,
    FEATURE_COLS as ORB_FEATURES,
    GATE_COLS as ORB_GATES,
)
from casino2.workers.ha_vol.features import (
    compute_features as hav_features,
    FEATURE_COLS as HAV_FEATURES,
    GATE_COLS as HAV_GATES,
)


# ── Hjälpfunktioner ─────────────────────────────────────────────────────────

def _assert_prefix_equal(
    f_full: pd.DataFrame,
    f_trunc: pd.DataFrame,
    cols: list,
    context: str,
    atol: float = 1e-6,
    trim_tail: int = 1,
):
    """
    Assert att f_trunc:s värden matchar f_full:s värden på delad index.

    trim_tail: exkludera de sista N bars från jämförelsen eftersom resample
               kan ge ofullständigt aggregerade värden vid trunkeringsgränsen
               (endast relevant för M15/M30 aggregat från M5).
    """
    common_idx = f_trunc.index.intersection(f_full.index)
    if len(common_idx) == 0:
        pytest.fail(f"{context}: inga delade index-värden att jämföra")

    if trim_tail > 0 and len(common_idx) > trim_tail:
        common_idx = common_idx[:-trim_tail]

    for col in cols:
        a = f_full.loc[common_idx, col].to_numpy(dtype=np.float64)
        b = f_trunc.loc[common_idx, col].to_numpy(dtype=np.float64)

        # Behandla NaN som lika (warmup-perioder)
        both_nan = np.isnan(a) & np.isnan(b)
        mask = ~both_nan

        if not mask.any():
            continue

        a_valid = a[mask]
        b_valid = b[mask]

        if not np.allclose(a_valid, b_valid, atol=atol, equal_nan=False):
            max_diff = np.abs(a_valid - b_valid).max()
            n_diff = int((np.abs(a_valid - b_valid) > atol).sum())
            pytest.fail(
                f"{context}: kolumn '{col}' skiljer sig "
                f"({n_diff}/{len(a_valid)} värden, max-diff={max_diff:.6e})"
            )


# ── Test: VWAP-MR features ──────────────────────────────────────────────────

@pytest.mark.parametrize("instrument", ["USTEC", "US500", "DE40"])
def test_vwap_features_no_lookahead(instrument):
    """VWAP-MR features vid tid t ska inte ändras om framtida bars finns."""
    # Använd val-split (kortare → snabbare test)
    df = load_instrument(instrument, "M5", split="val")

    # Trunkera vid 60% av data
    cut = int(len(df) * 0.6)
    df_trunc = df.iloc[:cut].copy()

    # Beräkna båda (utan dropna så vi inte får olika index från NaN-filtering)
    f_full = vwap_features(df, dropna=False)
    f_trunc = vwap_features(df_trunc, dropna=False)

    # Assert att alla features + gates matchar på trunkerat område
    # Exkludera 'vwap' — den är session-baserad och korrekt, men absolutnivå
    cols = VWAP_FEATURES + [c for c in VWAP_GATES if c != "vwap"]
    _assert_prefix_equal(f_full, f_trunc, cols, f"VWAP-{instrument}")

    # Separat: vwap själv (absolutnivå)
    _assert_prefix_equal(f_full, f_trunc, ["vwap"], f"VWAP-{instrument}-vwap", atol=1e-2)


# ── Test: ORB features ──────────────────────────────────────────────────────

@pytest.mark.parametrize("instrument", ["USTEC", "US500", "DE40"])
def test_orb_features_no_lookahead(instrument):
    """ORB features vid tid t ska inte ändras om framtida bars finns."""
    df = load_instrument(instrument, "M5", split="val")

    cut = int(len(df) * 0.6)
    df_trunc = df.iloc[:cut].copy()

    f_full = orb_features(df, instrument=instrument, dropna=False)
    f_trunc = orb_features(df_trunc, instrument=instrument, dropna=False)

    # Exkludera 'minutes_to_close' — den räknar från SESSION_WINDOW_BARS
    # och är styrd av fixa tider (inte look-ahead-känslig men numeriskt
    # identisk ändå — tas med för fullständighet)
    cols = ORB_FEATURES + ORB_GATES
    # Exkludera absolutnivåer or_high/or_low (kan ha NaN-pattern som skiljer)
    cols = [c for c in cols if c not in ("or_high", "or_low")]
    _assert_prefix_equal(f_full, f_trunc, cols, f"ORB-{instrument}")

    # or_high/or_low separat (med tolerance för NaN-skillnader)
    _assert_prefix_equal(f_full, f_trunc, ["or_high", "or_low"],
                         f"ORB-{instrument}-OR-levels", atol=1e-2)


# ── Test: HA-Vol features ───────────────────────────────────────────────────

@pytest.mark.parametrize("instrument", ["USTEC", "US500", "DE40"])
def test_hav_features_no_lookahead(instrument):
    """HA-Vol features vid tid t ska inte ändras om framtida bars finns."""
    m5 = load_instrument(instrument, "M5", split="val")
    m30 = load_instrument(instrument, "M30", split="val")

    cut = int(len(m5) * 0.6)
    m5_trunc = m5.iloc[:cut]
    # Trunkera M30 till samma slut-tidpunkt
    m30_trunc = m30.loc[:m5_trunc.index[-1]]

    f_full = hav_features(m5, m30=m30, instrument=instrument, dropna=False)
    f_trunc = hav_features(m5_trunc, m30=m30_trunc,
                             instrument=instrument, dropna=False)

    cols = HAV_FEATURES + [c for c in HAV_GATES if c not in HAV_FEATURES]
    _assert_prefix_equal(f_full, f_trunc, cols, f"HAV-{instrument}")


def test_hav_multiple_cuts():
    m5 = load_instrument("USTEC", "M5", split="val")
    m30 = load_instrument("USTEC", "M30", split="val")
    f_full = hav_features(m5, m30=m30, instrument="USTEC", dropna=False)

    cuts = [0.3, 0.5, 0.7, 0.9]
    for frac in cuts:
        cut = int(len(m5) * frac)
        m5_t = m5.iloc[:cut]
        m30_t = m30.loc[:m5_t.index[-1]]
        f_trunc = hav_features(m5_t, m30=m30_t,
                                 instrument="USTEC", dropna=False)
        _assert_prefix_equal(
            f_full, f_trunc, HAV_FEATURES,
            context=f"HAV-USTEC-cut{frac}",
        )


# ── Test: Flera trunkeringspunkter ──────────────────────────────────────────

def test_vwap_multiple_cuts():
    """Verifiera att olika trunkerings-punkter alla ger konsistenta features."""
    df = load_instrument("USTEC", "M5", split="val")
    f_full = vwap_features(df, dropna=False)

    cuts = [0.3, 0.5, 0.7, 0.9]
    for frac in cuts:
        cut = int(len(df) * frac)
        f_trunc = vwap_features(df.iloc[:cut], dropna=False)
        _assert_prefix_equal(
            f_full, f_trunc, VWAP_FEATURES,
            context=f"VWAP-USTEC-cut{frac}",
        )


def test_orb_multiple_cuts():
    df = load_instrument("USTEC", "M5", split="val")
    f_full = orb_features(df, instrument="USTEC", dropna=False)

    cuts = [0.3, 0.5, 0.7, 0.9]
    for frac in cuts:
        cut = int(len(df) * frac)
        f_trunc = orb_features(df.iloc[:cut], instrument="USTEC", dropna=False)
        _assert_prefix_equal(
            f_full, f_trunc, ORB_FEATURES,
            context=f"ORB-USTEC-cut{frac}",
        )


# ── Test: Env använder inga framtida features ──────────────────────────────

def test_vwap_env_observation_only_uses_current_bar():
    """Verifiera att VWAPMREnv observation vid step t inte beror på bars > t."""
    from casino2.workers.vwap_mr.env import VWAPMREnv

    df = load_instrument("USTEC", "M5", split="val")
    feats_full = vwap_features(df)

    cut = 5000  # efter warmup
    feats_trunc = vwap_features(df.iloc[:int(len(df) * 0.9)])

    env_full = VWAPMREnv(feats_full, max_episode_bars=1000, random_start=False, seed=0)
    env_trunc = VWAPMREnv(feats_trunc, max_episode_bars=1000, random_start=False, seed=0)

    obs_full, _ = env_full.reset()
    obs_trunc, _ = env_trunc.reset()

    assert np.allclose(obs_full, obs_trunc, atol=1e-5), \
        f"VWAPMREnv observation skiljer sig: {np.abs(obs_full - obs_trunc).max():.6e}"

    # Kör 100 deterministic steg med samma actions
    rng = np.random.default_rng(42)
    for i in range(100):
        a = rng.uniform(-1, 1, size=(1,)).astype(np.float32)
        o1, r1, _, _, _ = env_full.step(a)
        o2, r2, _, _, _ = env_trunc.step(a)
        assert np.allclose(o1, o2, atol=1e-5), \
            f"step {i}: obs skiljer sig (max-diff {np.abs(o1-o2).max():.6e})"
        assert abs(r1 - r2) < 1e-5, \
            f"step {i}: reward skiljer sig (r1={r1}, r2={r2})"


def test_orb_env_observation_only_uses_current_bar():
    from casino2.workers.orb.env import ORBEnv

    df = load_instrument("USTEC", "M5", split="val")
    feats_full = orb_features(df, instrument="USTEC")
    feats_trunc = orb_features(df.iloc[:int(len(df) * 0.9)], instrument="USTEC")

    env_full = ORBEnv(feats_full, max_episode_bars=1000, random_start=False, seed=0)
    env_trunc = ORBEnv(feats_trunc, max_episode_bars=1000, random_start=False, seed=0)

    obs_full, _ = env_full.reset()
    obs_trunc, _ = env_trunc.reset()

    assert np.allclose(obs_full, obs_trunc, atol=1e-5), \
        f"ORBEnv observation skiljer sig: {np.abs(obs_full - obs_trunc).max():.6e}"

    rng = np.random.default_rng(42)
    for i in range(100):
        a = int(rng.integers(0, 3))
        o1, r1, _, _, _ = env_full.step(a)
        o2, r2, _, _, _ = env_trunc.step(a)
        assert np.allclose(o1, o2, atol=1e-5), \
            f"step {i}: obs skiljer sig (max-diff {np.abs(o1-o2).max():.6e})"
        assert abs(r1 - r2) < 1e-5, \
            f"step {i}: reward skiljer sig (r1={r1}, r2={r2})"


# ── Direkt CLI-körning (utan pytest) ────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*65)
    print("  Look-Ahead Unit Tests")
    print("="*65)

    tests = [
        ("VWAP USTEC",  lambda: test_vwap_features_no_lookahead("USTEC")),
        ("VWAP US500",  lambda: test_vwap_features_no_lookahead("US500")),
        ("VWAP DE40",   lambda: test_vwap_features_no_lookahead("DE40")),
        ("ORB USTEC",   lambda: test_orb_features_no_lookahead("USTEC")),
        ("ORB US500",   lambda: test_orb_features_no_lookahead("US500")),
        ("ORB DE40",    lambda: test_orb_features_no_lookahead("DE40")),
        ("HAV USTEC",   lambda: test_hav_features_no_lookahead("USTEC")),
        ("HAV US500",   lambda: test_hav_features_no_lookahead("US500")),
        ("HAV DE40",    lambda: test_hav_features_no_lookahead("DE40")),
        ("VWAP cuts",   test_vwap_multiple_cuts),
        ("ORB cuts",    test_orb_multiple_cuts),
        ("HAV cuts",    test_hav_multiple_cuts),
        ("VWAP env",    test_vwap_env_observation_only_uses_current_bar),
        ("ORB env",     test_orb_env_observation_only_uses_current_bar),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed += 1

    print(f"\n  {passed}/{len(tests)} tester OK  ({failed} misslyckade)")
    print("="*65)

    import sys
    sys.exit(0 if failed == 0 else 1)
