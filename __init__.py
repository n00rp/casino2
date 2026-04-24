"""
Casino2 — Hierarchical RL Trading System
=========================================

Specialist-arkitektur: 4 workers + 1 meta-manager.
  workers/vwap_mr/   — VWAP Mean Reversion (SAC)
  workers/orb/       — Opening Range Breakout (QR-DQN)
  workers/ha_vol/    — Heikin Ashi Vol Expansion (QR-DQN)   [TBD]
  workers/gap/       — Gap Dynamics / Overnight Alpha (QR-DQN)   [TBD]
  manager/           — HRL Meta-agent (PPO)   [TBD]

Se HRL_PLAN.md för full specifikation.
"""
