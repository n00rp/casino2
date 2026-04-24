"""HA-Vol Worker — Heikin Ashi Volume Expansion trend-specialist.

Strategi:
  - M15 Heikin Ashi candles för trendsmoothing
  - Volume expansion (RVOL > 1.5) som confirmation
  - M30 EMA-trend + ADX för regim-filter
  - Entry efter N konsekutiva HA-candles i samma riktning
  - ATR-baserad SL/TP

Utveckling i stegordning:
  1. features.py  — Look-ahead-säker feature-pipeline
  2. baseline.py  — Rule-based backtester (validering av edge)
  3. env.py       — Gymnasium env (om baseline visar edge)
  4. train.py     — QR-DQN träning

Se casino2/HRL_PLAN.md för full spec.
"""
