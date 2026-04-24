# Casino2 — Hierarchical RL Trading System

A hierarchical reinforcement learning system for multi-instrument algorithmic trading, featuring 4 specialist RL workers and 1 meta-manager. The system trades USTEC, US500, and DE40 indices using market microstructure-based strategies.

## Overview

Casino2 implements a **Hierarchical Reinforcement Learning (HRL)** architecture where:
- **Workers**: 4 specialist agents trained on specific market regimes (VWAP Mean Reversion, Opening Range Breakout, Heikin Ashi Volatility, Gap Dynamics)
- **Manager**: 1 meta-agent that selects which worker controls capital per 30-minute window based on market regime

The system is designed with strict **look-ahead bias protection**, multi-timeframe feature engineering, and session-based trading logic.

## Architecture

```
                    ┌──────────────────┐
                    │  HRL Manager     │  (M30 features + regime)
                    │  action ∈ {0..3} │
                    └────────┬─────────┘
                             │ delegates
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
     ┌────────┐        ┌─────────┐        ┌──────────┐     ┌──────┐
     │  ORB   │        │ VWAP-MR │        │  HA-Vol  │     │ Gap  │
     │  (M5)  │        │  (M5)   │        │  (M15)   │     │(M15) │
     └────────┘        └─────────┘        └──────────┘     └──────┘
```

## Project Structure

```
casino2/
├── HRL_PLAN.md                  # Full architecture spec (state/action/reward per worker)
├── README.md                    # This file
├── loader.py                    # Data pipeline (CSV → parquet, splits)
├── data/
│   └── parquet/                 # Cache: {instr}_{tf}.parquet
├── workers/
│   ├── vwap_mr/                 # VWAP Mean Reversion (SAC) ✅ trained
│   │   ├── features.py
│   │   ├── env.py
│   │   ├── train.py
│   │   └── eval.py
│   ├── orb/                     # Opening Range Breakout (QR-DQN) 🏗️ implemented, untrained
│   ├── ha_vol/                  # Heikin Ashi Vol Expansion (QR-DQN) 📋 TBD
│   └── gap/                     # Gap Dynamics / Overnight Alpha (QR-DQN) 📋 TBD
├── manager/                     # HRL Meta-agent (PPO) 📋 TBD
├── models/
│   ├── vwap_mr/
│   │   ├── best_model.zip
│   │   ├── vecnormalize.pkl
│   │   └── ckpt/
│   └── orb/
├── logs/                        # TensorBoard logs
│   ├── vwap_mr/
│   └── orb/
└── tests/
    └── test_no_lookahead.py    # Empirical look-ahead bias verification
```

## Tech Stack

- **Python 3.8+**
- **RL Frameworks**:
  - `stable-baselines3` (SAC, PPO)
  - `sb3-contrib` (QR-DQN)
- **Environments**: `gymnasium`
- **Data**: `pandas`, `numpy`, `pyarrow` (parquet)
- **Visualization**: `tensorboard`
- **Testing**: `pytest`

## Installation

### Prerequisites

1. Python 3.8 or higher
2. Virtual environment (recommended)
3. CUDA-capable GPU (optional, for SAC training)

### Setup

```bash
# Clone the repository
cd OpenClaw_RL/

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install stable-baselines3 sb3-contrib gymnasium pandas numpy pyarrow tensorboard pytest
```

## Data Requirements

CSV data sources should be located at:
```
OpenClaw_RL/data/april2026/2017-2026/
  USTEC_M5_*.csv, USTEC_M30_*.csv
  US500_M5_*.csv, US500_M30_*.csv
  DE40_M5_*.csv,  DE40_M30_*.csv
```

CSV format (MT5 tab-separated):
- Columns: `<DATE>`, `<TIME>`, `<OPEN>`, `<HIGH>`, `<LOW>`, `<CLOSE>`, `<TICKVOL>`, `<VOL>`, `<SPREAD>`
- Timezone: UTC

## Quick Start

All commands run from `OpenClaw_RL/` with venv activated.

### 1. Prepare Data (First Time Only)

```powershell
# Convert CSV to parquet cache (runs once)
.\venv\Scripts\python.exe -X utf8 -m casino2.loader --prepare

# Verify cache
.\venv\Scripts\python.exe -X utf8 -m casino2.loader --check
```

### 2. Run Look-Ahead Bias Tests (Critical)

Always run before training to ensure no data leakage:

```powershell
.\venv\Scripts\python.exe -X utf8 -m casino2.tests.test_no_lookahead
```

### 3. Train Workers

#### VWAP-MR (SAC - Continuous Action)

```powershell
# Full training (5M steps, ~5h on GPU)
.\venv\Scripts\python.exe -X utf8 -m casino2.workers.vwap_mr.train --timesteps 5000000 --device cuda

# Quick test (100K steps)
.\venv\Scripts\python.exe -X utf8 -m casino2.workers.vwap_mr.train --timesteps 100000 --device cpu
```

#### ORB (QR-DQN - Discrete Action)

```powershell
# Full training (2M steps, ~10 min on CPU)
.\venv\Scripts\python.exe -X utf8 -m casino2.workers.orb.train --timesteps 2000000 --device cpu

# Quick test
.\venv\Scripts\python.exe -X utf8 -m casino2.workers.orb.train --timesteps 100000 --device cpu
```

### 4. Evaluate Workers

```powershell
# VWAP-MR evaluation
.\venv\Scripts\python.exe -X utf8 -m casino2.workers.vwap_mr.eval --splits val test april

# ORB evaluation
.\venv\Scripts\python.exe -X utf8 -m casino2.workers.orb.eval --splits val test april
```

### 5. Monitor Training with TensorBoard

```powershell
.\venv\Scripts\python.exe -m tensorboard --logdir casino2/logs
```

## Worker Specifications

### Worker 1: VWAP Mean Reversion (SAC)

- **Timeframe**: M5
- **Algorithm**: SAC (Soft Actor-Critic)
- **Action Space**: Continuous `Box(-1, +1, shape=(1,))` - position sizing
- **Features** (11): Distance to VWAP, Bollinger Bands, RSI, MACD, ATR, volume ratio, returns
- **Reward**: Log return - transaction cost + VWAP touch bonus - breakout penalty
- **Gates**: Blocked when ADX > 30 (trending market), max hold 40 bars
- **Status**: ✅ Trained (5M steps)

### Worker 2: Opening Range Breakout (QR-DQN)

- **Timeframe**: M5 (with M15/M30 range context)
- **Algorithm**: QR-DQN (Quantile Regression DQN)
- **Action Space**: Discrete(3) - Flat, Long, Short
- **Features** (13): Time of day, distance to OR high/low, OR width, RVOL, ATR, returns
- **Reward**: Log return - transaction cost - whipsaw penalty + volume confirmation bonus
- **Rules**: Active only during session opening (first 120 min), max 1 position
- **Status**: 🏗️ Implemented, untrained

### Worker 3: Heikin Ashi Volatility Expansion (QR-DQN)

- **Timeframe**: M15
- **Algorithm**: QR-DQN
- **Action Space**: Discrete(5) - Flat, Long-0.5, Long-1.0, Short-0.5, Short-1.0
- **Features** (14): HA OHLC, body size, wicks, streak, BB width, ATR, volume
- **Reward**: Asymmetric - rewards holding in trend, penalizes trading in compression
- **Rules**: Trade only if streak ≥ 2 bars, blocked in compression
- **Status**: 📋 TBD

### Worker 4: Gap Dynamics (QR-DQN)

- **Timeframe**: M15 (with M5 volume confirmation)
- **Algorithm**: QR-DQN
- **Action Space**: Discrete(3) - Ignore, Gap-and-Go, Fade-the-Gap
- **Features** (9): Gap size, direction, prev day trend, RVOL, time since open
- **Reward**: Log return + exhaustion fade bonus + breakaway go bonus
- **Rules**: Active first 60 min of session, 1 trade/day max, requires |gap| > 0.3%
- **Status**: 📋 TBD

### HRL Manager (PPO)

- **Timeframe**: M30 (updates every 30 min)
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Action Space**: Discrete(5) - {Flat, ORB, VWAP-MR, HA-Vol, Gap}
- **Features** (22): Regime indicators (ADX, ATR, BB width, VIX proxy), worker signals, worker PnL
- **Reward**: Portfolio return - action change penalty - drawdown penalty + sortino bonus
- **Rules**: Switch cooldown 4 bars (2h), Kelly sizing at portfolio level
- **Status**: 📋 TBD

## Data Splits

- **Train**: 2017-01 → 2024-12
- **Validation**: 2025-01 → 2025-10
- **Test**: 2025-11 → 2026-03
- **April26**: Forward-only sanity test (2026-04 onwards)

## Success Criteria

Before a worker is integrated into the HRL Manager:
- ✅ Sharpe > 0 on validation set
- ✅ Sharpe > -0.3 on test set (tolerance for regime shift)
- ✅ Maximum drawdown < 5%
- ✅ Reasonable trade count (not 3x baseline)

## Current Status

| Worker | Algorithm | Trained | Val Sharpe | Test Sharpe | April Sharpe |
|--------|-----------|---------|------------|-------------|--------------|
| VWAP-MR | SAC | ✅ 5M steps | **+1.78** | **+3.39** | **+1.48** |
| ORB | QR-DQN | ❌ | — | — | — |
| HA-Vol | QR-DQN | ❌ | — | — | — |
| Gap | QR-DQN | ❌ | — | — | — |

## Algorithm Selection Rationale

- **VWAP-MR** → SAC: Continuous action space, entropy-maximizing, sample-efficient for scale-in/out
- **ORB / HA-Vol / Gap** → QR-DQN: Off-policy with distributional RL for sparse discrete setups
- **Manager** → PPO: Vec-env parallelism, simple discrete action space

## Look-Ahead Bias Protection

The system implements strict look-ahead bias prevention:

1. **Resampling**: M5 → M15 with `label='right', closed='right'` (timestamp marks bar close)
2. **Higher TF Join**: Shift(1) on higher timeframes when joining to lower TF
3. **Rolling Calculations**: All rolling indicators computed cumulatively up to t-1
4. **Session Reset**: VWAP re-initialized at session open, not midnight
5. **Unit Tests**: `tests/test_no_lookahead.py` empirically verifies no leakage

## Hyperparameters

### SAC (VWAP-MR)
- Learning rate: 3e-4
- Buffer size: 1M
- Batch size: 256
- Tau: 0.005
- Entropy coefficient: "auto"
- Policy network: [64, 32]

### QR-DQN (ORB, HA-Vol, Gap)
- Learning rate: 1e-4
- Buffer size: 500k
- Batch size: 128
- Exploration fraction: 0.2-0.4
- N quantiles: 50
- Policy network: [64, 32]

### PPO (Manager)
- N steps: 2048
- Batch size: 256
- Target KL: 0.015
- Entropy coefficient: 0.005
- Policy network: [64, 32]

## Testing

Run the comprehensive test suite:

```bash
# Using pytest
.\venv\Scripts\python.exe -m pytest casino2/tests/test_no_lookahead.py -v

# Or directly
.\venv\Scripts\python.exe -X utf8 -m casino2.tests.test_no_lookahead
```

Tests verify:
- Feature computation consistency across data truncations
- Environment observations don't depend on future bars
- Multiple cut-point validation

## Model Checkpoints

Models are saved in `models/{worker}/`:
- `best_model.zip` - Best model on validation Sharpe
- `sac_final.zip` / `qrdqn_final.zip` - Final model after training
- `vecnormalize.pkl` - VecNormalize statistics for inference
- `ckpt/` - Periodic checkpoints every 500k steps

## Troubleshooting

### Out of Memory
Reduce buffer size or batch size in training script:
```bash
python -m casino2.workers.vwap_mr.train --buffer-size 100000
```

### Slow Training
Use CPU for QR-DQN (small networks):
```bash
python -m casino2.workers.orb.train --device cpu
```

### Data Not Found
Ensure parquet cache exists:
```bash
python -m casino2.loader --prepare
```

### Look-Ahead Test Fails
Check feature computation in `workers/*/features.py` for:
- Future bar references
- Incorrect resampling parameters
- Missing shift() operations

## Documentation

- `HRL_PLAN.md` - Complete architecture specification with state/action/reward definitions
- `OBS_Strategy.md` - Detailed Opening Range Breakout strategy analysis (Swedish)
- `LegsToStandOn.md` - Smart Money Concepts reference (PineScript)

## License

This project is part of a quantitative trading research system. See individual license files for details.

## Contributing

1. Ensure all look-ahead bias tests pass before committing
2. Document new features in HRL_PLAN.md
3. Update this README with status changes
4. Run evaluations on val/test/april splits before marking workers as complete
