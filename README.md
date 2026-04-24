# Casino2 — Hierarchical RL Trading System

4 specialist-RL-agenter + 1 meta-manager för multi-instrument handel (USTEC, US500, DE40).

## Struktur

```
casino2/
├── HRL_PLAN.md                  # Full arkitektur-spec (state/action/reward per worker)
├── README.md                    # Denna fil
├── loader.py                    # Data-pipeline (CSV → parquet, splits)
├── data/
│   └── parquet/                 # Cache: {instr}_{tf}.parquet
├── workers/
│   ├── vwap_mr/                 # VWAP Mean Reversion (SAC) ✅ tränad
│   │   ├── features.py
│   │   ├── env.py
│   │   ├── train.py
│   │   └── eval.py
│   ├── orb/                     # Opening Range Breakout (QR-DQN) 🏗️ klar, otränad
│   ├── ha_vol/                  # Heikin Ashi Vol Expansion 📋 TBD
│   └── gap/                     # Gap Dynamics / Overnight Alpha 📋 TBD
├── manager/                     # HRL Meta-agent (PPO) 📋 TBD
├── models/
│   ├── vwap_mr/
│   │   ├── best_model.zip
│   │   ├── vecnormalize.pkl
│   │   └── ckpt/
│   └── orb/
├── logs/                        # TensorBoard
│   ├── vwap_mr/
│   └── orb/
└── tests/
    └── test_no_lookahead.py    # Empirisk leak-verifiering
```

## Datakällor (utanför casino2/)

CSV-källor ligger kvar på gamla platsen för att undvika flytt:

```
OpenClaw_RL/data/april2026/2017-2026/
  USTEC_M5_*.csv, USTEC_M30_*.csv
  US500_M5_*.csv, US500_M30_*.csv
  DE40_M5_*.csv,  DE40_M30_*.csv
```

## Kom igång

Alla kommandon körs från `OpenClaw_RL/` med venv aktiv.

### Första gången
```powershell
# 1. Skapa parquet-cache (körs en gång)
.\venv\Scripts\python.exe -X utf8 -m casino2.loader --prepare

# 2. Verifiera cache
.\venv\Scripts\python.exe -X utf8 -m casino2.loader --check

# 3. Kör look-ahead-tester (alltid före träning)
.\venv\Scripts\python.exe -X utf8 -m casino2.tests.test_no_lookahead
```

### VWAP-MR
```powershell
# Full träning (5M steg, ~5h GPU)
.\venv\Scripts\python.exe -X utf8 -m casino2.workers.vwap_mr.train --timesteps 5000000 --device cuda

# Eval
.\venv\Scripts\python.exe -X utf8 -m casino2.workers.vwap_mr.eval --splits val test april
```

### ORB
```powershell
# Full träning (2M steg, ~10 min GPU)
.\venv\Scripts\python.exe -X utf8 -m casino2.workers.orb.train --timesteps 2000000 --device cuda

# Eval
.\venv\Scripts\python.exe -X utf8 -m casino2.workers.orb.eval --splits val test april
```

### TensorBoard
```powershell
.\venv\Scripts\python.exe -m tensorboard --logdir casino2/logs
```

## Success Criteria per Worker

Innan en worker går in i HRL Manager:
- ✅ Sharpe > 0 på val-set
- ✅ Sharpe > −0.3 på test-set
- ✅ Max drawdown < 5%
- ✅ Trade-count rimlig

## Nuvarande status

| Worker | Algorithm | Tränad | Val Sharpe | Test Sharpe | April Sharpe |
|--------|-----------|--------|------------|-------------|--------------|
| VWAP-MR | SAC | ✅ 5M steg | **+1.78** | **+3.39** | **+1.48** |
| ORB | QR-DQN | ❌ | — | — | — |
| HA-Vol | QR-DQN | ❌ | — | — | — |
| Gap | QR-DQN | ❌ | — | — | — |

## Algoritm-val per worker

- **VWAP-MR** → SAC (kontinuerlig action, entropi-maximerande)
- **ORB / HA-Vol / Gap** → QR-DQN (off-policy + distributional för sparse discrete)
- **Manager** → PPO (vec-env parallellism för diskret meta-val)
