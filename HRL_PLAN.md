# Hierarchical RL Plan — Casino Manager v2

## Översikt

Fyra specialist-RL-agenter (Workers) + en Meta-agent (Manager).
Varje worker tränas isolerat på sin strategi. Manager lär sig välja vilken worker som får styra kapitalet per M30-fönster.

```
                    ┌──────────────────┐
                    │  HRL Manager     │  (M30 features + regime)
                    │  action ∈ {0..3} │
                    └────────┬─────────┘
                             │ delegerar
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼
     ┌────────┐        ┌─────────┐        ┌──────────┐     ┌──────┐
     │  ORB   │        │ VWAP-MR │        │  HA-Vol  │     │ Gap  │
     │  (M5)  │        │  (M5)   │        │  (M15)   │     │(M15) │
     └────────┘        └─────────┘        └──────────┘     └──────┘
```

## Data

- **Källa:** `data/april2026/2017-2026/*_M5.csv` och `*_M30.csv`
- **Instrument:** USTEC, US500, DE40
- **M15 genereras** via `df.resample('15min', label='right', closed='right')` från M5
- **Split:**
  - Train: 2017-01 → 2024-12
  - Val:   2025-01 → 2025-10
  - Test:  2025-11 → 2026-03
  - April26: forward-only sanity test

### Look-Ahead Bias — kritiska regler

Ett M15-bar som stänger kl 10:15 får **inte** observeras av agenten vid M5-bar 10:05 eller 10:10. Regler som koden måste följa:

1. **Resample med `label='right', closed='right'`** → tidsstämpeln markerar bar-stängningen, inte början.
2. **Shift(1) på högre timeframes** när de joinas till lägre TF: `m15_features.shift(1).reindex(m5_index, method='ffill')`. Detta garanterar att M5-bar 10:05 bara ser M15-bar som stängde 10:00.
3. **VWAP/rolling beräknas cumulativt** med data upp till och med `t−1`. Aldrig `.rolling(center=True)`.
4. **Unit-test:** `tests/test_no_lookahead.py` — pick random timestamp, assert att feature-värden är identiska oavsett om dataframen trunkeras vid `t` eller innehåller framtida bars.
5. **Session-reset för VWAP:** re-initiera vid session-öppning, inte vid midnatt.

## Worker 1: Opening Range Breakout (ORB)

**Timeframe:** M5 (med M15/M30 range-kontext)

### State (13 features)
- `time_of_day` (cyclic sin/cos, 2 features)
- `minutes_since_session_open` (normaliserad)
- `dist_to_or_high` = (close − range_high) / ATR
- `dist_to_or_low`  = (range_low − close) / ATR
- `or_width_atr` = (range_high − range_low) / ATR
- `rvol_m5` = volume / rolling_20_mean
- `atr_pct` = ATR / close
- `returns_m5_lag_1..5` (5 features)

### Action
`Discrete(3)`: 0 = Flat, 1 = Long, 2 = Short

### Reward
```python
r_t = log_return(position) 
    - 0.0005 * |action_change|               # transaction cost
    - 2.0 * whipsaw_penalty                  # om utbrott utan volym-konfirm
    + 0.5 * bonus_if_RVOL>1.5_at_breakout
```

### Regler
- Endast aktiv under session-öppning (första 120 min)
- Max 1 position i taget
- Force-close vid session-slut

---

## Worker 2: VWAP Mean Reversion

**Timeframe:** M5

### State (11 features)
- `dist_vwap_pct` = (close − VWAP) / VWAP
- `bb_upper_dist` = (close − BB_upper) / close (2σ)
- `bb_lower_dist` = (BB_lower − close) / close (2σ)
- `rsi_2` (fast RSI)
- `macd_hist`
- `atr_pct`
- `volume_ratio` = vol / vol_sma_20
- `returns_lag_1..4` (4 features)

### Action
`Box(-1, +1, shape=(1,))` — kontinuerlig positionsstorlek (scale-in/out)

> **Algoritm-not:** VWAP-workern tränas med **SAC** (Soft Actor-Critic) istället för PPO.
> SAC är off-policy + entropi-maximerande och hanterar kontinuerliga action-rymder
> betydligt mer sample-effektivt än PPO, särskilt för scale-in/out-beslut.
> Övriga 3 workers + Manager använder PPO (diskreta actions).

### Reward
```python
r_t = log_return(position)
    - 0.001 * |action_change|
    + bonus_vwap_touch  # +0.5 om position stängs vid VWAP-återgång
    - 2.0 * penalty_if_position_and_price_breaks_3sigma  # trend-break straff
```

### Regler
- Blockerad när ADX > 30 (trendande marknad)
- Max hold 40 M5-bars (3.3h)

---

## Worker 3: Heikin Ashi Volatility Expansion

**Timeframe:** M15

### State (14 features)
- HA OHLC (4 features, normaliserade till pct-change)
- `ha_body_size` = |ha_close − ha_open| / ATR
- `ha_upper_wick` / ATR
- `ha_lower_wick` / ATR
- `ha_streak` (bars i samma färg, signed: + bull / − bear)
- `bb_width` = (BB_upper − BB_lower) / close
- `bb_width_zscore_50` (kompression-detektor)
- `atr_pct`
- `atr_ratio` = ATR_14 / ATR_50
- `volume_ratio`

### Action
`Discrete(5)`: {Flat, Long-0.5, Long-1.0, Short-0.5, Short-1.0}

### Reward
```python
# Asymmetrisk: belönas för att HÅLLA i trend, straffas för att agera i kompression
r_t = 2.0 * log_return(position) * same_color_streak_bonus
    - 1.0 * abs(action) * is_compression  # straff att handla i squeeze
    - 0.5 * on_reversal_color_change       # tidig exit-incentiv
```

### Regler
- `is_compression` = `bb_width_zscore_50 < -0.5`
- Trade endast om streak ≥ 2 bars samma HA-färg

---

## Worker 4: Gap Dynamics (Overnight Alpha)

**Timeframe:** M15 (med M5 volym-konfirm)

### State (9 features)
- `log_gap` = log(open_today / close_yesterday)
- `gap_size_atr` = |gap| / ATR_daily
- `gap_direction` (+1 up / -1 down)
- `prev_day_trend` (sign av föregående dags close-open)
- `rvol_first_30min` = vol_m5_first_6bars / historic_mean
- `distance_to_prev_close_pct`
- `time_since_open_min` (normaliserad)
- `atr_pct`
- `is_session_open_first_60min` (binary)

### Action
`Discrete(3)`: 0 = Ignore, 1 = Gap-and-Go, 2 = Fade-the-Gap

### Reward
```python
# Belönar korrekt gap-kategorisering
r_t = log_return(position)
    + 1.0 * bonus_exhaustion_fade      # om fade lyckas → pris tillbaka till prev_close
    + 1.0 * bonus_breakaway_go         # om gap-and-go → trend fortsätter
    - 0.0005 * transaction_cost
```

### Regler
- Endast aktiv första 60 min av session
- 1 trade per dag max
- Kräver `|log_gap| > 0.003` för att anses vara gap

---

## HRL Manager

**Timeframe:** M30 (uppdateras var 30:e min)

### State (22 features)

**Primära — regim-indikatorer (14 features, DOMINERAR manager-beslutet):**
- `adx_14`, `adx_slope_5` (trend-styrka + om den bildas/avtar)
- `atr_pct`, `atr_ratio_14_50` (volatilitets-regim)
- `bb_width_pct`, `bb_width_zscore_50` (kompression/expansion)
- `vix_proxy` (realiserad 20-bar vol) + `vix_zscore_100`
- `trend_slope_50`, `trend_slope_200` (macro-regim)
- `volume_regime` (vol/vol_sma_100)
- `time_of_day_sin`, `time_of_day_cos` (session-fas)
- `day_of_week` (onehot ej, bara normaliserad 0–1)

**Sekundära — worker-signaler (4 features):**
- 4 × `worker_signal` ∈ {−1, 0, +1} (sign av workerns senaste action)

**Tertiära — worker-PnL (4 features, AVVIKTADE):**
- 4 × `worker_recent_pnl_zscore` — rolling 20-bar reward, z-scored mot workerns egna 500-bar-distribution
- Varför z-score: undviker att Manager "chasar" en worker som bara just haft tur
- Dessa features **skalas ner med 0.3× via feature-normalisering** innan policy-nätet så regim-features dominerar

> **Anti-chasing:** Feature importance ska vara `regime > signal >> recent_pnl`.
> Om Manager lär sig att följa recent_pnl för starkt → lägg till `−λ * pnl_feature_grad`
> regularization i en custom callback.

### Action
`Discrete(5)`: {Flat, ORB, VWAP-MR, HA-Vol, Gap}

Manager väljer **vilken worker** som får sin position propagerad till portfolio-nivå.

### Reward
```python
r_t = portfolio_log_return
    - 0.5 * |manager_action_change|  # straff för frekventa byten
    - λ * drawdown_penalty
    + sortino_bonus_per_10_bars
```

### Regler
- Switch cooldown: 4 bars (= 2h på M30) mellan manager-actions
- Kelly sizing på portfolio-nivå (quarter Kelly)

---

## Implementationsordning

1. **`casino/data/loader.py`** — multi-instrument M5/M30 parquet-loader + train/val/test split
2. **`casino/workers/vwap_mr/`** — första worker (enklast att verifiera)
3. **`casino/workers/orb/`** — andra worker (tidsbaserad, enkel att debugga)
4. **`casino/workers/ha_vol/`** — tredje worker
5. **`casino/workers/gap/`** — fjärde worker
6. **`casino/manager/hrl_env.py` + `train_hrl.py`** — när alla 4 workers är positiva OOS

## Success Criteria per Worker

Innan vi går vidare till nästa worker:
- ✅ Sharpe > 0 på val-set
- ✅ Sharpe > −0.3 på test-set (tolerans för regime-shift)
- ✅ Max drawdown < 5%
- ✅ Trade-count rimlig (inte 3× baseline som nuvarande manager)

## Träningsinfrastruktur

### Algorithm per worker (off-policy preferred)

| Worker | Algorithm | Motiv |
|--------|-----------|-------|
| **VWAP-MR** | SAC (SB3) | Kontinuerlig action, entropi-maximerande |
| **ORB** | QR-DQN (sb3-contrib) | Sparse setups (1-2/dag), off-policy replay |
| **HA-Vol** | QR-DQN (sb3-contrib) | Diskret(5), distributional stabil |
| **Gap** | QR-DQN (sb3-contrib) | Diskret(3), 1 trade/dag max = extrem sparse |
| **Manager** | PPO (SB3) | Vec-env parallellism, enkel diskret(5) |

### Gemensamma settings

- **Policy:** MlpPolicy (små nät: [64, 32])
- **SAC hyperparams:** `lr=3e-4, buffer=1M, batch=256, tau=0.005, ent_coef="auto"`
- **QR-DQN hyperparams:** `lr=1e-4, buffer=500k, batch=128, exploration_fraction=0.2, n_quantiles=50`
- **PPO hyperparams:** `n_steps=2048, batch_size=256, target_kl=0.015, ent_coef=0.005`
- **Checkpoints:** var 500k steg, välj best på val-Sharpe
- **TensorBoard:** `logs/hrl/{worker}/`
