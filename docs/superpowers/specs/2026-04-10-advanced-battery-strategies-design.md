# Advanced Battery Strategy Design

**Date:** 2026-04-10
**Goal:** Build three advanced battery strategies — Oracle, MPC, and DQN — to find and approach the theoretical cost minimum for a home battery system.

## Context

The powerflow-simulator has 7 rule-based battery strategies. On a 15 kWh battery (4.8 kW charge/discharge, 2.5 kW manual charge) over Jan 2024 - Apr 2026:

- Smart Discharge (best): 1590.56 EUR
- Force Charge at Night: 1607.48 EUR
- Self Consume: 1740.34 EUR

The gap between strategies is narrow (~150 EUR over 2.3 years). The question: how much room for improvement exists, and can we build a deployable strategy that captures it?

## Tariff Structure

| Period | Hours | Rate (EUR/kWh) |
|--------|-------|----------------|
| Valley | 00:00-08:00 | 0.085 |
| Shoulder | 08:00-10:00 | 0.134 |
| Peak | 10:00-14:00 | 0.182 |
| Shoulder | 14:00-18:00 | 0.134 |
| Peak | 18:00-22:00 | 0.182 |
| Shoulder | 22:00-24:00 | 0.134 |
| Export (flat) | 00:00-24:00 | 0.08 |

Valley-to-peak spread: 2.14x. Net savings per kWh shifted from peak to valley (after 95% round-trip efficiency): ~0.092 EUR/kWh.

## Strategy 1: Oracle Optimizer (Theoretical Ceiling)

### Purpose

Compute the absolute minimum cost with perfect knowledge of all future solar generation and load. Not deployable — serves as the benchmark against which MPC and DQN are measured.

### Formulation

Linear program over all T timesteps (1-minute resolution):

**Decision variables per timestep t:**
- `charge[t]` — grid-to-battery power (kW, >= 0)
- `discharge[t]` — battery-to-load power (kW, >= 0)
- `grid_import[t]` — grid import after battery (kW, >= 0)
- `grid_export[t]` — grid export (kW, >= 0)
- `soc[t]` — battery state of charge (kWh)

**Objective:**
```
minimize Σ_t [ grid_import[t] × rate_import(t) × dt - grid_export[t] × rate_export(t) × dt ]
```

**Constraints:**
```
# Energy balance
solar[t] + grid_import[t] + discharge[t] = load[t] + charge[t] + grid_export[t]

# Battery dynamics
soc[t] = soc[t-1] + charge[t] × efficiency × dt - discharge[t] / efficiency × dt

# Battery limits
min_soc <= soc[t] <= capacity
0 <= charge[t] <= max_charge_rate
0 <= discharge[t] <= max_discharge_rate

# Grid limits
0 <= grid_import[t] <= max_grid_import
0 <= grid_export[t] <= max_grid_export
```

**Simplifications:**
- Ignores charge tapering above 90% SoC (nonlinear, LP requires linear constraints). The oracle will slightly overestimate achievable savings — acceptable for a ceiling benchmark.
- Efficiency is constant at 0.95.

**Solver:** `scipy.optimize.linprog` with HiGHS backend. Handles 1M+ variables.

**Integration:** Pre-solves the LP over the full simulation period, then replays the optimal schedule through `EnergySimulator` for consistent metric calculation. Receives full solar/load timeseries at construction time (same pattern as ForecastChargeStrategy receiving daily_forecasts).

## Strategy 2: MPC (Model Predictive Control)

### Purpose

Deployable real-time strategy. At each decision point, solve a small LP over a 24h forecast horizon, execute the first step, re-solve with updated state and forecasts.

### Algorithm

```
Every 15 minutes:
  1. Observe current battery SoC
  2. Build 24h forecast at 15-minute resolution (96 steps):
     - Solar: GHI forecast × GHI_TO_PV_FACTOR / 1000
     - Load: historical average for (hour, weekday_vs_weekend) over past 28 days
  3. Solve the Oracle LP formulation over 96 steps
  4. Cache the solution (charge/discharge plan for next 15 min)

Every minute (between re-solves):
  5. Execute the cached plan for the current 15-min window
  6. Apply solar/load using standard energy flow logic
```

### Forecast Construction

**Solar forecast:** For backtesting, uses `ghi_forecast` column from `data/irradiance/` CSV files (already hourly). For deployment, fetch from Open-Meteo or Solcast API.

**Load forecast:** Compute average load per (hour_of_day, is_weekend) bucket from the 28 days preceding the current timestamp. Simple but effective — household load patterns are regular.

### Performance

- 96 LP variables per solve × 4 solves/hour × 24h × ~820 days = ~78,700 LP solves for full backtest
- Each solve is tiny (96 steps × 5 variables). Full backtest should complete in minutes.

### Integration

Registers as strategy `mpc`. During `calculate_energy_flows()`, checks if 15 minutes have elapsed since last solve. If so, builds forecasts, solves horizon LP, caches plan. Returns current step's charge/discharge decision from cached plan.

## Strategy 3: DQN (Deep Q-Network) RL Agent

### Purpose

Learn a battery control policy from historical data that captures patterns the LP formulations miss — load spikes correlated with day-of-week, seasonal solar transitions, appliance schedules.

### State Space (16 dimensions)

```
state = [
  soc,                    # battery level (0-1, normalized)
  hour_sin, hour_cos,     # cyclical hour encoding
  dow_sin, dow_cos,       # cyclical day-of-week encoding
  month_sin, month_cos,   # cyclical month encoding
  current_solar,          # current solar generation (normalized kW)
  current_load,           # current house load (normalized kW)
  current_rate,           # current import tariff (normalized)
  solar_forecast_4h,      # avg forecasted solar next 4h
  solar_forecast_12h,     # avg forecasted solar next 12h
  load_avg_4h,            # rolling avg load past 4h
  hours_to_next_valley,   # hours until next valley rate starts
  hours_to_next_peak,     # hours until next peak rate starts
]
```

### Action Space

Discretized to 9 levels:
```
actions = [-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]
```
Where -1.0 = max discharge rate, +1.0 = max charge rate.

### Reward

Negative cost of the current timestep (dense, every minute):
```
reward = -(grid_import × import_rate × dt) + (grid_export × export_rate × dt)
```

### Network Architecture

- Input: 16 features
- Hidden: 2 layers × 128 units, ReLU
- Output: 9 Q-values (one per action)
- Optimizer: Adam, lr=1e-4
- Experience replay buffer: 100K transitions
- Target network: soft update every 1000 steps
- Epsilon: 1.0 → 0.05 over first 100 episodes

### Training

- Episodes = one month of historical data each
- ~27 months available × ~20 passes = ~500 episodes
- Training time: well under an hour on 45 TFLOPS
- Evaluation: held-out months, compare cost vs Oracle and MPC

### Integration

Registers as strategy `dqn_agent`. Trained model weights saved to `output_files/dqn_policy.pt`. During simulation, loads weights and runs inference — single forward pass per step.

## File Structure

```
src/domain/strategy/
├── model.py              # existing strategies (unchanged)
├── forecast_loader.py    # existing (reused by Oracle + MPC)
├── oracle.py             # Oracle LP solver + strategy wrapper
├── mpc.py                # MPC rolling LP + load forecaster
└── dqn/
    ├── agent.py          # DQN network, training loop
    ├── environment.py    # Gym-like env wrapping EnergySimulator
    └── train.py          # CLI entry point for training
```

## Registration

All three register in `simulation_service.py` alongside existing strategies:

- `oracle` — "Oracle Optimizer" — requires full timeseries upfront
- `mpc` — "Model Predictive Control" — uses forecast data
- `dqn_agent` — "DQN Agent" — uses trained model weights

## Dependencies

- `scipy` — LP solver (HiGHS) for Oracle and MPC
- `torch` — DQN training and inference (already present in project)

## Expected Outcomes

| Strategy | Type | Deployable | Expected result |
|----------|------|------------|-----------------|
| Oracle | Perfect hindsight LP | No (benchmark) | Theoretical minimum cost |
| MPC | Rolling forecast LP | Yes | 5-10% above oracle |
| DQN | Learned policy | Yes | Between MPC and oracle |
| Smart Discharge | Rule-based (current best) | Yes | 1590.56 EUR (measured) |
| FCAN | Rule-based | Yes | 1607.48 EUR (measured) |

## Build Order

1. Oracle first — establishes ceiling, validates LP formulation
2. MPC second — reuses Oracle's LP, adds forecast + rolling horizon
3. DQN third — uses Oracle as training signal / upper bound reference

Each validates the next. If MPC reaches 98% of Oracle, DQN may not add enough value to justify its complexity.
