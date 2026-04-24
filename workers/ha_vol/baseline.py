"""
M30 HA-Supertrend — Rule-Based Baseline
=========================================

Validerar edge INNAN RL-träning.

Strategi (riktig användning av HA):
  Position = ha_st_signal (+1/-1), gatad av ADX-regime + session.

  HA är ett trend-verktyg, INTE ett reversal-verktyg:
    ADX > 20 + HA bull  →  long (+1)
    ADX > 20 + HA bear  →  short (-1)
    ADX ≤ 20 (ranging)  →  flat (0)

  Session-filter (valfritt): EU/US aktiv → annars flat

  Kostnad: spread_bps / 10 000 per 100% positionsbyte (log-space)
  MIN_TRADE_BARS: minsta antal M30-bars mellan positionsbyten (default 1)

Kör:
    python -X utf8 -m casino2.workers.ha_vol.baseline
    python -X utf8 -m casino2.workers.ha_vol.baseline --sweep-slippage
    python -X utf8 -m casino2.workers.ha_vol.baseline --sweep-adx
    python -X utf8 -m casino2.workers.ha_vol.baseline --longs-only
    python -X utf8 -m casino2.workers.ha_vol.baseline --no-session
    python -X utf8 -m casino2.workers.ha_vol.baseline --splits val test
"""
from __future__ import annotations

import argparse

import numpy as np

from casino2.loader import INSTRUMENTS, load_instrument
from casino2.workers.ha_vol.features import (
    ADX_THRESHOLD,
    BARS_PER_YEAR_M30,
    compute_features,
)
from casino2.workers.vwap_mr.eval import compute_metrics


# ── Konstanter ───────────────────────────────────────────────────────────────

DEFAULT_SPREAD_BPS     = 2.0
DEFAULT_MIN_TRADE_BARS = 1     # M30 = 30 min per bar; 1 = ingen extra throttle


# ── Backtest ─────────────────────────────────────────────────────────────────

def backtest_ha_trend(
    instrument:      str,
    split:           str,
    *,
    spread_bps:      float = DEFAULT_SPREAD_BPS,
    min_trade_bars:  int   = DEFAULT_MIN_TRADE_BARS,
    adx_threshold:   float = ADX_THRESHOLD,
    longs_only:      bool  = False,
    shorts_only:     bool  = False,
    require_session: bool  = True,
) -> dict:
    """
    M30 HA-Supertrend backtest på (instrument, split).

    Returnerar dict med sharpe_per_trade, cagr, mdd, trades osv.
    """
    m30 = load_instrument(instrument, "M30", split=split)
    f   = compute_features(m30=m30, instrument=instrument, dropna=True)

    if len(f) < 50:
        return dict(sharpe=0.0, sharpe_per_trade=0.0, cagr=0.0,
                    mdd=0.0, trades=0, win_rate=0.0,
                    total_pnl=0.0, profit_factor=0.0, cagr_clipped=False)

    ha_st  = f["ha_st_signal"].to_numpy(np.float64)   # +1 / -1
    adx    = f["m30_adx_norm"].to_numpy(np.float64) * 100.0  # tillbaka till 0-100
    sess   = f["session"].to_numpy(np.float64)
    cl     = f["close"].to_numpy(np.float64)
    n      = len(f)

    tx_cost    = spread_bps / 10_000.0
    pnls       = np.zeros(n, np.float64)
    positions  = np.zeros(n, np.float64)

    pos        = 0.0
    bars_since = 0

    for t in range(n - 1):
        bars_since += 1

        # Bestäm target: HA-ST riktning, gatad av ADX + session
        if adx[t] > adx_threshold:
            target = ha_st[t]
        else:
            target = 0.0  # ranging → flat

        if require_session and sess[t] < 0.5:
            target = 0.0  # utanför session → flat

        if longs_only:
            target = max(target, 0.0)
        if shorts_only:
            target = min(target, 0.0)

        # Throttle: håll minst min_trade_bars mellan positionsbyten
        if bars_since < min_trade_bars:
            target = pos

        # P&L på nuvarande bar
        log_ret = (np.log(cl[t + 1] / cl[t])
                   if cl[t] > 0 and cl[t + 1] > 0 else 0.0)
        bar_pnl = pos * log_ret

        # Spread-kostnad vid positionsbyte
        delta = abs(target - pos)
        cost  = delta * tx_cost
        if delta > 0.01:
            bars_since = 0

        pnls[t]      = bar_pnl - cost
        positions[t] = pos
        pos          = target

    return compute_metrics(
        pnls=pnls,
        positions=positions,
        bars_per_year=BARS_PER_YEAR_M30,
    )


# ── Formatering ──────────────────────────────────────────────────────────────

def fmt_row(inst: str, m: dict) -> str:
    sht  = m.get("sharpe_per_trade", 0.0)
    sh   = m.get("sharpe", 0.0)
    flag = "OK" if sht > 0.3 else ("~~ " if sht > 0 else "XX")
    pf   = m.get("profit_factor", 0.0)
    pf_s = f"{pf:5.2f}" if np.isfinite(pf) else "  inf"
    cagr_s = f"{m['cagr'] * 100:+7.1f}%"
    if m.get("cagr_clipped"):
        cagr_s += "*"
    return (
        f"  {inst:6s} [{flag}]  "
        f"ShT={sht:+6.2f}  Sh={sh:+6.2f}  "
        f"CAGR={cagr_s}  PF={pf_s}  "
        f"MDD={m['mdd'] * 100:5.1f}%  "
        f"Trades={m['trades']:>5d}  "
        f"Win={m['win_rate'] * 100:4.1f}%"
    )


def run_split(split: str, instruments: list, **kwargs) -> float:
    sharpes = []
    print(f"\n  [{split.upper()}]")
    for inst in instruments:
        try:
            m = backtest_ha_trend(inst, split, **kwargs)
        except Exception as e:
            print(f"  {inst:6s} FEL: {e}")
            continue
        print(fmt_row(inst, m))
        sharpes.append(m.get("sharpe_per_trade", 0.0))
    if sharpes:
        mean    = float(np.mean(sharpes))
        verdict = "EDGE" if mean > 0.3 else ("SVAG" if mean > 0 else "INGEN")
        print(f"  {'Medel':6s}        ShT={mean:+.3f}  [{verdict}]")
        return mean
    return 0.0


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="M30 HA-Supertrend baseline")
    ap.add_argument("--splits", nargs="+",
                    default=["train", "val", "test", "april"])
    ap.add_argument("--instruments", nargs="+", default=list(INSTRUMENTS))
    ap.add_argument("--spread", type=float, default=DEFAULT_SPREAD_BPS)
    ap.add_argument("--min-trade-bars", type=int, default=DEFAULT_MIN_TRADE_BARS)
    ap.add_argument("--adx", type=float, default=ADX_THRESHOLD,
                    help="ADX-tröskel för trendregim (default 20)")
    ap.add_argument("--sweep-slippage",  action="store_true")
    ap.add_argument("--sweep-adx",       action="store_true",
                    help="Swepa ADX-trösklar 0/15/20/25/30/35")
    ap.add_argument("--longs-only",      action="store_true")
    ap.add_argument("--shorts-only",     action="store_true")
    ap.add_argument("--no-session",      action="store_true",
                    help="Inaktivera session-filtret (handel alla timmar)")
    args = ap.parse_args()

    require_session = not args.no_session

    print(f"\n{'=' * 85}")
    print(f"  M30 HA-Supertrend  (ATR=10, mult=3.0, ADX>{args.adx})")
    print(f"  spread={args.spread} bps  |  min_trade_bars={args.min_trade_bars}  "
          f"|  session={'ON' if require_session else 'OFF'}")
    print(f"{'=' * 85}")

    # ── Slippage sweep ──────────────────────────────────────────────────
    if args.sweep_slippage:
        print(f"\n  Slippage-robusthet")
        print(f"  {'Spread':>10s}", end="")
        for sp in args.splits:
            print(f"  {sp.upper():>10s}", end="")
        print()
        for slip in [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]:
            print(f"  {slip:>5.1f} bps   ", end="")
            for sp in args.splits:
                vals = []
                for inst in args.instruments:
                    try:
                        m = backtest_ha_trend(
                            inst, sp,
                            spread_bps=slip,
                            min_trade_bars=args.min_trade_bars,
                            adx_threshold=args.adx,
                            require_session=require_session,
                        )
                        vals.append(m.get("sharpe_per_trade", 0.0))
                    except Exception:
                        pass
                mean = float(np.mean(vals)) if vals else 0.0
                print(f"  {mean:>+10.3f}", end="")
            print()
        print(f"\n{'=' * 85}\n")
        return

    # ── ADX sweep ───────────────────────────────────────────────────────
    if args.sweep_adx:
        print(f"\n  ADX-tröskels robusthet")
        print(f"  {'ADX thr':>10s}", end="")
        for sp in ["train", "val", "test"]:
            print(f"  {sp.upper():>10s}", end="")
        print()
        for adx_thr in [0.0, 15.0, 20.0, 25.0, 30.0, 35.0]:
            print(f"  {adx_thr:>10.1f}", end="")
            for sp in ["train", "val", "test"]:
                vals = []
                for inst in args.instruments:
                    try:
                        m = backtest_ha_trend(
                            inst, sp,
                            spread_bps=args.spread,
                            adx_threshold=adx_thr,
                            require_session=require_session,
                        )
                        vals.append(m.get("sharpe_per_trade", 0.0))
                    except Exception:
                        pass
                mean = float(np.mean(vals)) if vals else 0.0
                print(f"  {mean:>+10.3f}", end="")
            print()
        print(f"\n{'=' * 85}\n")
        return

    # ── Standard-körning ────────────────────────────────────────────────
    kwargs = dict(
        spread_bps=args.spread,
        min_trade_bars=args.min_trade_bars,
        adx_threshold=args.adx,
        longs_only=args.longs_only,
        shorts_only=args.shorts_only,
        require_session=require_session,
    )
    mode = ("BARA LONGS" if args.longs_only
            else "BARA SHORTS" if args.shorts_only
            else "Long + Short")
    print(f"\n  {mode}")
    print("-" * 85)
    for sp in args.splits:
        run_split(sp, args.instruments, **kwargs)
    print(f"\n{'=' * 85}\n")


if __name__ == "__main__":
    main()
