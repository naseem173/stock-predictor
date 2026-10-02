"""Walk-forward daily backtest of the same ranking: each day pick top-N by score at the open,
enter at the open, exit at stop-loss / target / close. If SL and target both hit in one bar,
assume stop-loss first (conservative). Costs are charged per round trip."""
import numpy as np
import pandas as pd
from signals import feature_frame

def run_backtest(frames, top=3, gap_w=0.10, sl_mult=1.5, rr=2.0, min_score=0.15, cost=0.001, last_days=400):
    feats = {t: feature_frame(df, gap_w) for t, df in frames.items()}
    trades = []
    dates = sorted(set().union(*[set(f.index) for f in feats.values()]))[-last_days:]
    for dt in dates:
        cand = []
        for t, f in feats.items():
            if dt in f.index:
                r = f.loc[dt]
                if not np.isnan(r.score) and not np.isnan(r.atr_prev) and abs(r.score) >= min_score:
                    cand.append((abs(r.score), t, r))
        for _, t, r in sorted(cand, key=lambda x: -x[0])[:top]:
            side = 1 if r.score > 0 else -1; e = r.open; risk = sl_mult * r.atr_prev
            sl, tp = e - side * risk, e + side * rr * risk
            hit_sl = r.low <= sl if side == 1 else r.high >= sl
            hit_tp = r.high >= tp if side == 1 else r.low <= tp
            if hit_sl: x, why = sl, "SL"
            elif hit_tp: x, why = tp, "TARGET"
            else: x, why = r.close, "CLOSE"
            ret = side * (x - e) / e - cost
            trades.append(dict(date=dt, ticker=t, side="BUY" if side == 1 else "SELL", entry=e, exit=x,
                               reason=why, ret=ret))
    tr = pd.DataFrame(trades)
    if tr.empty: return tr, {}, pd.Series(dtype=float)
    daily = tr.groupby("date").ret.mean(); equity = (1 + daily).cumprod()
    dd = (equity / equity.cummax() - 1).min()
    wins = tr.ret > 0
    stats = dict(trades=len(tr), win_rate=float(wins.mean()), avg_return_pct=float(100*tr.ret.mean()),
                 avg_win_pct=float(100*tr.ret[wins].mean()) if wins.any() else 0.0,
                 avg_loss_pct=float(100*tr.ret[~wins].mean()) if (~wins).any() else 0.0,
                 total_return_pct=float(100*(equity.iloc[-1]-1)), max_drawdown_pct=float(100*dd),
                 sharpe=float(daily.mean()/daily.std()*np.sqrt(252)) if daily.std() > 0 else 0.0)
    return tr, stats, equity
