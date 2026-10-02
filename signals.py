"""Indicators, support/resistance, scoring and trade levels (entry / stop-loss / target).
All features use data up to the PREVIOUS close plus today's opening gap, so there is no look-ahead."""
import numpy as np
import pandas as pd

def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    d = close.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))

def atr(df, n=14):
    pc = df.close.shift()
    tr = pd.concat([df.high - df.low, (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def add_indicators(df):
    d = df.copy()
    d["ema20"], d["ema50"] = ema(d.close, 20), ema(d.close, 50)
    d["rsi"] = rsi(d.close); d["atr"] = atr(d)
    macd = ema(d.close, 12) - ema(d.close, 26); d["macd"], d["macd_sig"] = macd, ema(macd, 9)
    d["vol_ratio"] = d.volume / d.volume.rolling(20).mean()
    d["ret1"], d["ret5"] = d.close.pct_change(), d.close.pct_change(5)
    return d

def pivots(df):
    """Classic floor-trader pivots from the previous session's H/L/C."""
    h, l, c = df.high.iloc[-2], df.low.iloc[-2], df.close.iloc[-2]
    p = (h + l + c) / 3
    return {"R2": p + (h - l), "R1": 2*p - l, "Pivot": p, "S1": 2*p - h, "S2": p - (h - l)}

def swing_levels(df, lookback=60, order=3):
    """Nearest swing-high resistance and swing-low support from recent history."""
    d = df.iloc[-lookback:]; hi, lo = d.high.values, d.low.values; res, sup = [], []
    for i in range(order, len(d) - order):
        if hi[i] == hi[i-order:i+order+1].max(): res.append(hi[i])
        if lo[i] == lo[i-order:i+order+1].min(): sup.append(lo[i])
    return sorted(set(np.round(sup, 2))), sorted(set(np.round(res, 2)))

def feature_frame(df, gap_w=0.10):
    """Per-day score for each row using ONLY info known at that day's open."""
    d = add_indicators(df)
    p = d.shift(1)                                    # yesterday's indicators
    atr_pct = (p.atr / p.close).replace(0, np.nan)
    trend = (p.close > p.ema20).astype(float) + (p.ema20 > p.ema50).astype(float) - 1   # -1..1
    mom = (p.ret5 / atr_pct).clip(-3, 3) / 3
    volx = (p.vol_ratio.clip(0, 3) - 1).clip(-1, 2) * np.sign(p.ret1.fillna(0))
    rsi_t = ((p.rsi - 50) / 50).where((p.rsi < 78) & (p.rsi > 22), -(p.rsi - 50) / 50)  # fade extremes
    gap = d.open / p.close - 1
    gap_t = (gap / atr_pct).clip(-2, 2) / 2
    d["score"] = 0.40*trend + 0.35*mom + 0.15*volx.clip(-1, 1) + 0.10*rsi_t + gap_w*gap_t
    d["gap"] = gap; d["atr_prev"] = p.atr; d["close_prev"] = p.close
    return d

def trade_levels(entry, atr_val, side, sl_mult=1.5, rr=2.0):
    risk = sl_mult * atr_val
    if side == "BUY":  return entry - risk, entry + rr * risk
    return entry + risk, entry - rr * risk

def rank_today(frames, gap_w=0.10, sl_mult=1.5, rr=2.0, top=3, min_score=0.15):
    rows = []
    for t, df in frames.items():
        f = feature_frame(df, gap_w).iloc[-1]
        if np.isnan(f.score) or abs(f.score) < min_score: continue
        side = "BUY" if f.score > 0 else "SELL"; entry = float(df.close.iloc[-1])
        sl, tgt = trade_levels(entry, f.atr_prev, side, sl_mult, rr)
        rows.append(dict(Ticker=t, Side=side, Entry=round(entry, 2), StopLoss=round(sl, 2),
            Target=round(tgt, 2), RiskPct=round(100*abs(entry-sl)/entry, 2), Score=round(abs(f.score), 3),
            GapPct=round(100*f.gap, 2), VolX=round(float(f.vol_ratio), 2), RSI=round(float(f.rsi), 1)))
    return pd.DataFrame(rows).sort_values("Score", ascending=False).head(top).reset_index(drop=True) if rows else pd.DataFrame()

def reaction_profile(df, vol_mult=2.0, move_sd=1.5):
    """How does this stock behave after high-volume breakout days? (proxy for 'what it reacts to')"""
    d = add_indicators(df); sd = d.ret1.rolling(60).std()
    ev = d[(d.vol_ratio >= vol_mult) & (d.ret1.abs() >= move_sd * sd)].copy()
    ev["next_ret"] = d.ret1.shift(-1).reindex(ev.index)
    ev["cont"] = np.sign(ev.next_ret) == np.sign(ev.ret1)
    ev = ev.dropna(subset=["next_ret"])
    if ev.empty: return None
    return dict(events=len(ev), continuation_rate=float(ev.cont.mean()),
                avg_next_day_pct=float(100 * (ev.next_ret * np.sign(ev.ret1)).mean()),
                avg_event_move_pct=float(100 * ev.ret1.abs().mean()))
