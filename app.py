import os, sqlite3
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data import UNIVERSE, DB, load_daily, load_intraday, refresh_db
from signals import add_indicators, pivots, swing_levels, rank_today, reaction_profile
from backtest import run_backtest

st.set_page_config(page_title="NSE Momentum Scanner", layout="wide")
st.title("NSE Momentum Scanner (MVP)")
st.caption("Decision-support tool for personal research. Not investment advice. Signals are heuristic; check the backtest.")

with st.sidebar:
    demo = st.toggle("Demo mode (synthetic data)", value=os.environ.get("DEMO") == "1")
    if st.button("Refresh data from Yahoo/NSE", disabled=demo):
        with st.spinner("Downloading 3y of daily bars..."):
            try: ok, failed = refresh_db(); st.success(f"{len(ok)} stocks saved. Failed: {failed or 'none'}")
            except Exception as e: st.error(f"Download failed: {e}")
    st.subheader("Strategy settings")
    top = st.slider("Picks per day", 1, 5, 3)
    sl_mult = st.slider("Stop-loss (x ATR)", 0.5, 3.0, 1.5, 0.1)
    rr = st.slider("Reward : Risk", 1.0, 4.0, 2.0, 0.5)
    gap_w = st.slider("Opening-gap weight", 0.0, 0.5, 0.10, 0.05)
    min_score = st.slider("Min signal strength", 0.0, 0.6, 0.15, 0.05)

@st.cache_data(ttl=300, show_spinner=False)
def get_frames(demo): return load_daily(UNIVERSE, demo=demo)

frames = get_frames(demo)
if not frames:
    st.warning("No data yet. Click 'Refresh data' in the sidebar (needs internet), or switch on Demo mode."); st.stop()
if demo: st.warning("DEMO MODE: prices are synthetic random data for testing the app only.")

tab1, tab2, tab3, tab4 = st.tabs(["Top picks", "Chart & levels", "Backtest", "SQL explorer"])

with tab1:
    last = max(df.index[-1] for df in frames.values())
    st.subheader(f"Top {top} ideas  (latest bar: {last.date()})")
    picks = rank_today(frames, gap_w, sl_mult, rr, top, min_score)
    if picks.empty: st.info("No stock passes the minimum signal strength today.")
    else:
        c = st.columns(len(picks))
        for col, (_, r) in zip(c, picks.iterrows()):
            col.metric(f"{r.Ticker} - {r.Side}", f"Rs {r.Entry}", f"Target {r.Target}")
            col.write(f"Stop-loss **{r.StopLoss}** ({r.RiskPct}% risk)  \nScore {r.Score} | Gap {r.GapPct}% | Vol x{r.VolX} | RSI {r.RSI}")
        st.dataframe(picks, width="stretch", hide_index=True)
    st.markdown("**How the score works:** trend (EMA20/EMA50) 40%, 5-day momentum vs volatility 35%, volume surge in the direction of "
                "the last move 15%, RSI (fades extremes) 10%, plus the opening gap weight you set. Stop-loss = ATR multiple; target = reward:risk.")
    st.markdown("**What does each stock react to?** (high-volume breakout days and what happens next day)")
    rows = []
    for t in (picks.Ticker if not picks.empty else []):
        p = reaction_profile(frames[t])
        if p: rows.append(dict(Ticker=t, **{k: round(v, 3) for k, v in p.items()}))
    if rows: st.dataframe(pd.DataFrame(rows), hide_index=True)

with tab2:
    t = st.selectbox("Stock", sorted(frames), index=0)
    mode = st.radio("Timeframe", ["Daily", "Intraday 5m (live)"], horizontal=True)
    if mode == "Daily":
        d = add_indicators(frames[t]).iloc[-150:]
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[.6, .2, .2], vertical_spacing=.03)
        fig.add_trace(go.Candlestick(x=d.index, open=d.open, high=d.high, low=d.low, close=d.close, name="Price"), 1, 1)
        fig.add_trace(go.Scatter(x=d.index, y=d.ema20, name="EMA20"), 1, 1)
        fig.add_trace(go.Scatter(x=d.index, y=d.ema50, name="EMA50"), 1, 1)
        for k, v in pivots(frames[t]).items(): fig.add_hline(y=v, line_dash="dot", annotation_text=k, row=1, col=1)
        sup, res = swing_levels(frames[t])
        for v in sup[-2:]: fig.add_hline(y=v, line_color="green", opacity=.4, annotation_text="support", row=1, col=1)
        for v in res[:2]:  fig.add_hline(y=v, line_color="red", opacity=.4, annotation_text="resistance", row=1, col=1)
        fig.add_trace(go.Bar(x=d.index, y=d.volume, name="Volume"), 2, 1)
        fig.add_trace(go.Scatter(x=d.index, y=d.rsi, name="RSI"), 3, 1)
        fig.add_hline(y=70, line_dash="dash", row=3, col=1); fig.add_hline(y=30, line_dash="dash", row=3, col=1)
        fig.update_layout(height=800, xaxis_rangeslider_visible=False, margin=dict(t=20))
        st.plotly_chart(fig, width="stretch")
    else:
        @st.fragment(run_every=60)
        def live():
            i = load_intraday(t, demo=demo)
            if i is None or i.empty: st.info("No intraday data (market closed or feed unavailable)."); return
            i = i.copy(); day = i.index.date; tp = (i.high + i.low + i.close) / 3
            i["vwap"] = (tp * i.volume).groupby(day).cumsum() / i.volume.groupby(day).cumsum()
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.75, .25], vertical_spacing=.03)
            fig.add_trace(go.Candlestick(x=i.index, open=i.open, high=i.high, low=i.low, close=i.close, name="5m"), 1, 1)
            fig.add_trace(go.Scatter(x=i.index, y=i.vwap, name="VWAP"), 1, 1)
            fig.add_trace(go.Bar(x=i.index, y=i.volume, name="Volume"), 2, 1)
            fig.update_layout(height=650, xaxis_rangeslider_visible=False, margin=dict(t=20))
            st.plotly_chart(fig, width="stretch"); st.caption("Auto-refreshes every 60 s. Yahoo data may be delayed ~1 min.")
        live()

with tab3:
    st.write("Walk-forward test: each day, take the top picks at the open, exit at stop-loss / target / close (0.1% round-trip cost).")
    days = st.slider("Days to test", 60, 700, 400, 20)
    if st.button("Run backtest"):
        with st.spinner("Running..."):
            tr, stt, eq = run_backtest(frames, top, gap_w, sl_mult, rr, min_score, 0.001, days)
        if not stt: st.info("No trades generated.")
        else:
            a = st.columns(4)
            a[0].metric("Win rate", f"{100*stt['win_rate']:.1f}%"); a[1].metric("Avg trade", f"{stt['avg_return_pct']:.2f}%")
            a[2].metric("Total return", f"{stt['total_return_pct']:.1f}%"); a[3].metric("Max drawdown", f"{stt['max_drawdown_pct']:.1f}%")
            st.write(f"Trades: {stt['trades']} | Avg win {stt['avg_win_pct']:.2f}% | Avg loss {stt['avg_loss_pct']:.2f}% | Sharpe {stt['sharpe']:.2f}")
            f = go.Figure(go.Scatter(x=eq.index, y=eq.values, name="Equity")); f.update_layout(height=350, title="Equity curve (start = 1.0)")
            st.plotly_chart(f, width="stretch")
            st.write("Exit reasons:", tr.reason.value_counts().to_dict()); st.dataframe(tr.tail(60), hide_index=True)
            if stt["total_return_pct"] <= 0 or stt["win_rate"] < .5: st.warning("This setting does not beat chance after costs. Do not trade it as is.")

with tab4:
    st.write("Your data lives in a SQLite database. Try SQL (SELECT only):")
    q = st.text_area("Query", "SELECT ticker, MAX(date) AS last_date, COUNT(*) AS bars, ROUND(AVG(volume)) AS avg_vol\nFROM prices GROUP BY ticker ORDER BY avg_vol DESC LIMIT 10")
    if st.button("Run query"):
        if not q.strip().lower().startswith("select"): st.error("SELECT only.")
        elif demo or not os.path.exists(DB): st.info("Needs real data: switch off Demo mode and refresh data first.")
        else:
            try:
                con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True); st.dataframe(pd.read_sql(q, con)); con.close()
            except Exception as e: st.error(e)
