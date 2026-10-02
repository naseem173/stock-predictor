# NSE Momentum Scanner (MVP)

Run on your own computer (needs internet for real data):

    pip install -r requirements.txt
    streamlit run app.py

1. Click **Refresh data** in the sidebar (downloads 3 years of daily bars for 40 NSE stocks into market.db).
2. **Top picks** tab: top 3 ideas with entry, stop-loss, target. Open it at 9:30 IST.
3. **Chart & levels**: candlesticks, EMAs, pivots, support/resistance, RSI; intraday 5m + VWAP refreshes every 60s.
4. **Backtest**: run it BEFORE trusting any setting. Tune sliders and compare.
5. **SQL explorer**: query your own price database.

No internet? Toggle **Demo mode** (synthetic data, for testing the UI only).
Not investment advice. Yahoo data can be delayed; for true real-time swap `data.py` to a broker API (e.g. Kite Connect).
