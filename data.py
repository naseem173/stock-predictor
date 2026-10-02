"""Data layer: downloads NSE prices (yfinance), stores them in SQLite, and serves them to the app.
Set env DEMO=1 (or the sidebar toggle) to use synthetic data when offline."""
import os, sqlite3
import numpy as np
import pandas as pd

DB = os.path.join(os.path.dirname(__file__), "market.db")

UNIVERSE = ["RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","ITC","SBIN","BHARTIARTL",
    "KOTAKBANK","LT","AXISBANK","BAJFINANCE","ASIANPAINT","MARUTI","SUNPHARMA","TITAN","ULTRACEMCO",
    "NESTLEIND","WIPRO","HCLTECH","M&M","NTPC","POWERGRID","TATASTEEL","JSWSTEEL","ADANIENT",
    "ADANIPORTS","COALINDIA","ONGC","BAJAJFINSV","TECHM","INDUSINDBK","CIPLA","DRREDDY",
    "EICHERMOT","GRASIM","HINDALCO","BPCL","TRENT"]

def _yf(sym): return sym + ".NS"

def init_db():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS prices(
        ticker TEXT, date TEXT, open REAL, high REAL, low REAL, close REAL, volume REAL,
        PRIMARY KEY(ticker,date))""")
    con.commit(); return con

def refresh_db(tickers=UNIVERSE, period="3y"):
    """Download daily bars and upsert into SQLite. Returns (ok, failed) ticker lists."""
    import yfinance as yf
    con = init_db(); ok, failed = [], []
    raw = yf.download([_yf(t) for t in tickers], period=period, interval="1d",
                      group_by="ticker", auto_adjust=False, progress=False, threads=True)
    for t in tickers:
        try:
            df = raw[_yf(t)].dropna(subset=["Close"])
            if df.empty: raise ValueError("empty")
            rows = [(t, d.strftime("%Y-%m-%d"), r.Open, r.High, r.Low, r.Close, r.Volume)
                    for d, r in df.iterrows()]
            con.executemany("INSERT OR REPLACE INTO prices VALUES(?,?,?,?,?,?,?)", rows)
            ok.append(t)
        except Exception:
            failed.append(t)
    con.commit(); con.close()
    return ok, failed

def load_daily(tickers=UNIVERSE, demo=False):
    """Returns {ticker: DataFrame(open,high,low,close,volume) indexed by date}."""
    if demo: return synthetic_daily(tickers)
    con = init_db()
    df = pd.read_sql("SELECT * FROM prices", con, parse_dates=["date"]); con.close()
    out = {}
    for t, g in df.groupby("ticker"):
        if t in tickers and len(g) > 80:
            out[t] = g.set_index("date").sort_index()[["open","high","low","close","volume"]]
    return out

def load_intraday(ticker, interval="5m", period="5d", demo=False):
    if demo: return synthetic_intraday()
    import yfinance as yf
    df = yf.download(_yf(ticker), period=period, interval=interval, progress=False, auto_adjust=False)
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df[["open","high","low","close","volume"]]

# ---------- synthetic data (offline testing only; NOT real prices) ----------
def synthetic_daily(tickers, n=700, seed=7):
    rng = np.random.default_rng(seed); idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    out = {}
    for t in tickers:
        vol = rng.uniform(0.01, 0.022); p0 = rng.uniform(200, 4000)
        r = rng.normal(0.0003, vol, n); close = p0 * np.exp(np.cumsum(r))
        gap = rng.normal(0, vol * 0.4, n); open_ = np.r_[close[0], close[:-1]] * (1 + gap)
        rng_ = np.abs(rng.normal(0, vol, n)) + vol * 0.5
        high = np.maximum(open_, close) * (1 + rng_ * 0.5); low = np.minimum(open_, close) * (1 - rng_ * 0.5)
        v = rng.lognormal(14, 0.4, n) * (1 + 3 * np.abs(r) / vol / 3)
        out[t] = pd.DataFrame(dict(open=open_, high=high, low=low, close=close, volume=v), index=idx)
    return out

def synthetic_intraday(days=3, seed=3):
    rng = np.random.default_rng(seed); rows = []; p = 1000.0
    for d in pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days):
        for ts in pd.date_range(d + pd.Timedelta(hours=9, minutes=15), periods=75, freq="5min"):
            o = p; p = p * (1 + rng.normal(0, 0.0015)); h = max(o, p) * 1.0005; l = min(o, p) * 0.9995
            rows.append((ts, o, h, l, p, rng.lognormal(10, .5)))
    return pd.DataFrame(rows, columns=["ts","open","high","low","close","volume"]).set_index("ts")
