import argparse
import sys
import tempfile
import webbrowser
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import yfinance as yf

# Use days for robust start-date calculation (bulk download)
PERIOD_DAYS = {
    "1w": 7,
    "1m": 30,
    "3m": 90,
    "6m": 182,
    "1y": 365,
}


def read_symbols(path):
    """Read symbols file. Keep .VN suffix if present; otherwise append .VN as you had before."""
    with open(path, "r", encoding="utf-8") as f:
        syms = []
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            # keep uppercase, avoid double suffix
            s = s.upper()
            if not s.endswith(".VN"):
                s = f"{s}.VN"
            syms.append(s)
    # dedupe while preserving order
    return list(dict.fromkeys(syms))


def bulk_download_closes(symbols, start_date, end_date):
    """
    Download adjusted close prices for all symbols in a single call.
    Returns a dict: symbol -> pd.Series of adjusted closes (datetime indexed).
    """
    if not symbols:
        return {}

    # yf.download with multiple tickers returns columns as MultiIndex (ticker, field)
    # Use auto_adjust True to get adjusted prices
    df = yf.download(
        tickers=" ".join(symbols),
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        threads=True,
        progress=False,
    )

    closes = {}
    # detect format: multiindex columns when multiple tickers, single-level when one ticker
    if isinstance(df.columns, pd.MultiIndex):
        # df[ (ticker, 'Close') ] is series for that ticker
        for ticker in symbols:
            try:
                series = df[(ticker, "Close")].dropna()
            except Exception:
                # fallback: maybe ticker not present
                series = pd.Series(dtype=float)
            closes[ticker] = series
    else:
        # single ticker case: df['Close'] present
        # map the only symbol to that column
        ticker = symbols[0]
        if "Close" in df.columns:
            closes[ticker] = df["Close"].dropna()
        else:
            closes[ticker] = pd.Series(dtype=float)

    return closes


def fetch_metadata(symbol):
    """
    Fetch sector and marketCap via yfinance Ticker.info (best-effort).
    Return (sector, marketCap).
    """
    try:
        t = yf.Ticker(symbol)
        info = t.get_info() if hasattr(t, "get_info") else t.info
        if info:
            sector = info.get("sector") or info.get("industry") or "Unknown"
            marketCap = info.get("marketCap") or info.get("market_cap") or 0
            # ensure numeric
            if marketCap is None:
                marketCap = 0
            return sector, marketCap
    except Exception:
        pass
    return "Unknown", 0


def build_dataframe(symbols, days):
    """
    1) bulk download closes from start_date -> today
    2) compute pct_change per ticker using first available within window -> last available
    3) fetch per-ticker metadata (sector, marketCap)
    """
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=days)

    print(f"Bulk downloading price data from {start_date} to {end_date} ...")
    closes = bulk_download_closes(symbols, start_date, end_date)

    rows = []
    for idx, sym in enumerate(symbols, 1):
        series = closes.get(sym, pd.Series(dtype=float))
        # compute pct using first and last available closes in the window
        pct = 0.0
        if series is not None and len(series.dropna()) >= 2:
            s = series.dropna()
            first = float(s.iloc[0])
            last = float(s.iloc[-1])
            if first != 0:
                pct = (last - first) / first * 100.0
            else:
                pct = 0.0
        else:
            # fallback: try to fetch a tiny history per-ticker (rare)
            try:
                t = yf.Ticker(sym)
                hist = t.history(period="1mo", interval="1d", auto_adjust=True)
                if hist is not None and hist.shape[0] >= 2:
                    f = hist["Close"].iloc[0]
                    l = hist["Close"].iloc[-1]
                    pct = (l - f) / f * 100.0 if f != 0 else 0.0
                else:
                    pct = 0.0
            except Exception:
                pct = 0.0

        sector, marketCap = fetch_metadata(sym)

        # display symbol: strip .VN for compact labels
        display = sym.split(".")[0] if sym.upper().endswith(".VN") else sym

        rows.append(
            {
                "symbol": display,
                "raw_symbol": sym,
                "sector": sector or "Unknown",
                "marketCap": float(marketCap or 0),
                "pct_change": float(pct),
            }
        )
        print(
            f"[{idx}/{len(symbols)}] {sym}  pct={pct:+.2f}%  sector={sector}  mcap={marketCap}"
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No data fetched for any symbol.")

    # Replace zero or missing marketCap with a small positive number so treemap sizes behave
    df["marketCap"] = df["marketCap"].fillna(0.0)
    positive_mcaps = df.loc[df["marketCap"] > 0, "marketCap"]
    min_positive = positive_mcaps.min() if not positive_mcaps.empty else 1.0
    df.loc[df["marketCap"] <= 0, "marketCap"] = float(min_positive * 0.01)

    return df


def make_treemap(df, period_label):
    # Add label for hover/leaf content (we'll show pct in hover)
    df["label_pct"] = df["pct_change"].map(lambda v: f"{v:+.2f}%")
    # Use custom_data explicitly to control hovertemplate indices
    custom_cols = ["raw_symbol", "marketCap", "pct_change"]
    fig = px.treemap(
        df,
        path=["sector", "symbol"],
        values="marketCap",
        color="pct_change",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        custom_data=custom_cols,
    )

    # Show only symbol text inside boxes (keeps layout readable). Percent shown in hover.
    fig.update_traces(
        hovertemplate=(
            "<b>%{label}</b><br>"
            "raw: %{customdata[0]}<br>"
            "Market cap: %{customdata[1]:,.0f}<br>"
            "Change: %{customdata[2]:+.2f}%<extra></extra>"
        ),
        textinfo="label",
    )

    fig.update_layout(
        title=f"Interactive Market Treemap — period: {period_label}",
        margin=dict(t=50, l=25, r=25, b=25),
        coloraxis_colorbar=dict(title="% change"),
    )
    return fig


def open_fig_in_browser(fig):
    html = fig.to_html(full_html=True, include_plotlyjs="cdn")
    # write to a temp HTML file
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix=".html", mode="w", encoding="utf-8"
    )
    tmp.write(html)
    tmp.flush()
    tmp.close()
    path = tmp.name
    file_url = "file://" + path
    webbrowser.open(file_url, new=2)
    print("Opened local file:", path)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive treemap (sector -> ticker) using yfinance & plotly."
    )
    parser.add_argument(
        "symbols_file", help="Path to .txt file with symbols (one per line)."
    )
    parser.add_argument(
        "--period",
        "-p",
        choices=list(PERIOD_DAYS.keys()),
        default="1m",
        help="Period: 1w,1m,3m,6m,1y (default 1m).",
    )
    args = parser.parse_args()

    symbols = read_symbols(args.symbols_file)
    if not symbols:
        print("No symbols found.", file=sys.stderr)
        sys.exit(1)

    days = PERIOD_DAYS[args.period]
    print(
        f"Building treemap for {len(symbols)} symbols, period={args.period} ({days} days) ..."
    )

    df = build_dataframe(symbols, days)
    fig = make_treemap(df, args.period)
    print("Opening interactive treemap in your default browser...")
    open_fig_in_browser(fig)


if __name__ == "__main__":
    main()
