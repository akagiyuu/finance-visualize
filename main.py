import argparse
import requests
import pandas as pd


def fetch_history(symbol: str, count: int, timeout=10):
    url = f"https://cafef.vn/du-lieu/Ajax/PageNew/DataHistory/PriceHistory.ashx?Symbol={symbol}&PageIndex=1&PageSize={count}"
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()["Data"]["Data"]


def normalize_df(raw):
    df = pd.DataFrame(raw)
    df.columns = [c.lower() for c in df.columns]

    df = df.rename(
        columns={
            "ngay": "date",
            "giamocua": "open",
            "giadongcua": "close",
            "giacaonhat": "high",
            "giathapnhat": "low",
        }
    )

    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")

    # ensure numeric OHLC
    for c in ("open", "high", "low", "close"):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df.sort_values("date").set_index("date")
    return df


def plot_plotly(df: pd.DataFrame):
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="Candlestick",
            )
        ]
    )
    fig.update_layout(title="Candlestick", xaxis_rangeslider_visible=False)
    fig.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--count", required=True)
    args = p.parse_args()

    raw = fetch_history(args.symbol, args.count)
    df = normalize_df(raw)
    plot_plotly(df)


if __name__ == "__main__":
    main()
