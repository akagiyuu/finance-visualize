import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List

import pandas as pd
import requests
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.express as px

SYMBOLS_FILE = "symbols.txt"
SECTORS_FILE = "sectors.csv"
MARKETCAPS_FILE = "marketcaps.csv"
DEFAULT_WORKERS = 8
HTTP_TIMEOUT = 10

RANGE_TO_DAYS = {"week": 7, "month": 30, "3mo": 90, "6mo": 180, "year": 365}


def fetch_history(symbol: str, count: int, timeout=HTTP_TIMEOUT):
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

    for c in ("open", "high", "low", "close"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    vol_cols = [c for c in df.columns if c.lower() in ("kl", "volume", "vol", "khoiluong", "khoi_luong")]
    if vol_cols:
        df["volume"] = pd.to_numeric(df[vol_cols[0]], errors="coerce")

    df = df.dropna(subset=["date"])
    if "close" in df.columns:
        df = df.dropna(subset=["close"])
    df = df.sort_values("date").set_index("date")
    return df


def read_symbols(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]
    return [l.upper().replace(".", "-") for l in lines]


def read_sector_file(path: str) -> Dict[str, Dict]:
    df = pd.read_csv(path, dtype=str, header=0)
    cols = [c.lower() for c in df.columns]
    mapping = {}
    if "symbol" in cols and "sector" in cols:
        s_col = cols.index("symbol")
        sec_col = cols.index("sector")
        name_col = cols.index("name") if "name" in cols else None
        for _, row in df.iterrows():
            sym = str(row.iloc[s_col]).upper()
            name = str(row.iloc[name_col]) if name_col is not None and pd.notna(row.iloc[name_col]) else sym
            mapping[sym] = {"sector": str(row.iloc[sec_col]) if pd.notna(row.iloc[sec_col]) else "Unknown", "name": name}
    else:
        for _, row in df.iterrows():
            sym = str(row.iloc[0]).upper()
            sec = str(row.iloc[1]) if len(row) > 1 and pd.notna(row.iloc[1]) else "Unknown"
            name = str(row.iloc[2]) if len(row) > 2 and pd.notna(row.iloc[2]) else sym
            mapping[sym] = {"sector": sec, "name": name}
    return mapping


def read_marketcaps(path: str) -> Dict[str, float]:
    df = pd.read_csv(path, dtype=str, header=0)
    cols = [c.lower() for c in df.columns]
    mapping = {}
    if "symbol" in cols and ("market_cap" in cols or "marketcap" in cols or "market-cap" in cols):
        s_col = cols.index("symbol")
        mc_col = None
        for candidate in ("market_cap", "marketcap", "market-cap"):
            if candidate in cols:
                mc_col = cols.index(candidate)
                break
        for _, row in df.iterrows():
            sym = str(row.iloc[s_col]).upper()
            try:
                mapping[sym] = float(str(row.iloc[mc_col]).replace(",", "").strip())
            except Exception:
                mapping[sym] = None
    else:
        for _, row in df.iterrows():
            sym = str(row.iloc[0]).upper()
            try:
                mapping[sym] = float(str(row.iloc[1]).replace(",", "").strip())
            except Exception:
                mapping[sym] = None
    return mapping


def process_symbol(symbol: str, days: int, timeout=HTTP_TIMEOUT) -> Dict:
    try:
        raw = fetch_history(symbol, count=days, timeout=timeout)
        df = normalize_df(raw)
        if df.empty or "close" not in df.columns:
            return {"symbol": symbol, "price": None, "change": None, "change_pct": None, "avg_volume": None, "n_points": 0}
        first = float(df["close"].iloc[0])
        last = float(df["close"].iloc[-1])
        change = last - first
        change_pct = (change / first * 100.0) if first != 0 else None
        avg_vol = None
        if "volume" in df.columns:
            vol = df["volume"].dropna()
            if not vol.empty:
                avg_vol = float(vol.mean())
        return {"symbol": symbol, "price": last, "change": change, "change_pct": change_pct, "avg_volume": avg_vol, "n_points": len(df)}
    except Exception as e:
        return {"symbol": symbol, "price": None, "change": None, "change_pct": None, "avg_volume": None, "n_points": 0, "error": str(e)}


def build_tv_json_and_heatmap(items: List[Dict], sector_map: Dict[str, Dict], mc_map: Dict[str, float], range_name: str):
    flat = []
    for it in items:
        sym = it["symbol"]
        if it.get("change_pct") is None:
            continue
        name = sym
        sector = "Unknown"
        if sector_map and sym in sector_map:
            name = sector_map[sym].get("name", sym)
            sector = sector_map[sym].get("sector", "Unknown")
        mc = None
        if mc_map and sym in mc_map and mc_map[sym] is not None and not (isinstance(mc_map[sym], float) and math.isnan(mc_map[sym])):
            mc = float(mc_map[sym])
        else:
            if it.get("avg_volume") and it.get("price"):
                try:
                    mc = float(it["avg_volume"]) * float(it["price"])
                except Exception:
                    mc = None
            else:
                mc = None

        entry = {
            "symbol": sym,
            "name": name,
            "sector": sector,
            "price": None if it["price"] is None else float(it["price"]),
            "change": None if it["change"] is None else float(it["change"]),
            "change_pct": None if it["change_pct"] is None else float(it["change_pct"]),
            "market_cap_basic": None if mc is None else float(mc),
            "volume": None if it.get("avg_volume") is None else float(it["avg_volume"]),
            "n_points": int(it.get("n_points", 0))
        }
        flat.append(entry)

    sectors: Dict[str, Dict] = {}
    for e in flat:
        sec = e["sector"] or "Unknown"
        sectors.setdefault(sec, {"name": sec, "sector_market_cap": 0.0, "items": []})
        sectors[sec]["items"].append(e)
        if e["market_cap_basic"] is not None:
            sectors[sec]["sector_market_cap"] += e["market_cap_basic"]

    sectors_list = []
    for sec_name, sec_data in sectors.items():
        smc = sec_data["sector_market_cap"] if sec_data["sector_market_cap"] > 0 else None
        sectors_list.append({"name": sec_name, "sector_market_cap": smc, "items": sec_data["items"]})

    meta = {
        "dataSource": "CAFEEF_HISTORY",
        "range": range_name,
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": "Blocks are uniform size; color = % change"
    }

    tv_json = {"meta": meta, "sectors": sectors_list, "flat": flat}

    df = pd.DataFrame(flat)
    if df.empty:
        raise SystemExit("No valid items to plot (no data)")

    df["size_uniform"] = 1

    fig = px.treemap(
        df,
        path=["sector", "symbol"],
        values="size_uniform",
        color="change_pct",
        hover_data={"name": True, "change_pct": True},
        labels={"change_pct": "% change"},
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
    )
    fig.update_traces(hovertemplate="%{label}<br>%{customdata[1]:.2f}%")
    fig.update_layout(margin=dict(t=50, l=25, r=25, b=25), title=f"Heatmap-style blocks (range={range_name})")

    html_out = f"heatmap_{range_name}.html"
    json_out = f"tv_data_{range_name}.json"
    fig.write_html(html_out)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(tv_json, f, ensure_ascii=False, indent=2)

    print("Wrote HTML:", html_out)
    print("Wrote JSON:", json_out)


def main():
    p = argparse.ArgumentParser(description="Heatmap-style blocks from cafef — only change range.")
    p.add_argument("--range", default="month", choices=list(RANGE_TO_DAYS.keys()), help="Range: week, month, 3mo, 6mo, year")
    args = p.parse_args()

    try:
        symbols = read_symbols(SYMBOLS_FILE)
    except FileNotFoundError:
        raise SystemExit(f"Missing {SYMBOLS_FILE}")

    sector_map = {}
    try:
        sector_map = read_sector_file(SECTORS_FILE)
        print(f"Loaded sectors from {SECTORS_FILE}")
    except FileNotFoundError:
        print(f"No {SECTORS_FILE} — defaulting to Unknown sector")

    mc_map = {}
    try:
        mc_map = read_marketcaps(MARKETCAPS_FILE)
        print(f"Loaded market caps from {MARKETCAPS_FILE}")
    except FileNotFoundError:
        print(f"No {MARKETCAPS_FILE} — will estimate sizes")

    days = RANGE_TO_DAYS[args.range]
    results = []
    with ThreadPoolExecutor(max_workers=DEFAULT_WORKERS) as ex:
        futures = {ex.submit(process_symbol, sym, days, HTTP_TIMEOUT): sym for sym in symbols}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Fetching"):
            res = fut.result()
            results.append(res)

    build_tv_json_and_heatmap(results, sector_map, mc_map, args.range)


if __name__ == "__main__":
    main()
