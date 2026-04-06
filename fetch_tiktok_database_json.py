import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options


SCRIPT_ID_CANDIDATES = [
    "SIGI_STATE",
    "__UNIVERSAL_DATA_FOR_REHYDRATION__",
    "__NEXT_DATA__",
]


def make_driver(headless: bool = True) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--window-size=1400,900")
    opts.add_argument("--lang=en-US")
    driver = webdriver.Chrome(options=opts)
    driver.set_page_load_timeout(45)
    return driver


def extract_embedded_json_from_html(html: str) -> tuple[Optional[dict], Optional[str]]:
    for sid in SCRIPT_ID_CANDIDATES:
        m = re.search(
            rf'<script[^>]+id="{re.escape(sid)}"[^>]*>(.*?)</script>',
            html,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            blob = m.group(1).strip()
            blob = blob.replace("&quot;", '"').replace("&amp;", "&")
            try:
                return json.loads(blob), sid
            except json.JSONDecodeError:
                pass
    return None, None


def extract_video_id(url: str) -> Optional[str]:
    if not isinstance(url, str):
        return None
    m = re.search(r"/video/(\d+)", url)
    return m.group(1) if m else None


@dataclass
class FetchResult:
    input_url: str
    final_url: str
    video_id: Optional[str]
    status: str
    source_script: Optional[str]
    out_json_path: str


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sheet", default="tiktok_database.csv")
    ap.add_argument("--outdir", default="tiktok_database_metadata")
    ap.add_argument("--out-report", default="tiktok_database_metadata_fetch_report.csv")
    ap.add_argument("--headless", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=7.0)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.sheet, low_memory=False)
    if "LINK" not in df.columns:
        raise ValueError("Expected LINK column in tiktok_database.csv")

    urls = df["LINK"].dropna().astype(str).tolist()
    if args.limit >= 0:
        urls = urls[: args.limit]

    driver = make_driver(headless=bool(args.headless))
    results: list[FetchResult] = []
    try:
        for i, url in enumerate(urls, start=1):
            print(f"[{i}/{len(urls)}] {url}")
            try:
                driver.get(url)
                time.sleep(args.sleep)
                final_url = driver.current_url
                html = driver.page_source
                data, source = extract_embedded_json_from_html(html)
                video_id = extract_video_id(final_url) or extract_video_id(url)

                if not data:
                    results.append(FetchResult(url, final_url, video_id, "no_embedded_json", source, ""))
                    continue

                if not video_id:
                    # fallback: attempt item id from common shape
                    item_id = (
                        data.get("itemInfo", {})
                        .get("itemStruct", {})
                        .get("id")
                    ) if isinstance(data, dict) else None
                    video_id = str(item_id) if item_id else None

                name = f"{video_id}.json" if video_id else f"unknown_{i}.json"
                out_path = os.path.join(args.outdir, name)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)

                results.append(FetchResult(url, final_url, video_id, "ok", source, out_path))
            except Exception as e:
                results.append(FetchResult(url, "", None, f"error:{type(e).__name__}", None, ""))
    finally:
        driver.quit()

    rep = pd.DataFrame([r.__dict__ for r in results])
    rep.to_csv(args.out_report, index=False)
    print("\nstatus counts:")
    print(rep["status"].value_counts(dropna=False).to_string())
    print(f"\nWrote report: {args.out_report}")
    print(f"Wrote JSON dir: {args.outdir}")


if __name__ == "__main__":
    main()
