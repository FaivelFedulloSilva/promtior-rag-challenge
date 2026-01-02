import os
import re
import json
import gzip
import asyncio
from typing import List
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv
from tqdm import tqdm

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def is_same_domain(url: str, base: str) -> bool:
    return urlparse(url).netloc == urlparse(base).netloc


def looks_like_asset(url: str) -> bool:
    u = url.lower()
    return any(u.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg", ".pdf", ".zip", ".rar", ".7z", ".mp4", ".mp3", ".css", ".js"))


def fetch_text(url: str, headers: dict) -> str | None:
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code != 200:
            return None
        content = r.content
        if url.endswith(".gz"):
            content = gzip.decompress(content)
        return content.decode("utf-8", errors="ignore")
    except Exception:
        return None


def parse_sitemap_urls(xml: str) -> List[str]:
    # simple loc parser (good enough)
    urls = re.findall(r"<loc>(.*?)</loc>", xml)
    return [u.strip() for u in urls if u.strip()]


def get_sitemap_urls(base_url: str, user_agent: str) -> List[str]:
    """
    Supports:
      - urlset sitemap
      - sitemapindex with nested sitemaps
      - .gz variants
    """
    headers = {"User-Agent": user_agent}

    candidates = [
        base_url.rstrip("/") + "/sitemap.xml",
        base_url.rstrip("/") + "/sitemap_index.xml",
        base_url.rstrip("/") + "/sitemap.xml.gz",
        base_url.rstrip("/") + "/sitemap_index.xml.gz",
    ]

    for sm in candidates:
        xml = fetch_text(sm, headers=headers)
        if not xml:
            continue

        if "<urlset" in xml:
            urls = parse_sitemap_urls(xml)
            return urls

        if "<sitemapindex" in xml:
            submaps = parse_sitemap_urls(xml)
            all_urls: List[str] = []
            for sub in submaps:
                sub_xml = fetch_text(sub, headers=headers)
                if sub_xml and "<urlset" in sub_xml:
                    all_urls.extend(parse_sitemap_urls(sub_xml))
            return all_urls

    return []


async def run(base_url: str, max_pages: int, out_path: str, min_len: int = 200):
    ua = os.getenv("USER_AGENT", "promtior-rag-challenge/0.1 (contact: you@example.com)")

    sitemap_urls = get_sitemap_urls(base_url, ua)
    # filter + de-dup
    urls = []
    seen = set()
    for u in sitemap_urls:
        if not u.startswith(("http://", "https://")):
            continue
        if not is_same_domain(u, base_url):
            continue
        if looks_like_asset(u):
            continue
        if u not in seen:
            seen.add(u)
            urls.append(u)

    if not urls:
        raise RuntimeError(
            "No usable URLs found in sitemap. "
            "Either sitemap is empty/blocked, or the site doesn't expose URLs. "
            "Next step: manually seed URLs."
        )

    # limit
    urls = urls[:max_pages]
    print(f"[sitemap] urls_found={len(sitemap_urls)} usable={len(urls)} using_first={len(urls)}")

    browser_cfg = BrowserConfig(headless=True, user_agent=ua)
    run_cfg = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        async with AsyncWebCrawler(config=browser_cfg) as crawler:
            for url in tqdm(urls, desc="Render & export", unit="page"):
                result = await crawler.arun(url=url, config=run_cfg)

                md = clean_text(getattr(result, "markdown", "") or "")
                if len(md) < min_len:
                    continue

                f.write(json.dumps({"source": url, "text": md}, ensure_ascii=False) + "\n")
                kept += 1

    print(f"[done] kept={kept} out={out_path}")


def main():
    load_dotenv()
    base_url = os.getenv("PROMTIOR_BASE_URL", "https://promtior.ai/").strip()
    max_pages = int(os.getenv("CRAWL_MAX_PAGES", "60"))
    out_path = os.getenv("CRAWL_OUT", "./data/promtior_docs.jsonl")

    asyncio.run(run(base_url=base_url, max_pages=max_pages, out_path=out_path))


if __name__ == "__main__":
    main()
