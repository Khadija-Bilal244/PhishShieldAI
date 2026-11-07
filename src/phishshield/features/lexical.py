"""Engineered lexical / host-based URL features.

Pure Python + numpy (no network calls, no torch) so the exact same code runs in
training, in the FastAPI container and in tests -> no training/serving skew.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

SUSPICIOUS_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "icu", "shop", "online", "site", "live",
                   "click", "support", "info", "biz", "rest", "cam", "buzz", "monster", "work"}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "rb.gy", "cutt.ly", "is.gd", "shorturl.at", "goo.gl",
              "ow.ly", "buff.ly", "tiny.cc", "s.id"}
CLOUD_HOSTS = ("web.app", "firebaseapp.com", "github.io", "netlify.app", "vercel.app", "pages.dev",
               "weebly.com", "wixsite.com", "000webhostapp.com", "glitch.me", "sites.google.com",
               "forms.gle", "storage.googleapis.com", "r2.dev", "herokuapp.com", "blogspot.com")
KEYWORDS = ("login", "signin", "verify", "secure", "account", "update", "confirm", "banking", "webscr",
            "support", "unlock", "suspend", "billing", "wallet", "recovery", "auth", "password", "ebayisapi")
BRANDS = ("paypal", "apple", "microsoft", "amazon", "netflix", "facebook", "instagram", "chase",
          "wellsfargo", "bankofamerica", "office365", "outlook", "dhl", "fedex", "hbl", "meezan",
          "easypaisa", "jazzcash", "daraz", "binance", "coinbase", "metamask", "steam", "whatsapp", "google")
IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
HEX_RE = re.compile(r"%[0-9a-fA-F]{2}")

FEATURE_NAMES: list[str] = [
    "url_length", "host_length", "path_length", "query_length", "num_dots", "num_hyphens",
    "num_underscores", "num_slashes", "num_digits", "digit_ratio", "num_special", "has_at",
    "has_ip", "has_port", "is_https", "num_subdomains", "tld_suspicious", "tld_length",
    "entropy_url", "entropy_host", "has_punycode", "is_shortener", "is_cloud_host",
    "num_keywords", "brand_mismatch", "double_slash_path", "longest_token", "host_digit_letter_mix",
    "num_query_params", "has_hex_encoding", "host_vowel_ratio",
]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def _parse(url: str):
    try:
        return urlsplit(url if "://" in url else "http://" + url)
    except ValueError:
        return urlsplit("http://invalid")


def extract(url: str) -> dict[str, float]:
    url = str(url).strip()
    p = _parse(url)
    netloc = p.netloc.lower()
    host = netloc.rsplit("@", 1)[-1]
    port = None
    if ":" in host:
        host, _, port = host.partition(":")
    labels = [x for x in host.split(".") if x]
    tld = labels[-1] if labels else ""
    registered = ".".join(labels[-2:]) if len(labels) >= 2 else host
    low = url.lower()
    tokens = [t for t in re.split(r"[/\.\-_?=&@:]", low) if t]
    letters = [c for c in host if c.isalpha()]

    brand_hits = [b for b in BRANDS if b in low]
    # brand appears in the URL but the registered domain is not that brand (e.g. paypal-login.xyz, x.com/paypal)
    brand_mismatch = float(any(registered.split(".")[0] != b for b in brand_hits)) if brand_hits else 0.0
    mix = sum(1 for a, b in zip(host, host[1:]) if a.isalpha() != b.isalpha() and (a.isdigit() or b.isdigit()))

    feats = {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(p.path),
        "query_length": len(p.query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": sum(c.isdigit() for c in url),
        "digit_ratio": sum(c.isdigit() for c in url) / max(len(url), 1),
        "num_special": sum(url.count(c) for c in "@?=&%~!*,;$"),
        "has_at": float("@" in netloc),
        "has_ip": float(bool(IP_RE.match(host))),
        "has_port": float(port is not None and port != ""),
        "is_https": float(p.scheme == "https"),
        "num_subdomains": max(len(labels) - 2, 0),
        "tld_suspicious": float(tld in SUSPICIOUS_TLDS),
        "tld_length": len(tld),
        "entropy_url": _entropy(low),
        "entropy_host": _entropy(host),
        "has_punycode": float("xn--" in host),
        "is_shortener": float(host in SHORTENERS or registered in SHORTENERS),
        "is_cloud_host": float(any(c in low for c in CLOUD_HOSTS)),
        "num_keywords": sum(low.count(k) for k in KEYWORDS),
        "brand_mismatch": brand_mismatch,
        "double_slash_path": float("//" in p.path),
        "longest_token": max((len(t) for t in tokens), default=0),
        "host_digit_letter_mix": mix,
        "num_query_params": len([q for q in p.query.split("&") if q]),
        "has_hex_encoding": float(bool(HEX_RE.search(url))),
        "host_vowel_ratio": sum(c in "aeiou" for c in letters) / max(len(letters), 1),
    }
    return {k: float(feats[k]) for k in FEATURE_NAMES}


def extract_many(urls) -> pd.DataFrame:
    return pd.DataFrame([extract(u) for u in urls], columns=FEATURE_NAMES)


def to_matrix(urls) -> np.ndarray:
    return extract_many(urls).to_numpy(dtype=np.float32)
