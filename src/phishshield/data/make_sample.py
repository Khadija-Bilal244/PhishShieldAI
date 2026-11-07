"""Synthetic, time-stamped phishing URL generator.

Why synthetic data at all?
  * The whole pipeline (DVC, CI, Docker, Airflow) can run offline in minutes.
  * Each time period has a *controlled* mix of attack styles, so concept/data drift is
    known in advance - this is what lets us evaluate drift detectors and update policies
    fairly (the proposal states simulated shifts are reported separately from real data).

Drift design (phishing side):
  periods 0-2: IP-address hosts, '@' tricks, long keyword-stuffed subdomains, http
  periods 3-5: typosquats, punycode/homoglyphs, URL shorteners, abuse of cloud hosting
               (web.app, firebase, github.io ...), cheap new TLDs, mostly https
Legitimate side also shifts (more SaaS/cloud apps, small sites on new TLDs, tracking params),
and contains hard negatives (real login/OAuth pages, brand names in paths, shortened links).
1.5% label noise mimics imperfect blocklist labels.

For real data use `python -m phishshield.data.ingest` (data.source: real).
"""

from __future__ import annotations

import argparse
import random
import string
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from phishshield.config import RAW_DIR, ensure_dirs, load_params, set_seed
from phishshield.utils.common import get_logger

log = get_logger("make_sample")

LEGIT_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "wikipedia.org", "amazon.com", "twitter.com",
    "instagram.com", "linkedin.com", "microsoft.com", "apple.com", "github.com", "stackoverflow.com",
    "reddit.com", "netflix.com", "paypal.com", "dropbox.com", "adobe.com", "zoom.us", "bbc.co.uk",
    "nytimes.com", "cnn.com", "dawn.com", "daraz.pk", "nu.edu.pk", "hec.gov.pk", "olx.com.pk",
    "medium.com", "spotify.com", "yahoo.com", "bing.com", "office.com", "live.com", "outlook.com",
    "chase.com", "bankofamerica.com", "wellsfargo.com", "hbl.com", "meezanbank.com", "ebay.com",
    "aliexpress.com", "walmart.com", "imdb.com", "quora.com", "pinterest.com", "tumblr.com",
    "kaggle.com", "huggingface.co", "arxiv.org", "coursera.org", "udemy.com", "khanacademy.org",
    "w3schools.com", "python.org", "pytorch.org", "tensorflow.org", "docker.com", "aws.amazon.com",
    "cloud.google.com", "azure.microsoft.com", "notion.so", "slack.com", "trello.com", "atlassian.com",
]
LEGIT_CLOUD = ["vercel.app", "netlify.app", "herokuapp.com", "pages.dev", "github.io"]
BRANDS = ["paypal", "apple", "microsoft", "amazon", "netflix", "facebook", "instagram", "chase",
          "wellsfargo", "bankofamerica", "office365", "outlook", "dhl", "fedex", "hbl", "meezan",
          "easypaisa", "jazzcash", "daraz", "binance", "coinbase", "metamask", "steam", "whatsapp"]
PHISH_WORDS = ["login", "signin", "verify", "secure", "account", "update", "confirm", "banking",
               "webscr", "support", "unlock", "suspended", "billing", "wallet", "recovery", "auth"]
OLD_TLDS = ["tk", "ml", "ga", "cf", "gq", "com", "net", "info", "biz"]
NEW_TLDS = ["xyz", "top", "icu", "shop", "online", "site", "live", "click", "support", "app"]
SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "rb.gy", "cutt.ly", "is.gd", "shorturl.at"]
ABUSED_HOSTS = ["web.app", "firebaseapp.com", "github.io", "netlify.app", "vercel.app",
                "pages.dev", "weebly.com", "wixsite.com", "000webhostapp.com", "glitch.me",
                "sites.google.com/view", "forms.gle", "storage.googleapis.com", "r2.dev"]
HOMOGLYPHS = {"a": ["4", "@", "à"], "o": ["0", "ò"], "l": ["1", "I"], "i": ["1", "l", "í"],
              "e": ["3", "è"], "s": ["5", "$"], "m": ["rn"], "w": ["vv"]}
LEGIT_PATHS = ["", "/", "/about", "/help", "/search?q={w}", "/news/{w}-{n}", "/products/{w}",
               "/user/{w}", "/blog/{y}/{w}-{w2}", "/docs/{w}/{w2}", "/watch?v={id}", "/wiki/{W}",
               "/en-us/{w}", "/questions/{n}/{w}-{w2}", "/r/{w}/comments/{id}", "/p/{id}"]
WORDS = ["python", "data", "cloud", "music", "travel", "sports", "health", "finance", "machine",
         "learning", "news", "weather", "recipes", "football", "cricket", "lahore", "karachi",
         "laptop", "phone", "books", "movies", "science", "history", "design", "jobs", "career"]
TRACKING = ["utm_source=newsletter", "utm_medium=email", "utm_campaign=autumn", "ref=homepage",
            "fbclid={id}", "gclid={id}", "lang=en", "page={n}"]

# attack-style probabilities per period (rows: period 0..5)
PHISH_MIX = {
    #               ip    at   kwsub long  typo  puny  short cloud newtld compromised
    0: np.array([0.22, 0.13, 0.27, 0.18, 0.07, 0.00, 0.00, 0.03, 0.00, 0.10]),
    1: np.array([0.19, 0.10, 0.27, 0.18, 0.09, 0.00, 0.01, 0.04, 0.01, 0.11]),
    2: np.array([0.15, 0.08, 0.25, 0.16, 0.12, 0.02, 0.02, 0.05, 0.02, 0.13]),
    3: np.array([0.06, 0.03, 0.15, 0.08, 0.17, 0.05, 0.09, 0.14, 0.08, 0.15]),
    4: np.array([0.03, 0.02, 0.10, 0.05, 0.18, 0.07, 0.12, 0.18, 0.10, 0.15]),
    5: np.array([0.02, 0.01, 0.06, 0.03, 0.18, 0.08, 0.14, 0.20, 0.12, 0.16]),
}
STYLES = ["ip", "at", "kwsub", "long", "typo", "puny", "short", "cloud", "newtld", "compromised"]
GENERIC = ["doc", "invoice", "share", "view", "file", "office", "drive", "secure", "portal", "payment", "delivery", "parcel"]
SMALL_TLDS = ["com", "pk", "org", "net", "shop", "app", "online", "store", "io", "co"]


def _rid(n: int = 8) -> str:
    return "".join(random.choices(string.ascii_letters + string.digits, k=n))


def _fill(t: str) -> str:
    return (t.replace("{w}", random.choice(WORDS)).replace("{w2}", random.choice(WORDS))
            .replace("{W}", random.choice(WORDS).title()).replace("{n}", str(random.randint(1, 99999)))
            .replace("{y}", str(random.randint(2015, 2026))).replace("{id}", _rid(11)))


def legit_url(period: int) -> str:
    scheme = "https" if random.random() < 0.85 + 0.02 * period else "http"
    r = random.random()
    if r < 0.07:  # real login / OAuth pages contain "phishy" keywords and long tokens
        return random.choice([
            f"https://accounts.google.com/signin/v2/identifier?continue=https%3A%2F%2Fmail.google.com&service=mail&flowName={_rid(10)}",
            f"https://login.microsoftonline.com/common/oauth2/authorize?client_id={_rid(32)}&response_type=code",
            f"https://www.paypal.com/signin?returnUri={_rid(16)}&state={_rid(12)}",
            f"https://appleid.apple.com/account/manage/security?ref={_rid(8)}",
            f"https://www.amazon.com/ap/signin?openid.return_to={_rid(24)}&openid.assoc_handle=usflex",
            f"https://secure.hbl.com/login/verify?session={_rid(20)}",
        ])
    if r < 0.14:  # brand names in legitimate paths
        brand = random.choice(BRANDS)
        return random.choice([
            f"https://github.com/{brand}/{random.choice(WORDS)}-{random.choice(WORDS)}",
            f"https://techcrunch.com/{random.randint(2019, 2026)}/{brand}-{random.choice(WORDS)}-update",
            f"https://medium.com/@{_rid(6).lower()}/{brand}-{random.choice(WORDS)}-{_rid(12).lower()}",
            f"https://www.reddit.com/r/{brand}/comments/{_rid(6).lower()}/account_{random.choice(WORDS)}",
        ])
    if r < 0.14 + 0.10 + 0.03 * period:  # small businesses / personal sites, incl. new TLDs
        host = f"{random.choice(WORDS)}{random.choice(['', '-'])}{random.choice(WORDS)}.{random.choice(SMALL_TLDS)}"
        return f"{scheme}://{random.choice(['www.', ''])}{host}{_fill(random.choice(LEGIT_PATHS))}"
    if r < 0.28 + 0.03 * period + 0.01 + 0.01 * period:  # legitimate shortened links
        return f"https://{random.choice(SHORTENERS)}/{_rid(random.randint(5, 8))}"
    if random.random() < 0.03 + 0.03 * period:  # SaaS apps on cloud hosting (legit drift)
        host = f"{random.choice(WORDS)}-{random.choice(WORDS)}.{random.choice(LEGIT_CLOUD)}"
    else:
        dom = random.choice(LEGIT_DOMAINS)
        sub = random.choice(["www.", "", "", "m.", "docs.", "news.", "mail."]) if dom.count(".") == 1 else ""
        host = sub + dom
    url = f"{scheme}://{host}{_fill(random.choice(LEGIT_PATHS))}"
    if random.random() < 0.10 + 0.04 * period:
        sep = "&" if "?" in url else "?"
        url += sep + "&".join(_fill(t) for t in random.sample(TRACKING, k=random.randint(1, 3)))
    return url


def _typo(brand: str) -> str:
    b = list(brand)
    i = random.randrange(len(b))
    op = random.random()
    if op < 0.5 and b[i] in HOMOGLYPHS:
        b[i] = random.choice(HOMOGLYPHS[b[i]])
    elif op < 0.75:
        b.insert(i, b[i])  # doubled letter
    else:
        b.pop(i) if len(b) > 4 else b.insert(i, "-")
    return "".join(b).replace("à", "a").replace("ò", "o").replace("è", "e").replace("í", "i")


def phish_url(period: int) -> str:
    style = np.random.choice(STYLES, p=PHISH_MIX[period] / PHISH_MIX[period].sum())
    brand = random.choice(BRANDS)
    word = random.choice(PHISH_WORDS)
    https_p = 0.2 + 0.13 * period
    scheme = "https" if random.random() < https_p else "http"
    path = f"/{word}" + random.choice(["", ".php", ".html", "/index.php", f"/{_rid(6)}"])
    if style == "ip":
        host = ".".join(str(random.randint(1, 254)) for _ in range(4))
        if random.random() < 0.3:
            host += f":{random.choice([8080, 8443, 8000, 3000])}"
        return f"{scheme}://{host}/{brand}{path}"
    if style == "at":
        return f"{scheme}://www.{brand}.com@{_rid(7).lower()}.{random.choice(OLD_TLDS)}{path}"
    if style == "kwsub":
        parts = [brand + ".com", random.choice(PHISH_WORDS), random.choice(PHISH_WORDS) + "-" + _rid(4).lower()]
        return f"{scheme}://{'.'.join(parts)}.{random.choice(OLD_TLDS)}{path}"
    if style == "long":
        q = "&".join(f"{random.choice(['id', 'session', 'token', 'cmd', 'dispatch'])}={_rid(random.randint(16, 40))}"
                     for _ in range(random.randint(2, 5)))
        return f"{scheme}://{_rid(6).lower()}-{brand}.{random.choice(OLD_TLDS)}{path}?{q}"
    if style == "typo":
        tld = random.choice(["com", "net", "co"] + NEW_TLDS[:3])
        extra = random.choice(["", "-" + word, "-support", "-help"])
        return f"{scheme}://{_typo(brand)}{extra}.{tld}{path}"
    if style == "puny":
        return f"{scheme}://xn--{brand[:4]}-{_rid(4).lower()}.com{path}"
    if style == "short":
        return f"https://{random.choice(SHORTENERS)}/{_rid(random.randint(5, 8))}"
    if style == "cloud":
        host = random.choice(ABUSED_HOSTS)
        if "/" in host:
            return f"https://{host}/{brand}-{word}-{_rid(5).lower()}"
        return f"https://{brand}-{word}-{_rid(4).lower()}.{host}{path}"
    if style == "compromised":  # hacked small-business site, often no brand at all
        site = f"{random.choice(WORDS)}{random.choice(WORDS)}.{random.choice(SMALL_TLDS)}"
        kit = random.choice(GENERIC) + random.choice(["", "-" + random.choice(GENERIC)])
        if random.random() < 0.3:
            kit = brand + "-" + kit
        return f"{scheme}://{random.choice(['www.', ''])}{site}/{random.choice(['wp-content/uploads', 'wp-includes', 'images', 'files'])}/{_rid(5).lower()}/{kit}{random.choice(['.php', '/', '.html', ''])}"
    # newtld
    return f"{scheme}://{brand}{random.choice(['', '-', '.'])}{word}.{random.choice(NEW_TLDS)}{path}"


def generate(n: int, n_periods: int, seed: int, phish_ratio: float = 0.45, label_noise: float = 0.015) -> pd.DataFrame:
    set_seed(seed)
    start = datetime(2025, 1, 1)
    rows = []
    per = n // n_periods
    for p in range(n_periods):
        for _ in range(per):
            is_phish = random.random() < phish_ratio
            url = phish_url(p) if is_phish else legit_url(p)
            ts = start + timedelta(days=60 * p + random.uniform(0, 60))
            label = int(is_phish) if random.random() > label_noise else int(not is_phish)
            rows.append({"url": url, "label": label, "timestamp": ts.isoformat(timespec="seconds"),
                         "period": p, "source": "synthetic"})
    df = pd.DataFrame(rows).drop_duplicates("url").sort_values("timestamp").reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(RAW_DIR / "urls.csv"))
    args = ap.parse_args()
    params = load_params()
    ensure_dirs()
    d = params["data"]
    df = generate(d["sample_size"], d["n_periods"], params["seed"])
    df.to_csv(args.out, index=False)
    log.info("Wrote %d URLs (%.1f%% phishing) -> %s", len(df), 100 * df.label.mean(), args.out)


if __name__ == "__main__":
    main()
