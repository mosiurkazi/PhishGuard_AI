import re
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    "login", "verify", "account", "bank", "update", "secure",
    "signin", "confirm", "password", "urgent", "free", "win"
]


def has_ip_address(url: str) -> int:
    pattern = r"(\d{1,3}\.){3}\d{1,3}"
    return 1 if re.search(pattern, url) else 0


def count_suspicious_words(url: str) -> int:
    url_lower = url.lower()
    return sum(word in url_lower for word in SUSPICIOUS_WORDS)


def extract_url_features(url: str) -> dict:
    parsed = urlparse(url)
    domain = parsed.netloc
    path = parsed.path

    features = {
        "url_length": len(url),
        "domain_length": len(domain),
        "path_length": len(path),
        "dot_count": url.count('.'),
        "hyphen_count": url.count('-'),
        "slash_count": url.count('/'),
        "question_count": url.count('?'),
        "equal_count": url.count('='),
        "at_count": url.count('@'),
        "digit_count": sum(c.isdigit() for c in url),
        "https": 1 if parsed.scheme == "https" else 0,
        "has_ip": has_ip_address(url),
        "suspicious_word_count": count_suspicious_words(url),
        "subdomain_count": max(0, len(domain.split('.')) - 2),
    }
    return features