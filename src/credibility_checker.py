"""
Credibility checker for the Fake News Detector.

Provides a two-layer verification system:
  1. **Google Fact Check API** - queries Google's ClaimReview database to see
     if the claim (or similar claims) have already been reviewed by trusted
     fact-checking organizations.
  2. **Source domain trust score** - looks up the article's domain against a
     curated whitelist/blacklist defined in config.yaml and returns a trust
     label (Trusted / Unreliable / Unknown).

API key is loaded from the project-root .env file (GOOGLE_FACT_CHECK_API_KEY).
"""

import os
import requests
import yaml
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load .env from project root so GOOGLE_FACT_CHECK_API_KEY is available
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# ── Config ─────────────────────────────────────────────────
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

CRED_CFG = CONFIG["credibility"]


class CredibilityChecker:
    """
    Two-layer credibility check:
      1. Google Fact Check API -- has this claim been debunked?
      2. Source domain trust score -- is the source reliable?
    """

    def __init__(self):
        """
        Initialize the checker with API key and trust lists from config.

        The API key comes from the .env file (not config.yaml) to avoid
        committing secrets to version control.
        """
        # API key from .env file -- NOT from config.yaml
        self.api_key     = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "")
        self.max_results = CRED_CFG["max_results"]
        self.trusted     = CRED_CFG["trusted_sources"]
        self.untrusted   = CRED_CFG["untrusted_sources"]

    def check_claim(self, text: str) -> dict:
        """
        Query Google Fact Check API for matching fact-checks.

        Args:
            text: The claim or article text to search for. Only the first
                  200 characters are sent to keep the query focused.

        Returns:
            dict with keys:
              - found (bool): Whether any fact-checks matched.
              - results (list): Up to 3 matching reviews, each with
                claim, rating, source, and url.
              - error (str, optional): Present only if the API call failed.
        """
        # Truncate to 200 chars -- long queries return worse results from the API
        query = text[:200].strip()

        url = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
        params = {
            "query":        query,
            "key":          self.api_key,
            "pageSize":     self.max_results,
            "languageCode": "en"
        }

        try:
            response = requests.get(url, params=params, timeout=5)
            data     = response.json()

            if "claims" not in data:
                return {"found": False, "results": []}

            # Each claim can have multiple reviews from different fact-checkers
            results = []
            for claim in data["claims"]:
                for review in claim.get("claimReview", []):
                    results.append({
                        "claim":  claim.get("text", ""),
                        "rating": review.get("textualRating", "Unknown"),
                        "source": review.get("publisher", {}).get("name", "Unknown"),
                        "url":    review.get("url", "")
                    })

            # Cap at 3 results to keep the UI concise
            return {"found": len(results) > 0, "results": results[:3]}

        except Exception as e:
            return {"found": False, "results": [], "error": str(e)}

    def check_source(self, url: str) -> dict:
        """
        Check domain trust score against the curated lists in config.yaml.

        Args:
            url: Full URL of the article source (e.g. https://reuters.com/...).

        Returns:
            dict with keys:
              - score (float or None): Trust score if domain is in a list.
              - label (str): Human-readable trust label with emoji.
              - domain (str or None): Extracted domain name.
        """
        if not url or not url.startswith("http"):
            return {"score": None, "label": "Unknown", "domain": None}

        # Strip "www." so "www.reuters.com" and "reuters.com" match the same entry
        domain = urlparse(url).netloc.replace("www.", "")

        if domain in self.trusted:
            return {"score": self.trusted[domain], "label": "✅ Trusted Source", "domain": domain}
        elif domain in self.untrusted:
            return {"score": self.untrusted[domain], "label": "🚨 Known Unreliable Source", "domain": domain}
        else:
            return {"score": None, "label": "⚠️ Unknown Source", "domain": domain}

    def full_check(self, text: str, source_url: str = None) -> dict:
        """
        Combined check -- fact check + source credibility.

        Args:
            text:       Article or claim text to verify.
            source_url: Optional URL of the article source.

        Returns:
            dict with keys: fact_check (from check_claim) and
            source_check (from check_source, or None if no URL given).
        """
        fact_check   = self.check_claim(text)
        source_check = self.check_source(source_url) if source_url else None

        return {
            "fact_check":   fact_check,
            "source_check": source_check
        }


# ── Quick test ─────────────────────────────────────────────
if __name__ == "__main__":
    checker = CredibilityChecker()

    result = checker.full_check(
    text="Biden did not say he would ban fracking",
    source_url="https://reuters.com/article/123"
)

    print("=== Fact Check ===")
    if result["fact_check"]["found"]:
        for r in result["fact_check"]["results"]:
            print(f"Claim:  {r['claim'][:80]}")
            print(f"Rating: {r['rating']}")
            print(f"Source: {r['source']}")
            print()
    else:
        print("No fact-checks found.")

    print("=== Source Check ===")
    print(result["source_check"])