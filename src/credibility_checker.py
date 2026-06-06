import os
import requests
import yaml
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

CRED_CFG = CONFIG["credibility"]


class CredibilityChecker:
    """
    Two-layer credibility check:
      1. Google Fact Check API — has this claim been debunked?
      2. Source domain trust score — is the source reliable?
    """

    def __init__(self):
        # API key from .env file — NOT from config.yaml
        self.api_key     = os.getenv("GOOGLE_FACT_CHECK_API_KEY", "")
        self.max_results = CRED_CFG["max_results"]
        self.trusted     = CRED_CFG["trusted_sources"]
        self.untrusted   = CRED_CFG["untrusted_sources"]

    def check_claim(self, text: str) -> dict:
        """Query Google Fact Check API — returns matching fact-checks if found."""
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

            results = []
            for claim in data["claims"]:
                for review in claim.get("claimReview", []):
                    results.append({
                        "claim":  claim.get("text", ""),
                        "rating": review.get("textualRating", "Unknown"),
                        "source": review.get("publisher", {}).get("name", "Unknown"),
                        "url":    review.get("url", "")
                    })

            return {"found": len(results) > 0, "results": results[:3]}

        except Exception as e:
            return {"found": False, "results": [], "error": str(e)}

    def check_source(self, url: str) -> dict:
        """Check domain trust score from config."""
        if not url or not url.startswith("http"):
            return {"score": None, "label": "Unknown", "domain": None}

        domain = urlparse(url).netloc.replace("www.", "")

        if domain in self.trusted:
            return {"score": self.trusted[domain], "label": "✅ Trusted Source", "domain": domain}
        elif domain in self.untrusted:
            return {"score": self.untrusted[domain], "label": "🚨 Known Unreliable Source", "domain": domain}
        else:
            return {"score": None, "label": "⚠️ Unknown Source", "domain": domain}

    def full_check(self, text: str, source_url: str = None) -> dict:
        """Combined check — fact check + source credibility."""
        fact_check   = self.check_claim(text)
        source_check = self.check_source(source_url) if source_url else None

        return {
            "fact_check":   fact_check,
            "source_check": source_check
        }


# Quick test
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