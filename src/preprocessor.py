"""Text preprocessing pipeline for fake news detection.

Provides the TextPreprocessor class which handles:
  - Contraction expansion, lowercasing, URL/special-char removal
  - Stopword removal, short-token filtering, and lemmatization
  - Extraction of stylistic meta-features (exclamation counts, caps ratio, etc.)

When run as __main__, it loads the ISOT and LIAR datasets, merges them into
a single binary-labelled corpus, cleans every article, attaches meta-features,
and writes the result to data/processed/cleaned_dataset.csv.
"""

import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data silently on first import
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """Full text cleaning pipeline for fake news detection.

    Encapsulates all NLP preprocessing steps so the same pipeline can be
    reused during training, evaluation, and inference without duplicating
    logic.  Heavy resources (stopword set, lemmatizer) are loaded once in
    __init__ and reused across calls.
    """

    def __init__(self):
        # Using a set for O(1) stopword lookups instead of the default list
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        """Run every cleaning step on a single text string.

        Args:
            text: Raw article body or statement string.

        Returns:
            A lowercased, lemmatized string with stopwords, URLs, and
            special characters removed.  Returns an empty string for
            non-string or blank inputs.
        """
        if not isinstance(text, str) or not text.strip():  # guard against non-string or empty inputs
            return ""

        # Expand contractions first so "don't" becomes "do not" before
        # lowercasing, preventing artefacts like "don t"
        text = contractions.fix(text)
        text = text.lower()

        text = re.sub(r"https?://\S+|www\.\S+", "", text)  # URLs carry no semantic signal for fake-news detection
        text = re.sub(r"[^a-z\s]", " ", text)              # strip digits/punctuation; replace with space to avoid fusing words

        tokens = text.split()
        # Drop stopwords AND tokens <= 2 chars (e.g. "us", "an") which add
        # noise without meaningful discriminatory power
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        # Lemmatize to collapse inflected forms ("running" -> "run")
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)

    def clean_batch(self, texts: list) -> list:
        """Apply clean() to every element in a list.

        Convenience wrapper for pandas columns: pass df['text'].tolist()
        or directly the Series.

        Args:
            texts: Iterable of raw text strings.

        Returns:
            List of cleaned strings, same length as *texts*.
        """
        return [self.clean(t) for t in texts]
    
    def get_meta_features(self, text: str) -> dict:
        """Extract stylistic / structural features from the *raw* text.

        These features capture writing-style cues (e.g. excessive
        exclamation marks, ALL-CAPS words, quote density) that often
        differ between genuine and fabricated news.

        Args:
            text: The original (uncleaned) article text so punctuation
                  and casing information are still available.

        Returns:
            Dict mapping feature names to their numeric values.
        """
        if not isinstance(text, str):  # guard against non-string inputs
            text = ""

        words = text.split()
        # Use max(1, ...) to avoid division-by-zero when no sentence-ending
        # punctuation is present (e.g. headlines)
        sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))

        return {
            "char_count": len(text),
            "word_count": len(words),
            "sentence_count": sentence_count,
            "exclamation_count": text.count('!'),        # sensationalism signal
            "question_count": text.count('?'),            # rhetorical-question signal
            "cap_word_count": sum(1 for w in words if w.isupper() and len(w) > 1),  # ALL-CAPS words (len>1 skips "I", "A")
            "avg_word_len": sum(len(w) for w in words) / max(1, len(words)),
            "avg_sentence_len": len(words) / sentence_count,
            "quote_count": text.count('"') + text.count("'"),  # attribution / hearsay indicator
            "digit_ratio": sum(c.isdigit() for c in text) / max(1, len(text))  # numeric-density of text
        }


# ---------------------------------------------------------------------------
# CLI entry point: load raw data, clean, extract features, and save
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pandas as pd
    import os
    from pathlib import Path

    # Resolve paths relative to this file so the script works regardless of
    # the directory it is invoked from (e.g. `python src/preprocessor.py`)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"

    pp = TextPreprocessor()

    # ── 1. Load ISOT dataset (True.csv + Fake.csv) ──────────────────────────
    # ISOT ships as two separate CSVs, one per class.  We assign binary
    # labels here and tag the source so we can audit per-dataset performance.
    print("=" * 50)
    print("[1/5] Loading ISOT dataset...")
    true_df = pd.read_csv(DATA_DIR / "True.csv")
    fake_df = pd.read_csv(DATA_DIR / "Fake.csv")

    true_df['label'] = 0   # 0 = Real
    fake_df['label'] = 1   # 1 = Fake
    true_df['source_dataset'] = 'ISOT'
    fake_df['source_dataset'] = 'ISOT'

    isot_df = pd.concat([true_df, fake_df], ignore_index=True)
    # ISOT already contains a 'text' column, no renaming needed
    print(f"  ISOT: {len(true_df)} Real + {len(fake_df)} Fake = {len(isot_df)} total")

    # ── 2. Load LIAR dataset (train.tsv + test.tsv + valid.tsv) ─────────────
    # LIAR is distributed as headerless TSV files split into train/test/valid.
    # We load all three splits and concatenate them because we will later
    # apply our own train/test split.
    print("[2/5] Loading LIAR dataset...")

    # Column order defined by the LIAR README:
    # 0:ID, 1:label, 2:statement, 3:subject, 4:speaker, 5:job, 6:state,
    # 7:party, 8-12:credit_counts, 13:context
    LIAR_COLS = [
        "id", "label_raw", "text", "subject", "speaker", "job",
        "state", "party", "barely_true", "false_count", "half_true",
        "mostly_true", "pants_fire", "context"
    ]

    liar_parts = []
    for tsv_file in ["train.tsv", "test.tsv", "valid.tsv"]:
        tsv_path = DATA_DIR / tsv_file
        if tsv_path.exists():
            part = pd.read_csv(tsv_path, sep="\t", header=None, names=LIAR_COLS)
            liar_parts.append(part)
            print(f"  {tsv_file}: {len(part)} rows")

    liar_df = pd.concat(liar_parts, ignore_index=True)

    # Map LIAR 6-class labels -> binary
    # Real (0): true, mostly-true, half-true
    # Fake (1): barely-true, false, pants-fire
    LABEL_MAP = {
        "true":        0,
        "mostly-true": 0,
        "half-true":   0,
        "barely-true": 1,
        "false":       1,
        "pants-fire":  1,
    }
    liar_df["label"] = liar_df["label_raw"].map(LABEL_MAP)
    liar_df["source_dataset"] = "LIAR"

    # Drop rows whose original label did not match the LABEL_MAP keys;
    # these appear as NaN after .map() and would break integer casting.
    before = len(liar_df)
    liar_df = liar_df.dropna(subset=["label"])
    liar_df["label"] = liar_df["label"].astype(int)
    if len(liar_df) < before:
        print(f"  Dropped {before - len(liar_df)} rows with unknown labels")

    liar_real = (liar_df["label"] == 0).sum()
    liar_fake = (liar_df["label"] == 1).sum()
    print(f"  LIAR: {liar_real} Real + {liar_fake} Fake = {len(liar_df)} total")

    # ── 3. Merge both datasets ──────────────────────────────────────────────
    print("[3/5] Merging ISOT + LIAR...")

    # Retain only the three columns shared by both datasets; extra LIAR
    # metadata (speaker, party, etc.) is not used downstream.
    isot_df = isot_df[["text", "label", "source_dataset"]]
    liar_df = liar_df[["text", "label", "source_dataset"]]

    df = pd.concat([isot_df, liar_df], ignore_index=True)

    # Shuffle to interleave ISOT and LIAR rows so that any sequential
    # train/test split does not accidentally separate the two sources.
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Remove null/empty text rows that would produce meaningless embeddings
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""]

    total_real = (df["label"] == 0).sum()
    total_fake = (df["label"] == 1).sum()
    print(f"  Combined: {total_real} Real + {total_fake} Fake = {len(df)} total")

    # ── 4. Clean text + extract meta features ────────────────────────────────
    print("[4/5] Cleaning text...")
    df['clean_text'] = pp.clean_batch(df['text'])

    # Some articles reduce to an empty string after aggressive cleaning
    # (e.g. very short tweets or non-English text).  Remove them so the
    # model never trains on zero-length inputs.
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df = df[df["clean_text"].str.strip().ne("")]
    df = df.reset_index(drop=True)

    print(f"  Rows after removing empty clean_text: {len(df)}")

    # Meta features are computed on the ORIGINAL text (not clean_text)
    # because punctuation, capitalisation, and digit info has already been
    # stripped from clean_text.
    print("  Generating meta features...")
    meta_df = pd.DataFrame(
        [pp.get_meta_features(text) for text in df['text']]
    ).reset_index(drop=True)

    # Horizontally merge meta features into the main DataFrame
    df = pd.concat([df, meta_df], axis=1)

    # Sanity check: report how many empty rows remain (should be 0)
    print(
        "  Empty clean_text before save:",
        (df["clean_text"].fillna("").str.strip() == "").sum()
    )

    # ── 5. Save final dataset ────────────────────────────────────────────────
    print("[5/5] Saving...")
    out_dir = DATA_DIR / "processed"
    os.makedirs(out_dir, exist_ok=True)  # create output dir if it doesn't exist
    df.to_csv(out_dir / "cleaned_dataset.csv", index=False)

    print(f"\nDone! Saved to {out_dir / 'cleaned_dataset.csv'}")
    print(f"Final shape: {df.shape}")
    print(f"\nDataset breakdown:")
    print(df.groupby(["source_dataset", "label"]).size().unstack(fill_value=0))
    print(f"\nSample rows:")
    print(df[['text', 'clean_text', 'label', 'source_dataset']].head(3))