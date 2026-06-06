# import contractions
# import re  # re = regular expressions - To find patterns in text
# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer

# text = "BREAKING!! Trump doesn't know what he's doing! Visit http://fake.com"

# #Contractions expand karo
# fixed = contractions.fix(text)
# # print(fixed)
# # Output: "BREAKING!! Trump does not know what he is doing! Visit http://fake.com"

# # Converted to lowercase 
# lowered = fixed.lower()
# # print(lowered)
# # Output: "breaking!! trump does not know what he is doing! visit http://fake.com"

# # Removed URLs
# no_urls = re.sub(r"https?://\S+|www\.\S+", "", lowered)
# # print(no_urls)
# # Output: "breaking!! trump does not know what he is doing! visit "

# # Step 4 - sirf letters aur spaces rakho, baaki sab hatao
# clean = re.sub(r"[^a-z\s]", "", no_urls)
# # print(clean)
# # Output: "breaking trump does not know what he is doing visit "

# # Step 5 - Removed stopwords 
# nltk.download('stopwords', quiet=True)
# stop_words = set(stopwords.words("english"))
# tokens = clean.split()  # breaking sentance into words
# filtered = [word for word in tokens if word not in stop_words]
# # print(" ".join(filtered))
# # Output: "breaking trump know visit"

# nltk.download('wordnet', quiet=True)  # WordNet — lemmatization dictionary
# lemmatizer = WordNetLemmatizer()
# lemmatized = [lemmatizer.lemmatize(word) for word in filtered]
# print(" ".join(lemmatized))
# # Output: "breaking trump know visit"





import re
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


class TextPreprocessor:
    """Full text cleaning pipeline for fake news detection."""

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        """Run all cleaning steps and return cleaned text."""
        if not isinstance(text, str) or not text.strip():# guard against non-string or empty inputs
            return ""
        text = contractions.fix(text)
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", "", text)  # remove URLs
        text = re.sub(r"[^a-z\s]", " ", text)              # replace special chars with space
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]  # remove stopwords and short tokens
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]             # lemmatize
        return " ".join(tokens)

    def clean_batch(self, texts: list) -> list:
        """Clean a list of texts - use this on DataFrame column."""
        return [self.clean(t) for t in texts]  
    
    def get_meta_features(self, text: str) -> dict:
        if not isinstance(text, str):# guard against non-string inputs
            text = ""

        words = text.split()
        sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))

        return {
            "char_count": len(text),
            "word_count": len(words),
            "sentence_count": sentence_count,
            "exclamation_count": text.count('!'),
            "question_count": text.count('?'),
            "cap_word_count": sum(1 for w in words if w.isupper() and len(w) > 1),
            "avg_word_len": sum(len(w) for w in words) / max(1, len(words)),
            "avg_sentence_len": len(words) / sentence_count,
            "quote_count": text.count('"') + text.count("'"),
            "digit_ratio": sum(c.isdigit() for c in text) / max(1, len(text))
        }


# Quick test
if __name__ == "__main__":
    import pandas as pd
    import os
    from pathlib import Path

    # Resolve paths relative to this file — works from any directory
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data"

    pp = TextPreprocessor()

    # ── 1. Load ISOT dataset (True.csv + Fake.csv) ──────────────────────────
    print("=" * 50)
    print("[1/5] Loading ISOT dataset...")
    true_df = pd.read_csv(DATA_DIR / "True.csv")
    fake_df = pd.read_csv(DATA_DIR / "Fake.csv")

    true_df['label'] = 0   # Real
    fake_df['label'] = 1   # Fake
    true_df['source_dataset'] = 'ISOT'
    fake_df['source_dataset'] = 'ISOT'

    isot_df = pd.concat([true_df, fake_df], ignore_index=True)
    # ISOT has 'text' column already
    print(f"  ISOT: {len(true_df)} Real + {len(fake_df)} Fake = {len(isot_df)} total")

    # ── 2. Load LIAR dataset (train.tsv + test.tsv + valid.tsv) ─────────────
    print("[2/5] Loading LIAR dataset...")

    # LIAR TSV columns (no header):
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

    # Map LIAR 6-class labels → binary
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

    # Drop rows with unmapped labels (if any)
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

    # Keep only common columns
    isot_df = isot_df[["text", "label", "source_dataset"]]
    liar_df = liar_df[["text", "label", "source_dataset"]]

    df = pd.concat([isot_df, liar_df], ignore_index=True)

    # Shuffle — mix real/fake and ISOT/LIAR randomly
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Remove null/empty text
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""]

    total_real = (df["label"] == 0).sum()
    total_fake = (df["label"] == 1).sum()
    print(f"  Combined: {total_real} Real + {total_fake} Fake = {len(df)} total")

    # ── 4. Clean text + meta features ───────────────────────────────────────
    print("[4/5] Cleaning text...")
    df['clean_text'] = pp.clean_batch(df['text'])

    # Remove rows where cleaned text became empty
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df = df[df["clean_text"].str.strip().ne("")]
    df = df.reset_index(drop=True)

    print(f"  Rows after removing empty clean_text: {len(df)}")

    print("  Generating meta features...")
    meta_df = pd.DataFrame(
        [pp.get_meta_features(text) for text in df['text']]
    ).reset_index(drop=True)

    df = pd.concat([df, meta_df], axis=1)

    print(
        "  Empty clean_text before save:",
        (df["clean_text"].fillna("").str.strip() == "").sum()
    )

    # ── 5. Save ─────────────────────────────────────────────────────────────
    print("[5/5] Saving...")
    out_dir = DATA_DIR / "processed"
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(out_dir / "cleaned_dataset.csv", index=False)

    print(f"\nDone! Saved to {out_dir / 'cleaned_dataset.csv'}")
    print(f"Final shape: {df.shape}")
    print(f"\nDataset breakdown:")
    print(df.groupby(["source_dataset", "label"]).size().unstack(fill_value=0))
    print(f"\nSample rows:")
    print(df[['text', 'clean_text', 'label', 'source_dataset']].head(3))