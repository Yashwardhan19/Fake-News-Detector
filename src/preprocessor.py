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
        if not isinstance(text, str) or not text.strip():
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
        """Clean a list of texts — use this on DataFrame column."""
        return [self.clean(t) for t in texts]  
    
    def get_meta_features(self, text: str) -> dict:
        if not isinstance(text, str):
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


# Quck test
if __name__ == "__main__":
    import pandas as pd
    import os

    pp = TextPreprocessor()

    # load data
    true_df = pd.read_csv("data/True.csv")
    fake_df = pd.read_csv("data/Fake.csv")

    true_df['label'] = 0
    fake_df['label'] = 1

    df = pd.concat([true_df, fake_df], ignore_index=True)

    # shuffle dataset — mix real and fake rows randomly
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Remove rows where text is missing
    df = df.dropna(subset=["text"])

    # Remove rows with empty text
    df = df[df["text"].astype(str).str.strip() != ""]

    print(f"Rows after removing null/empty text: {len(df)}")

    print(f"Total rows: {len(df)}")

    # clean text column and save to new column 'clean_text'
    print("Cleaning text...")

    df['clean_text'] = pp.clean_batch(df['text'])

    # Remove rows where cleaned text became empty
    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df = df[df["clean_text"].str.strip().ne("")]

    # Reset index after filtering
    df = df.reset_index(drop=True)

    print(f"Rows after removing empty clean_text: {len(df)}")

    print("Generating meta features...")

    meta_df = pd.DataFrame(
        [pp.get_meta_features(text) for text in df['text']]
    ).reset_index(drop=True)

    df = pd.concat([df, meta_df], axis=1)

    print(
        "Empty clean_text before save:",
        (df["clean_text"].fillna("").str.strip() == "").sum()
    )

    # save cleaned dataset for future use
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/cleaned_dataset.csv", index=False)

    print("Done! Saved to data/processed/cleaned_dataset.csv")
    print(df[['text', 'clean_text', 'label']].head(3))