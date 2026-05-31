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


#Quck test
if __name__ == "__main__":
    import pandas as pd

    pp = TextPreprocessor()
    df = pd.read_csv("data/True.csv")

    sample = df['text'].head()
    for text in sample:
        print("Original:", text[:80])
        print("Cleaned :", pp.clean(text[:80]))
        print("---")

