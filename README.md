# 📰 Fake News Detector

A multi-layer fake news detection system that combines **Machine Learning**, **SHAP Explainability**, and **Google Fact Check API** to identify and explain fake news articles.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![ML](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![UI](https://img.shields.io/badge/UI-Streamlit-red)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)

---

## 🎯 Features

- **3 ML Models** — Logistic Regression, Random Forest, and Soft Voting Ensemble
- **SHAP Explainability** — See exactly which words pushed toward "FAKE" or "REAL"
- **Credibility Engine** — Google Fact Check API + source domain trust scores
- **Interactive UI** — Streamlit web app with real-time analysis
- **Dual Dataset Training** — ISOT (44K articles) + LIAR (12K statements) = 57K total

## 🏗️ Architecture

```
User Input (news article + optional URL)
        │
        ├── ML Engine
        │     ├── Text Preprocessing (clean, lemmatize, meta features)
        │     ├── TF-IDF Vectorization (50,000 features)
        │     └── 3-Model Prediction (LR, RF, Ensemble)
        │
        ├── SHAP Explainer
        │     └── Word-level importance (red = fake signal, green = real)
        │
        └── Credibility Engine
              ├── Google Fact Check API (has this been debunked?)
              └── Source Trust Score (is this domain reliable?)
```

## 📁 Project Structure

```
Fake News Detector/
├── app/
│   └── app.py                  # Streamlit web application
├── src/
│   ├── preprocessor.py         # Text cleaning + meta feature extraction
│   ├── feature_extractor.py    # TF-IDF + SBERT (optional) + meta scaling
│   ├── model_trainer.py        # Train LR, RF, Ensemble + save .pkl
│   ├── explainer.py            # SHAP-based model interpretability
│   └── credibility_checker.py  # Google Fact Check API + source trust
├── data/
│   ├── True.csv                # ISOT real news (Reuters)
│   ├── Fake.csv                # ISOT fake news
│   ├── train.tsv               # LIAR dataset (train split)
│   ├── test.tsv                # LIAR dataset (test split)
│   └── valid.tsv               # LIAR dataset (validation split)
├── models/
│   ├── lr_model.pkl            # Trained Logistic Regression
│   ├── rf_model.pkl            # Trained Random Forest
│   ├── ensemble_model.pkl      # Trained Ensemble (LR + RF)
│   ├── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
│   ├── meta_scaler.pkl         # Fitted StandardScaler for meta features
│   └── shap_explainer.pkl      # Pre-computed SHAP background
├── config.yaml                 # Centralized configuration
├── .env                        # API keys (gitignored)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Yashwardhan19/Fake-News-Detector.git
cd Fake-News-Detector
pip install -r requirements.txt
```

### 2. Download NLTK Data

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### 3. Set Up API Key (Optional)

Create a `.env` file in the project root:

```
GOOGLE_FACT_CHECK_API_KEY=your_api_key_here
```

Get a free API key from [Google Cloud Console](https://console.cloud.google.com/) → Enable "Fact Check Tools API" → Create API Key.

### 4. Run the Pipeline

```bash
# Step 1: Preprocess data (merge ISOT + LIAR, clean text)
python src/preprocessor.py

# Step 2: Extract features (TF-IDF + meta)
python src/feature_extractor.py

# Step 3: Train models (LR, RF, Ensemble)
python src/model_trainer.py

# Step 4: Build SHAP explainer
python src/explainer.py
```

### 5. Launch the App

```bash
streamlit run app/app.py
```

## 📊 Model Performance

Evaluated on 20% held-out test set (stratified split, trained on ISOT + LIAR combined):

| Model | F1 (Fake) | Precision | Recall | ROC-AUC | Inference Time |
|-------|-----------|-----------|--------|---------|----------------|
| Logistic Regression | 0.8908 | 0.8908 | 0.8908 | 0.9709 | 0.02s |
| **Random Forest** | **0.9089** | **0.9206** | **0.8974** | **0.9782** | 2.11s |
| Ensemble (LR + RF) | 0.9074 | 0.9144 | 0.9004 | 0.9786 | 2.16s |

> **Why did accuracy drop from 99% to ~90%?** The previous 99% was trained on ISOT-only data (easy — formal Reuters vs. informal fake sources). After adding the LIAR dataset (12K real-world political statements), the model now faces a harder, more realistic classification task. **~90% F1 on mixed data is a much more honest and credible result for your portfolio.**

## 🔍 How It Works

### Text Preprocessing
1. Expand contractions ("don't" → "do not")
2. Lowercase everything
3. Remove URLs and special characters
4. Remove stopwords and short tokens (< 3 chars)
5. Lemmatize (running → run, better → good)
6. Extract 10 meta features (word count, caps ratio, etc.)

### Feature Extraction
- **TF-IDF**: 50,000 word/bigram features (sparse matrix for memory efficiency)
- **Meta Features**: 10 hand-crafted features (char_count, exclamation_count, etc.)
- **Total**: 50,010 features per article

### Model Training
- **Logistic Regression**: Fast, interpretable, works well with sparse TF-IDF
- **Random Forest**: 200 trees, captures non-linear patterns
- **Ensemble**: Soft voting with RF weighted 2x over LR

### SHAP Explainability
- Uses `LinearExplainer` on the LR model for speed
- Highlights words that pushed toward FAKE (red) or REAL (green)
- Only shows words actually present in the article (no noise)

### Credibility Engine
- **Fact Check API**: Searches Google's database of fact-checked claims
- **Source Trust**: Checks domain against curated list of trusted/untrusted sources

## 📦 Datasets

| Dataset | Rows | Type | Source |
|---------|------|------|--------|
| ISOT | 44,898 | Full articles | [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) |
| LIAR | 12,836 | Short statements | [UCSB](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) |

### LIAR Label Mapping (6 → Binary)
- **Real (0)**: true, mostly-true, half-true
- **Fake (1)**: barely-true, false, pants-fire

## ⚠️ Limitations

- Trained on English text only — no Hindi/multilingual support
- ISOT dataset is mostly US political news (2015-2018)
- Model detects **writing style** more than **factual accuracy**
- High accuracy on test set may not transfer to real-world news
- Short text (tweets, headlines) may not work as well as full articles

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| ML Models | Scikit-learn |
| Text Processing | NLTK, TF-IDF |
| Explainability | SHAP |
| Web UI | Streamlit |
| Visualization | Plotly, Matplotlib, Seaborn |
| Configuration | YAML |
| API | Google Fact Check Tools |

## 📄 License

This project is for educational purposes. Datasets are used under their respective licenses.

---

*Built as a portfolio project demonstrating ML pipeline design, model explainability, and web deployment.*
