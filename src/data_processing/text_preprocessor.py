
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def clean_text(text):
    """Clean article text before NLP analysis."""
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """Tokenize, remove stop words, and lemmatize text."""
    text = clean_text(text)

    if not text:
        return ""

    tokens = word_tokenize(text)

    if remove_stopwords:
        tokens = [token for token in tokens if token not in stop_words]

    if lemmatize:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]

    tokens = [token for token in tokens if len(token) > 2]

    return " ".join(tokens)
