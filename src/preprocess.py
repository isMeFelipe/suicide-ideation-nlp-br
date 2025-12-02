import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import os

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def load_datasets():
    """Carrega o dataset mesclado."""
    merged_path = "data/merged_dataset.csv"
    if os.path.exists(merged_path):
        df = pd.read_csv(merged_path)
        return df
    else:
        import sys
        sys.path.append(os.path.dirname(__file__))
        from merge_datasets import merge_datasets
        print("Dataset mesclado não encontrado. Criando dataset mesclado...")
        df = merge_datasets()
        return df

def clean_text(text):
    """Limpeza de texto: lowercase, remoção de URLs, mentions, caracteres especiais e stopwords."""
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def preprocess(df):
    """Aplica limpeza de texto no dataframe."""
    df = df.dropna(subset=['text']).copy()
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'].str.len() > 0].copy()
    return df