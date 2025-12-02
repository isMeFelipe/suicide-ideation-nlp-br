from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import sys
import os

sys.path.append(os.path.dirname(__file__))
from preprocess import load_datasets, preprocess
from evaluate import evaluate_model
from config import (
    RANDOM_STATE, TEST_SIZE,
    TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF, TFIDF_NGRAM_RANGE,
    LOGISTIC_REGRESSION_MAX_ITER, LOGISTIC_REGRESSION_C,
    MODEL_FILES
)

# 1. Carregar e pré-processar dados
df = preprocess(load_datasets())
X = df['clean_text']
y = df['label']

# 2. Separar treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

# 3. TF-IDF
vectorizer = TfidfVectorizer(
    max_features=TFIDF_MAX_FEATURES,
    min_df=TFIDF_MIN_DF,
    max_df=TFIDF_MAX_DF,
    ngram_range=TFIDF_NGRAM_RANGE
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 4. Treinar modelo
model = LogisticRegression(
    max_iter=LOGISTIC_REGRESSION_MAX_ITER,
    C=LOGISTIC_REGRESSION_C,
    random_state=RANDOM_STATE
)
model.fit(X_train_tfidf, y_train)

# 5. Avaliar
y_pred = model.predict(X_test_tfidf)
y_proba = model.predict_proba(X_test_tfidf)

evaluate_model(y_test, y_pred, y_proba, "Logistic Regression")

# 6. Salvar modelo e vetor
with open(MODEL_FILES['vectorizer'], "wb") as f:
    pickle.dump(vectorizer, f)

with open(MODEL_FILES['logistic'], "wb") as f:
    pickle.dump(model, f)

print(f"\nModelo e vetor salvos em {MODEL_FILES['vectorizer'].parent}!")
