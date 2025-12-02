import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

TFIDF_MAX_FEATURES = 5000
TFIDF_MIN_DF = 2
TFIDF_MAX_DF = 0.95
TFIDF_NGRAM_RANGE = (1, 2)

LOGISTIC_REGRESSION_MAX_ITER = 1000
LOGISTIC_REGRESSION_C = 1.0

CV_FOLDS = 5

MODEL_FILES = {
    'vectorizer': MODELS_DIR / "tfidf_vectorizer.pkl",
    'logistic': MODELS_DIR / "logistic_model.pkl",
    'svm': MODELS_DIR / "svm_model.pkl",
    'random_forest': MODELS_DIR / "random_forest_model.pkl"
}

def get_model_path(dataset_name, language, model_type='logistic_regression'):
    return MODELS_DIR / f"{dataset_name}_{language}" / f"{model_type}.pkl"

def get_vectorizer_path(dataset_name, language):
    return MODELS_DIR / f"{dataset_name}_{language}" / "vectorizer.pkl"

DATASET_PATHS = {
    'merged': DATA_DIR / "merged_dataset.csv",
    'reddit': DATA_DIR / "suicidal_ideation_reddit.csv",
    'twitter': DATA_DIR / "suicidal_ideation_twitter.csv",
    'merged_pt': DATA_DIR / "merged_dataset_pt.csv",
    'reddit_pt': DATA_DIR / "suicidal_ideation_reddit_pt.csv",
    'twitter_pt': DATA_DIR / "suicidal_ideation_twitter_pt.csv"
}

