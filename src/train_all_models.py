import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle
import json
from pathlib import Path
import logging
import re
import nltk
from nltk.corpus import stopwords

try:
    from .evaluate import evaluate_model
    from .config import (
        RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE,
        TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF, TFIDF_NGRAM_RANGE,
        LOGISTIC_REGRESSION_MAX_ITER, LOGISTIC_REGRESSION_C,
        CV_FOLDS, MODELS_DIR, RESULTS_DIR, DATA_DIR
    )
except ImportError:
    from evaluate import evaluate_model
    from config import (
        RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE,
        TFIDF_MAX_FEATURES, TFIDF_MIN_DF, TFIDF_MAX_DF, TFIDF_NGRAM_RANGE,
        LOGISTIC_REGRESSION_MAX_ITER, LOGISTIC_REGRESSION_C,
        CV_FOLDS, MODELS_DIR, RESULTS_DIR, DATA_DIR
    )

nltk.download('stopwords', quiet=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_text(text, language='english'):
    if pd.isna(text) or text is None:
        return ""
    
    if language == 'portuguese':
        stop_words = set(stopwords.words('portuguese'))
    else:
        stop_words = set(stopwords.words('english'))
    
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-záàâãéèêíìîóòôõúùûç\s]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def preprocess_df(df, language='english'):
    df = df.dropna(subset=['text']).copy()
    df['clean_text'] = df['text'].apply(lambda x: clean_text(x, language))
    df = df[df['clean_text'].str.len() > 0].copy()
    return df

def load_reddit_en():
    df = pd.read_csv(DATA_DIR / "suicidal_ideation_reddit.csv")
    return pd.DataFrame({
        'text': df['usertext'],
        'label': df['label'].astype(int),
        'source': 'reddit'
    })

def load_twitter_en():
    df = pd.read_csv(DATA_DIR / "suicidal_ideation_twitter.csv")
    return pd.DataFrame({
        'text': df['Tweet'],
        'label': df['Suicide'].apply(
            lambda x: 1 if 'Potential Suicide post' in str(x) else 0
        ),
        'source': 'twitter'
    })

def load_merged_en():
    path = DATA_DIR / "merged_dataset.csv"
    if path.exists():
        return pd.read_csv(path)
    else:
        reddit = load_reddit_en()
        twitter = load_twitter_en()
        merged = pd.concat([reddit, twitter], ignore_index=True)
        merged.to_csv(path, index=False)
        return merged

def load_reddit_pt():
    pt_file = DATA_DIR / "suicidal_ideation_reddit_pt.csv"
    
    if not pt_file.exists():
        raise FileNotFoundError(f"Arquivo Reddit PT não encontrado: {pt_file}")
    
    df = pd.read_csv(pt_file)
    
    df_processed = pd.DataFrame({
        'text': df['usertext_pt'],
        'label': df['label'].astype(int),
        'source': 'reddit'
    })
    
    df_processed = df_processed[
        (df_processed['text'].notna()) & 
        (df_processed['text'] != "") & 
        (df_processed['text'] != "AJUSTE MANUAL")
    ]
    
    return df_processed

def load_twitter_pt():
    pt_file = DATA_DIR / "suicidal_ideation_twitter_pt.csv"
    
    if not pt_file.exists():
        raise FileNotFoundError(f"Arquivo Twitter PT não encontrado: {pt_file}")
    
    df = pd.read_csv(pt_file)
    
    df_processed = pd.DataFrame({
        'text': df['Tweet_pt'],
        'label': df['Suicide'].apply(
            lambda x: 1 if 'Potential Suicide post' in str(x) else 0
        ),
        'source': 'twitter'
    })
    
    df_processed = df_processed[
        (df_processed['text'].notna()) & 
        (df_processed['text'] != "") & 
        (df_processed['text'] != "AJUSTE MANUAL")
    ]
    
    return df_processed

def load_merged_pt():
    path = DATA_DIR / "merged_dataset_pt.csv"
    if path.exists():
        return pd.read_csv(path)
    else:
        reddit = load_reddit_pt()
        twitter = load_twitter_pt()
        merged = pd.concat([reddit, twitter], ignore_index=True)
        merged.to_csv(path, index=False)
        return merged

def train_logistic_regression(X_train, y_train, X_val, y_val):
    model = LogisticRegression(
        max_iter=LOGISTIC_REGRESSION_MAX_ITER,
        C=LOGISTIC_REGRESSION_C,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    metrics = evaluate_model(y_val, y_pred, y_proba, "Logistic Regression", save_results=False)
    return model, metrics

def train_svm(X_train, y_train, X_val, y_val):
    model = SVC(
        kernel='linear',
        probability=True,
        random_state=RANDOM_STATE,
        C=1.0
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    metrics = evaluate_model(y_val, y_pred, y_proba, "SVM", save_results=False)
    return model, metrics

def train_random_forest(X_train, y_train, X_val, y_val):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)
    metrics = evaluate_model(y_val, y_pred, y_proba, "Random Forest", save_results=False)
    return model, metrics

def cross_validate_model(model, X, y, cv_folds=CV_FOLDS):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted', n_jobs=-1)
    return {
        'mean': float(scores.mean()),
        'std': float(scores.std()),
        'scores': scores.tolist()
    }

def train_dataset_models(dataset_name, language, df):
    logger.info(f"\n{'='*80}")
    logger.info(f"TREINANDO MODELOS: {dataset_name.upper()} ({language.upper()})")
    logger.info(f"{'='*80}")
    
    df_processed = preprocess_df(df, language)
    X = df_processed['clean_text']
    y = df_processed['label']
    
    logger.info(f"Dataset: {len(df_processed)} registros")
    logger.info(f"Distribuição de classes: {y.value_counts().to_dict()}")
    
    if len(y.unique()) < 2:
        logger.warning(f"Dataset {dataset_name} ({language}) tem apenas uma classe. Pulando...")
        return None
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEST_SIZE + VALIDATION_SIZE, 
        stratify=y, random_state=RANDOM_STATE
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=TEST_SIZE/(TEST_SIZE + VALIDATION_SIZE),
        stratify=y_temp, random_state=RANDOM_STATE
    )
    
    logger.info(f"Divisão: Treino={len(X_train)}, Validação={len(X_val)}, Teste={len(X_test)}")
    
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        ngram_range=TFIDF_NGRAM_RANGE
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    
    models_results = {}
    models = {}
    
    logger.info("\nTreinando modelos...")
    lr_model, lr_metrics = train_logistic_regression(X_train_tfidf, y_train, X_val_tfidf, y_val)
    models_results['logistic_regression'] = lr_metrics
    models['logistic_regression'] = lr_model
    
    svm_model, svm_metrics = train_svm(X_train_tfidf, y_train, X_val_tfidf, y_val)
    models_results['svm'] = svm_metrics
    models['svm'] = svm_model
    
    rf_model, rf_metrics = train_random_forest(X_train_tfidf, y_train, X_val_tfidf, y_val)
    models_results['random_forest'] = rf_metrics
    models['random_forest'] = rf_model
    
    logger.info("\nValidação cruzada...")
    X_full = vectorizer.transform(X)
    cv_results = {}
    
    for name, model in models.items():
        cv = cross_validate_model(model, X_full, y)
        cv_results[name] = cv
        logger.info(f"  {name}: F1={cv['mean']:.4f} (+/- {cv['std']:.4f})")
    
    logger.info("\nAvaliação no conjunto de teste...")
    test_results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_tfidf)
        y_proba = model.predict_proba(X_test_tfidf)
        test_metrics = evaluate_model(y_test, y_pred, y_proba, f"{name} (Teste)", save_results=False)
        test_results[name] = test_metrics
    
    best_model_name = max(models_results.keys(), 
                         key=lambda k: models_results[k]['f1_score'])
    
    model_dir = MODELS_DIR / f"{dataset_name}_{language}"
    model_dir.mkdir(exist_ok=True)
    
    logger.info(f"\nSalvando modelos em {model_dir}...")
    
    with open(model_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    for name, model in models.items():
        with open(model_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(model, f)
    
    summary = {
        'dataset': dataset_name,
        'language': language,
        'n_samples': len(df_processed),
        'class_distribution': y.value_counts().to_dict(),
        'validation_metrics': {k: {m: float(v) for m, v in metrics.items()} 
                               for k, metrics in models_results.items()},
        'cross_validation': cv_results,
        'test_metrics': {k: {m: float(v) for m, v in metrics.items()} 
                        for k, metrics in test_results.items()},
        'best_model': best_model_name
    }
    
    summary_path = RESULTS_DIR / f"{dataset_name}_{language}_results.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Resultados salvos em {summary_path}")
    
    return summary

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Treina modelos para todos os datasets e idiomas")
    parser.add_argument("--datasets", nargs="+", 
                       choices=["reddit", "twitter", "merged", "all"],
                       default=["all"],
                       help="Quais datasets treinar (default: all)")
    parser.add_argument("--languages", nargs="+",
                       choices=["en", "pt", "all"],
                       default=["all"],
                       help="Quais idiomas treinar (default: all)")
    
    args = parser.parse_args()
    
    if "all" in args.datasets:
        datasets_to_train = ["reddit", "twitter", "merged"]
    else:
        datasets_to_train = args.datasets
    
    if "all" in args.languages:
        languages_to_train = ["en", "pt"]
    else:
        languages_to_train = args.languages
    
    loaders = {
        ('reddit', 'en'): load_reddit_en,
        ('twitter', 'en'): load_twitter_en,
        ('merged', 'en'): load_merged_en,
        ('reddit', 'pt'): load_reddit_pt,
        ('twitter', 'pt'): load_twitter_pt,
        ('merged', 'pt'): load_merged_pt,
    }
    
    all_results = []
    
    for dataset_name in datasets_to_train:
        for language in languages_to_train:
            key = (dataset_name, language)
            if key not in loaders:
                continue
            
            try:
                logger.info(f"\nCarregando {dataset_name} ({language})...")
                df = loaders[key]()
                
                if df is None or len(df) == 0:
                    logger.warning(f"Dataset {dataset_name} ({language}) está vazio. Pulando...")
                    continue
                
                result = train_dataset_models(dataset_name, language, df)
                if result:
                    all_results.append(result)
                    
            except FileNotFoundError as e:
                logger.error(f"Arquivo não encontrado para {dataset_name} ({language}): {e}")
            except Exception as e:
                logger.error(f"Erro ao treinar {dataset_name} ({language}): {e}", exc_info=True)
    
    if all_results:
        comparison_path = RESULTS_DIR / "model_comparison_all.json"
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*80}")
        logger.info("COMPARAÇÃO DE MODELOS")
        logger.info(f"{'='*80}")
        
        for result in all_results:
            best = result['best_model']
            test_f1 = result['test_metrics'][best]['f1_score']
            logger.info(f"{result['dataset']} ({result['language']}): "
                       f"Melhor={best}, F1-Teste={test_f1:.4f}")
        
        logger.info(f"\nComparação completa salva em {comparison_path}")
    
    logger.info("\nTreinamento concluído!")

if __name__ == "__main__":
    main()

