import pandas as pd
import json
from pathlib import Path
from logger_config import setup_logger

logger = setup_logger()

RESULTS_DIR = Path(__file__).parent.parent / "results"

def load_all_results():
    results = []
    for result_file in RESULTS_DIR.glob("*_results.json"):
        if result_file.name == "model_comparison_all.json":
            continue
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                results.append(json.load(f))
        except Exception as e:
            logger.warning(f"Erro ao carregar {result_file}: {e}")
    return results

def create_comparison_table(results):
    comparison_data = []
    
    for result in results:
        dataset = result['dataset']
        language = result['language']
        best_model = result['best_model']
        
        val_metrics = result['validation_metrics'][best_model]
        test_metrics = result['test_metrics'][best_model]
        cv_metrics = result['cross_validation'][best_model]
        
        comparison_data.append({
            'Dataset': dataset,
            'Idioma': language,
            'Melhor Modelo': best_model,
            'N Amostras': result['n_samples'],
            'F1-Validação': f"{val_metrics['f1_score']:.4f}",
            'F1-Teste': f"{test_metrics['f1_score']:.4f}",
            'F1-CV (média)': f"{cv_metrics['mean']:.4f}",
            'F1-CV (std)': f"{cv_metrics['std']:.4f}",
            'Precisão-Teste': f"{test_metrics['precision']:.4f}",
            'Recall-Teste': f"{test_metrics['recall']:.4f}",
            'Acurácia-Teste': f"{test_metrics['accuracy']:.4f}"
        })
    
    df = pd.DataFrame(comparison_data)
    return df

def create_detailed_comparison(results):
    detailed_data = []
    
    for result in results:
        dataset = result['dataset']
        language = result['language']
        
        for model_name in ['logistic_regression', 'svm', 'random_forest']:
            if model_name not in result['test_metrics']:
                continue
            
            test_metrics = result['test_metrics'][model_name]
            cv_metrics = result['cross_validation'][model_name]
            
            detailed_data.append({
                'Dataset': dataset,
                'Idioma': language,
                'Modelo': model_name,
                'F1-Teste': test_metrics['f1_score'],
                'Precisão-Teste': test_metrics['precision'],
                'Recall-Teste': test_metrics['recall'],
                'Acurácia-Teste': test_metrics['accuracy'],
                'F1-CV (média)': cv_metrics['mean'],
                'F1-CV (std)': cv_metrics['std']
            })
    
    return pd.DataFrame(detailed_data)

def main():
    logger.info("Carregando resultados...")
    results = load_all_results()
    
    if not results:
        logger.error("Nenhum resultado encontrado!")
        logger.info("Execute primeiro: python src/train_all_models.py")
        return
    
    logger.info(f"Encontrados {len(results)} resultados")
    
    comparison_df = create_comparison_table(results)
    detailed_df = create_detailed_comparison(results)
    
    output_path = RESULTS_DIR / "model_comparison_table.csv"
    comparison_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    logger.info(f"\nTabela de comparação salva em: {output_path}")
    
    detailed_path = RESULTS_DIR / "model_comparison_detailed.csv"
    detailed_df.to_csv(detailed_path, index=False, encoding='utf-8-sig')
    logger.info(f"Comparação detalhada salva em: {detailed_path}")
    
    logger.info("\n" + "="*80)
    logger.info("RESUMO DA COMPARAÇÃO")
    logger.info("="*80)
    print("\n", comparison_df.to_string(index=False))
    
    logger.info("\n" + "="*80)
    logger.info("MELHORES MODELOS POR DATASET")
    logger.info("="*80)
    
    for dataset in ['reddit', 'twitter', 'merged']:
        for lang in ['en', 'pt']:
            subset = comparison_df[
                (comparison_df['Dataset'] == dataset) & 
                (comparison_df['Idioma'] == lang)
            ]
            if not subset.empty:
                best = subset.loc[subset['F1-Teste'].astype(float).idxmax()]
                logger.info(f"\n{dataset.upper()} ({lang.upper()}):")
                logger.info(f"  Modelo: {best['Melhor Modelo']}")
                logger.info(f"  F1-Teste: {best['F1-Teste']}")
                logger.info(f"  F1-CV: {best['F1-CV (média)']} (+/- {best['F1-CV (std)']})")

if __name__ == "__main__":
    main()

