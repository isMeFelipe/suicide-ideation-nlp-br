import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from src.config import MODELS_DIR, RESULTS_DIR
from src.xai import ModelExplainer

class MultiModelExplainer:
    
    def __init__(self, dataset='reddit_en'):
        self.dataset = dataset
        self.dataset_path = MODELS_DIR / dataset
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        
        self.explainers = {}
        self.available_models = []
        
        self._load_models()
    
    def _load_models(self):
        model_files = {
            'logistic_regression': self.dataset_path / 'logistic_regression.pkl',
            'svm': self.dataset_path / 'svm.pkl',
            'random_forest': self.dataset_path / 'random_forest.pkl'
        }
        
        vectorizer_path = self.dataset_path / 'vectorizer.pkl'
        
        for model_name, model_path in model_files.items():
            if model_path.exists():
                try:
                    self.explainers[model_name] = ModelExplainer(
                        model_path=str(model_path),
                        vectorizer_path=str(vectorizer_path)
                    )
                    self.available_models.append(model_name)
                    print(f"✓ {model_name} carregado")
                except Exception as e:
                    print(f"✗ Erro ao carregar {model_name}: {e}")
    
    def get_model_names(self) -> List[str]:
        return self.available_models
    
    def explain_text_all_models(self, text: str, methods: List[str] = ['shap']) -> Dict[str, Any]:
        results = {
            'text': text,
            'models': {}
        }
        
        for model_name in self.available_models:
            explainer = self.explainers[model_name]
            
            try:
                pred_result = explainer.predict_with_explanation(text)
                
                results['models'][model_name] = {
                    'prediction': pred_result,
                    'explanations': {}
                }
                
                if 'shap' in methods:
                    try:
                        shap_result = explainer.explain_with_shap(text, max_features=15)
                        results['models'][model_name]['explanations']['shap'] = shap_result
                    except Exception as e:
                        results['models'][model_name]['explanations']['shap'] = {'error': str(e)}
                
                if 'lime' in methods:
                    try:
                        lime_result = explainer.explain_with_lime(text, num_features=15)
                        results['models'][model_name]['explanations']['lime'] = lime_result
                    except Exception as e:
                        results['models'][model_name]['explanations']['lime'] = {'error': str(e)}
                        
            except Exception as e:
                results['models'][model_name] = {'error': str(e)}
        
        return results
    
    def compare_predictions(self, text: str) -> pd.DataFrame:
        predictions = []
        
        for model_name in self.available_models:
            explainer = self.explainers[model_name]
            result = explainer.predict_with_explanation(text)
            
            predictions.append({
                'Modelo': model_name.replace('_', ' ').title(),
                'Predição': result['prediction_label'],
                'Confiança': f"{result['probability']:.2%}",
                'Prob. Não Suicida': f"{result['probabilities']['Não Suicida']:.2%}",
                'Prob. Suicida': f"{result['probabilities']['Suicida']:.2%}"
            })
        
        return pd.DataFrame(predictions)
    
    def compare_shap_features(self, text: str, top_n: int = 10) -> Dict[str, List]:
        comparison = {}
        
        for model_name in self.available_models:
            explainer = self.explainers[model_name]
            try:
                shap_result = explainer.explain_with_shap(text, max_features=top_n)
                comparison[model_name] = shap_result['top_features']
            except Exception as e:
                comparison[model_name] = {'error': str(e)}
        
        return comparison
    
    def plot_model_comparison(self, text: str, save_path: str = None, top_n: int = 10):
        comparison = self.compare_shap_features(text, top_n=top_n)
        
        n_models = len([m for m in self.available_models if m in comparison and 'error' not in comparison[m]])
        
        if n_models == 0:
            print("Nenhum modelo válido para comparação")
            return None
        
        fig, axes = plt.subplots(1, n_models, figsize=(8*n_models, 10))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(self.available_models):
            if model_name not in comparison or 'error' in comparison[model_name]:
                continue
            
            features = comparison[model_name][:top_n]
            feature_names = [f['feature'] for f in features]
            shap_values = [f['shap_value'] for f in features]
            
            colors = ['#d62728' if x > 0 else '#2ca02c' for x in shap_values]
            y_pos = np.arange(len(feature_names))
            
            axes[idx].barh(y_pos, shap_values, color=colors, alpha=0.7)
            axes[idx].set_yticks(y_pos)
            axes[idx].set_yticklabels(feature_names)
            axes[idx].set_xlabel('Valor SHAP', fontsize=11)
            axes[idx].set_title(f'{model_name.replace("_", " ").title()}', 
                               fontsize=13, fontweight='bold')
            axes[idx].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            axes[idx].invert_yaxis()
        
        fig.suptitle(f'Comparação de Explicações SHAP entre Modelos\nTexto: "{text[:60]}..."',
                     fontsize=15, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo: {save_path}")
        
        return fig
    
    def plot_predictions_comparison(self, text: str, save_path: str = None):
        
        predictions = []
        model_labels = []
        
        for model_name in self.available_models:
            explainer = self.explainers[model_name]
            result = explainer.predict_with_explanation(text)
            
            predictions.append([
                result['probabilities']['Não Suicida'],
                result['probabilities']['Suicida']
            ])
            model_labels.append(model_name.replace('_', ' ').title())
        
        predictions = np.array(predictions)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        x = np.arange(len(model_labels))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, predictions[:, 0], width, 
                       label='Não Suicida', color='#2ca02c', alpha=0.8)
        bars2 = ax1.bar(x + width/2, predictions[:, 1], width,
                       label='Suicida', color='#d62728', alpha=0.8)
        
        ax1.set_xlabel('Modelos', fontsize=12)
        ax1.set_ylabel('Probabilidade', fontsize=12)
        ax1.set_title('Probabilidades por Modelo', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_labels, rotation=15, ha='right')
        ax1.legend()
        ax1.set_ylim([0, 1])
        ax1.grid(axis='y', alpha=0.3)
        
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax1.text(bar1.get_x() + bar1.get_width()/2., height1,
                    f'{height1:.1%}', ha='center', va='bottom', fontsize=9)
            ax1.text(bar2.get_x() + bar2.get_width()/2., height2,
                    f'{height2:.1%}', ha='center', va='bottom', fontsize=9)
        
        predictions_class = [1 if p[1] > p[0] else 0 for p in predictions]
        confidences = [max(p) for p in predictions]
        colors_pred = ['#d62728' if p == 1 else '#2ca02c' for p in predictions_class]
        
        ax2.barh(model_labels, confidences, color=colors_pred, alpha=0.7)
        ax2.set_xlabel('Confiança', fontsize=12)
        ax2.set_title('Confiança das Predições', fontsize=14, fontweight='bold')
        ax2.set_xlim([0, 1])
        ax2.grid(axis='x', alpha=0.3)
        
        for i, (conf, pred) in enumerate(zip(confidences, predictions_class)):
            label = 'Suicida' if pred == 1 else 'Não Suicida'
            ax2.text(conf, i, f'  {conf:.1%} ({label})', 
                    va='center', fontsize=10, fontweight='bold')
        
        fig.suptitle(f'Comparação de Predições entre Modelos\nTexto: "{text[:60]}..."',
                     fontsize=15, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico salvo: {save_path}")
        
        return fig
    
    def generate_comparison_report(self, text: str, save_dir: str = None):
        
        if save_dir is None:
            save_dir = RESULTS_DIR / 'xai_multi_model' / self.dataset
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        results = self.explain_text_all_models(text, methods=['shap', 'lime'])
        
        report_path = save_dir / f'comparison_report_{timestamp}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RELATÓRIO COMPARATIVO DE EXPLICABILIDADE (MÚLTIPLOS MODELOS)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Texto Original:\n{text}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("COMPARAÇÃO DE PREDIÇÕES\n")
            f.write("-"*80 + "\n\n")
            
            df_pred = self.compare_predictions(text)
            f.write(df_pred.to_string(index=False))
            f.write("\n\n")
            
            for model_name in self.available_models:
                if model_name not in results['models'] or 'error' in results['models'][model_name]:
                    continue
                
                model_data = results['models'][model_name]
                
                f.write("="*80 + "\n")
                f.write(f"MODELO: {model_name.upper()}\n")
                f.write("="*80 + "\n\n")
                
                pred = model_data['prediction']
                f.write(f"Predição: {pred['prediction_label']}\n")
                f.write(f"Confiança: {pred['probability']:.2%}\n\n")
                
                if 'shap' in model_data['explanations'] and 'error' not in model_data['explanations']['shap']:
                    f.write("-"*80 + "\n")
                    f.write("TOP 15 FEATURES (SHAP)\n")
                    f.write("-"*80 + "\n")
                    
                    for i, feat in enumerate(model_data['explanations']['shap']['top_features'][:15], 1):
                        f.write(f"{i:2d}. {feat['feature']:20s} | {feat['shap_value']:+8.4f} | {feat['contribution']}\n")
                    f.write("\n")
                
                if 'lime' in model_data['explanations'] and 'error' not in model_data['explanations']['lime']:
                    f.write("-"*80 + "\n")
                    f.write("TOP 15 FEATURES (LIME)\n")
                    f.write("-"*80 + "\n")
                    
                    for i, feat in enumerate(model_data['explanations']['lime']['top_features'][:15], 1):
                        f.write(f"{i:2d}. {feat['feature']:20s} | {feat['weight']:+8.4f} | {feat['contribution']}\n")
                    f.write("\n")
            
            f.write("="*80 + "\n")
            f.write(f"Relatório gerado em: {pd.Timestamp.now()}\n")
            f.write("="*80 + "\n")
        
        print(f"Relatório salvo: {report_path}")
        
        pred_plot_path = save_dir / f'predictions_comparison_{timestamp}.png'
        self.plot_predictions_comparison(text, save_path=str(pred_plot_path))
        
        shap_plot_path = save_dir / f'shap_comparison_{timestamp}.png'
        self.plot_model_comparison(text, save_path=str(shap_plot_path))
        
        return {
            'report': str(report_path),
            'plots': {
                'predictions': str(pred_plot_path),
                'shap_comparison': str(shap_plot_path)
            },
            'results': results
        }


def main():
    print("="*80)
    print("COMPARAÇÃO DE EXPLICABILIDADE ENTRE MODELOS")
    print("="*80)
    
    datasets = ['reddit_en', 'twitter_en']
    
    test_cases = [
        {
            'label': 'Alto Risco',
            'text': "I can't take it anymore. Life is meaningless and I want to end it all."
        },
        {
            'label': 'Sem Risco',
            'text': "Had an amazing day with family. Life is beautiful and I'm grateful!"
        }
    ]
    
    for dataset in datasets:
        print(f"\n{'='*80}")
        print(f"DATASET: {dataset.upper()}")
        print(f"{'='*80}")
        
        try:
            multi_explainer = MultiModelExplainer(dataset=dataset)
            
            print(f"\nModelos disponíveis: {', '.join(multi_explainer.get_model_names())}")
            
            for case in test_cases:
                print(f"\n{'-'*80}")
                print(f"Caso: {case['label']}")
                print(f"Texto: {case['text']}")
                print(f"{'-'*80}\n")
                
                df_pred = multi_explainer.compare_predictions(case['text'])
                print(df_pred.to_string(index=False))
                
                print("\nTop 5 Features SHAP por Modelo:")
                comparison = multi_explainer.compare_shap_features(case['text'], top_n=5)
                
                for model_name, features in comparison.items():
                    if 'error' in features:
                        continue
                    print(f"\n  {model_name.upper()}:")
                    for i, feat in enumerate(features[:5], 1):
                        icon = "🔴" if feat['shap_value'] > 0 else "🟢"
                        print(f"    {i}. {icon} {feat['feature']:15s} | {feat['shap_value']:+7.4f}")
            
            print(f"\n{'='*80}")
            print(f"GERANDO RELATÓRIO COMPLETO PARA {dataset.upper()}")
            print(f"{'='*80}\n")
            
            report = multi_explainer.generate_comparison_report(test_cases[0]['text'])
            print(f"\n✓ Relatório: {report['report']}")
            print(f"✓ Gráficos: {len(report['plots'])} gerados")
            
        except Exception as e:
            print(f"✗ Erro ao processar {dataset}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

