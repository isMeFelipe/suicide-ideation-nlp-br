import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import shap
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from src.config import MODEL_FILES, RESULTS_DIR
from src.preprocess import clean_text

class ModelExplainer:
    
    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        if model_path is None:
            model_path = MODEL_FILES['logistic']
        if vectorizer_path is None:
            vectorizer_path = MODEL_FILES['vectorizer']
            
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        
        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)
        
        self.model_name = Path(model_path).stem
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        self.shap_explainer = None
        self.lime_explainer = None
        
    def _get_shap_explainer(self):
        if self.shap_explainer is None:
            from shap.maskers import Independent
            background_data = self.vectorizer.transform(["text sample"])
            masker = Independent(background_data)
            
            model_type = type(self.model).__name__
            
            if model_type in ['LogisticRegression', 'LinearSVC', 'SGDClassifier']:
                self.shap_explainer = shap.LinearExplainer(self.model, masker)
            elif model_type in ['RandomForestClassifier', 'GradientBoostingClassifier']:
                self.shap_explainer = shap.TreeExplainer(self.model)
            else:
                self.shap_explainer = shap.KernelExplainer(
                    self.model.predict_proba,
                    background_data
                )
        
        return self.shap_explainer
    
    def _get_lime_explainer(self):
        if self.lime_explainer is None:
            self.lime_explainer = LimeTextExplainer(
                class_names=['Não Suicida', 'Suicida'],
                bow=True
            )
        return self.lime_explainer
    
    def predict_with_explanation(self, text: str) -> Dict[str, Any]:
        clean = clean_text(text)
        tfidf_input = self.vectorizer.transform([clean])
        
        pred = self.model.predict(tfidf_input)[0]
        proba = self.model.predict_proba(tfidf_input)[0]
        
        return {
            'text': text,
            'cleaned_text': clean,
            'prediction': int(pred),
            'prediction_label': 'Suicida' if pred == 1 else 'Não Suicida',
            'probability': float(proba[pred]),
            'probabilities': {
                'Não Suicida': float(proba[0]),
                'Suicida': float(proba[1])
            }
        }
    
    def explain_with_shap(self, text: str, max_features: int = 20) -> Dict[str, Any]:
        clean = clean_text(text)
        tfidf_input = self.vectorizer.transform([clean])
        
        explainer = self._get_shap_explainer()
        shap_values = explainer.shap_values(tfidf_input)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        if hasattr(shap_values, 'toarray'):
            shap_values_array = shap_values.toarray()[0]
        else:
            shap_values_array = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
        feature_values = tfidf_input.toarray()[0]
        
        non_zero_indices = np.where(feature_values > 0)[0]
        
        relevant_features = []
        for idx in non_zero_indices:
            relevant_features.append({
                'feature': self.feature_names[idx],
                'shap_value': float(shap_values_array[idx]),
                'tfidf_value': float(feature_values[idx]),
                'contribution': 'Positiva' if shap_values_array[idx] > 0 else 'Negativa'
            })
        
        relevant_features.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        return {
            'method': 'SHAP',
            'expected_value': float(explainer.expected_value),
            'top_features': relevant_features[:max_features],
            'all_features': relevant_features,
            'shap_values': shap_values_array,
            'feature_names': self.feature_names
        }
    
    def explain_with_lime(self, text: str, num_features: int = 20) -> Dict[str, Any]:
        
        def predict_proba_pipeline(texts):
            cleaned = [clean_text(t) for t in texts]
            tfidf = self.vectorizer.transform(cleaned)
            return self.model.predict_proba(tfidf)
        
        explainer = self._get_lime_explainer()
        explanation = explainer.explain_instance(
            text,
            predict_proba_pipeline,
            num_features=num_features,
            top_labels=2
        )
        
        pred_class = 1 if self.model.predict(self.vectorizer.transform([clean_text(text)]))[0] == 1 else 0
        
        lime_features = explanation.as_list(label=pred_class)
        
        formatted_features = []
        for feature, weight in lime_features:
            formatted_features.append({
                'feature': feature,
                'weight': float(weight),
                'contribution': 'Positiva' if weight > 0 else 'Negativa'
            })
        
        return {
            'method': 'LIME',
            'predicted_class': pred_class,
            'top_features': formatted_features,
            'explanation_object': explanation
        }
    
    def explain_text(self, text: str, methods: List[str] = ['shap', 'lime']) -> Dict[str, Any]:
        
        result = self.predict_with_explanation(text)
        
        explanations = {}
        
        if 'shap' in methods:
            try:
                explanations['shap'] = self.explain_with_shap(text)
            except Exception as e:
                explanations['shap'] = {'error': str(e)}
        
        if 'lime' in methods:
            try:
                explanations['lime'] = self.explain_with_lime(text)
            except Exception as e:
                explanations['lime'] = {'error': str(e)}
        
        result['explanations'] = explanations
        
        return result
    
    def plot_shap_explanation(self, text: str, save_path: str = None, max_features: int = 15):
        
        clean = clean_text(text)
        tfidf_input = self.vectorizer.transform([clean])
        
        explainer = self._get_shap_explainer()
        shap_values = explainer.shap_values(tfidf_input)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        if hasattr(shap_values, 'toarray'):
            shap_values_array = shap_values.toarray()[0]
        else:
            shap_values_array = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
        feature_values = tfidf_input.toarray()[0]
        
        non_zero_indices = np.where(feature_values > 0)[0]
        non_zero_shap = shap_values_array[non_zero_indices]
        non_zero_features = self.feature_names[non_zero_indices]
        
        top_indices = np.argsort(np.abs(non_zero_shap))[-max_features:]
        top_shap = non_zero_shap[top_indices]
        top_features = non_zero_features[top_indices]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#d62728' if x > 0 else '#2ca02c' for x in top_shap]
        y_pos = np.arange(len(top_features))
        
        ax.barh(y_pos, top_shap, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Valor SHAP (Contribuição para Predição)', fontsize=12)
        ax.set_title(f'Top {max_features} Features - SHAP Explanation\nTexto: "{text[:50]}..."', 
                     fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        
        ax.text(0.02, 0.98, '← Negativo (Não Suicida)', transform=ax.transAxes,
                verticalalignment='top', color='#2ca02c', fontweight='bold')
        ax.text(0.98, 0.98, 'Positivo (Suicida) →', transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right', 
                color='#d62728', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico SHAP salvo em: {save_path}")
        
        return fig
    
    def plot_lime_explanation(self, text: str, save_path: str = None, num_features: int = 15):
        
        explanation_data = self.explain_with_lime(text, num_features=num_features)
        features = explanation_data['top_features']
        
        feature_names = [f['feature'] for f in features]
        weights = [f['weight'] for f in features]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#d62728' if x > 0 else '#2ca02c' for x in weights]
        y_pos = np.arange(len(feature_names))
        
        ax.barh(y_pos, weights, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('Peso da Feature (LIME)', fontsize=12)
        ax.set_title(f'Top {num_features} Features - LIME Explanation\nTexto: "{text[:50]}..."',
                     fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        
        ax.text(0.02, 0.98, '← Negativo (Não Suicida)', transform=ax.transAxes,
                verticalalignment='top', color='#2ca02c', fontweight='bold')
        ax.text(0.98, 0.98, 'Positivo (Suicida) →', transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                color='#d62728', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico LIME salvo em: {save_path}")
        
        return fig
    
    def plot_comparison(self, text: str, save_path: str = None, max_features: int = 10):
        
        shap_data = self.explain_with_shap(text, max_features=max_features)
        lime_data = self.explain_with_lime(text, num_features=max_features)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        shap_features = shap_data['top_features']
        shap_names = [f['feature'] for f in shap_features]
        shap_values = [f['shap_value'] for f in shap_features]
        
        colors_shap = ['#d62728' if x > 0 else '#2ca02c' for x in shap_values]
        y_pos = np.arange(len(shap_names))
        
        ax1.barh(y_pos, shap_values, color=colors_shap, alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(shap_names)
        ax1.set_xlabel('Valor SHAP', fontsize=11)
        ax1.set_title('SHAP Explanation', fontsize=13, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        
        lime_features = lime_data['top_features']
        lime_names = [f['feature'] for f in lime_features]
        lime_weights = [f['weight'] for f in lime_features]
        
        colors_lime = ['#d62728' if x > 0 else '#2ca02c' for x in lime_weights]
        y_pos2 = np.arange(len(lime_names))
        
        ax2.barh(y_pos2, lime_weights, color=colors_lime, alpha=0.7)
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(lime_names)
        ax2.set_xlabel('Peso LIME', fontsize=11)
        ax2.set_title('LIME Explanation', fontsize=13, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        
        fig.suptitle(f'Comparação SHAP vs LIME\nTexto: "{text[:60]}..."',
                     fontsize=15, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico de comparação salvo em: {save_path}")
        
        return fig
    
    def get_global_feature_importance(self, X_test: pd.Series, y_test: pd.Series, 
                                     top_n: int = 30) -> pd.DataFrame:
        
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        model_type = type(self.model).__name__
        
        if model_type == 'LogisticRegression':
            coefficients = self.model.coef_[0]
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': np.abs(coefficients),
                'coefficient': coefficients
            })
            importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
            
        elif model_type in ['RandomForestClassifier', 'GradientBoostingClassifier']:
            importances = self.model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            })
            importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        else:
            explainer = self._get_shap_explainer()
            shap_values = explainer.shap_values(X_test_tfidf)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            if hasattr(shap_values, 'toarray'):
                shap_values = shap_values.toarray()
            
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_abs_shap
            })
            importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def plot_global_importance(self, X_test: pd.Series, y_test: pd.Series, 
                              save_path: str = None, top_n: int = 20):
        
        importance_df = self.get_global_feature_importance(X_test, y_test, top_n=top_n)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        y_pos = np.arange(len(importance_df))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
        
        ax.barh(y_pos, importance_df['importance'].values, color=colors, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['feature'].values)
        ax.set_xlabel('Importância', fontsize=12)
        ax.set_title(f'Top {top_n} Features Mais Importantes (Global)\nModelo: {self.model_name}',
                     fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico de importância global salvo em: {save_path}")
        
        return fig
    
    def generate_report(self, text: str, save_dir: str = None) -> Dict[str, Any]:
        
        if save_dir is None:
            save_dir = RESULTS_DIR / 'xai_reports'
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        full_explanation = self.explain_text(text, methods=['shap', 'lime'])
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            shap_plot_path = save_dir / f'shap_explanation_{timestamp}.png'
            self.plot_shap_explanation(text, save_path=str(shap_plot_path))
            full_explanation['plots'] = {'shap': str(shap_plot_path)}
        except Exception as e:
            print(f"Erro ao gerar gráfico SHAP: {e}")
        
        try:
            lime_plot_path = save_dir / f'lime_explanation_{timestamp}.png'
            self.plot_lime_explanation(text, save_path=str(lime_plot_path))
            full_explanation['plots']['lime'] = str(lime_plot_path)
        except Exception as e:
            print(f"Erro ao gerar gráfico LIME: {e}")
        
        try:
            comparison_plot_path = save_dir / f'comparison_{timestamp}.png'
            self.plot_comparison(text, save_path=str(comparison_plot_path))
            full_explanation['plots']['comparison'] = str(comparison_plot_path)
        except Exception as e:
            print(f"Erro ao gerar gráfico de comparação: {e}")
        
        report_path = save_dir / f'explanation_report_{timestamp}.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("RELATÓRIO DE EXPLICABILIDADE (XAI)\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Texto Original:\n{text}\n\n")
            f.write(f"Texto Limpo:\n{full_explanation['cleaned_text']}\n\n")
            
            f.write("-"*80 + "\n")
            f.write("PREDIÇÃO\n")
            f.write("-"*80 + "\n")
            f.write(f"Classe Predita: {full_explanation['prediction_label']}\n")
            f.write(f"Confiança: {full_explanation['probability']:.2%}\n")
            f.write(f"Probabilidades:\n")
            for label, prob in full_explanation['probabilities'].items():
                f.write(f"  - {label}: {prob:.2%}\n")
            
            if 'shap' in full_explanation['explanations']:
                f.write("\n" + "-"*80 + "\n")
                f.write("EXPLICAÇÃO SHAP (Top 15 Features)\n")
                f.write("-"*80 + "\n")
                shap_data = full_explanation['explanations']['shap']
                for i, feat in enumerate(shap_data['top_features'][:15], 1):
                    f.write(f"{i:2d}. {feat['feature']:20s} | Valor SHAP: {feat['shap_value']:+8.4f} | {feat['contribution']}\n")
            
            if 'lime' in full_explanation['explanations']:
                f.write("\n" + "-"*80 + "\n")
                f.write("EXPLICAÇÃO LIME (Top 15 Features)\n")
                f.write("-"*80 + "\n")
                lime_data = full_explanation['explanations']['lime']
                for i, feat in enumerate(lime_data['top_features'][:15], 1):
                    f.write(f"{i:2d}. {feat['feature']:20s} | Peso: {feat['weight']:+8.4f} | {feat['contribution']}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write(f"Relatório gerado em: {pd.Timestamp.now()}\n")
            f.write("="*80 + "\n")
        
        print(f"\nRelatório completo salvo em: {report_path}")
        full_explanation['report_path'] = str(report_path)
        
        return full_explanation


def main():
    explainer = ModelExplainer()
    
    test_texts = [
        "I feel hopeless and don't want to live anymore. Life has no meaning.",
        "Just had a great day with friends! Feeling blessed and happy.",
        "I'm feeling really depressed lately but trying to get help.",
        "Can't take it anymore. Nobody would miss me if I was gone.",
        "Looking forward to my vacation next week. So excited!"
    ]
    
    print("="*80)
    print("DEMONSTRAÇÃO DE EXPLICABILIDADE (XAI)")
    print("="*80)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*80}")
        print(f"EXEMPLO {i}")
        print(f"{'='*80}")
        print(f"Texto: {text}")
        print("-"*80)
        
        result = explainer.explain_text(text)
        
        print(f"\nPredição: {result['prediction_label']}")
        print(f"Confiança: {result['probability']:.2%}")
        print(f"Probabilidades: Não Suicida={result['probabilities']['Não Suicida']:.2%}, Suicida={result['probabilities']['Suicida']:.2%}")
        
        if 'shap' in result['explanations'] and 'error' not in result['explanations']['shap']:
            print("\nTop 5 Features (SHAP):")
            for j, feat in enumerate(result['explanations']['shap']['top_features'][:5], 1):
                print(f"  {j}. {feat['feature']:15s} | Valor: {feat['shap_value']:+7.4f} | {feat['contribution']}")
        
        if 'lime' in result['explanations'] and 'error' not in result['explanations']['lime']:
            print("\nTop 5 Features (LIME):")
            for j, feat in enumerate(result['explanations']['lime']['top_features'][:5], 1):
                print(f"  {j}. {feat['feature']:15s} | Peso: {feat['weight']:+7.4f} | {feat['contribution']}")
    
    print(f"\n{'='*80}")
    print("GERANDO RELATÓRIO COMPLETO PARA PRIMEIRO EXEMPLO")
    print(f"{'='*80}")
    
    report = explainer.generate_report(test_texts[0])
    
    print(f"\n{'='*80}")
    print("XAI DEMO COMPLETO!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()


