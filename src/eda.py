import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import numpy as np
try:
    from .config import RESULTS_DIR, DATA_DIR
except ImportError:
    from config import RESULTS_DIR, DATA_DIR

def analyze_dataset(df, save_plots=True):
    print("="*60)
    print("ANÁLISE EXPLORATÓRIA DE DADOS (EDA)")
    print("="*60)
    
    eda_dir = RESULTS_DIR / "eda"
    eda_dir.mkdir(exist_ok=True)
    
    print(f"\n1. Informações Gerais do Dataset:")
    print(f"   Total de registros: {len(df):,}")
    print(f"   Total de colunas: {len(df.columns)}")
    print(f"   Colunas: {', '.join(df.columns)}")
    
    print(f"\n2. Distribuição de Classes:")
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        print(f"   Classe 0 (Sem Risco): {label_counts.get(0, 0):,} ({label_counts.get(0, 0)/len(df)*100:.2f}%)")
        print(f"   Classe 1 (Risco): {label_counts.get(1, 0):,} ({label_counts.get(1, 0)/len(df)*100:.2f}%)")
        
        if save_plots:
            plt.figure(figsize=(8, 6))
            label_counts.plot(kind='bar', color=['#3498db', '#e74c3c'])
            plt.title('Distribuição de Classes')
            plt.xlabel('Classe')
            plt.ylabel('Frequência')
            plt.xticks([0, 1], ['Sem Risco (0)', 'Risco (1)'], rotation=0)
            plt.tight_layout()
            plt.savefig(eda_dir / "class_distribution.png", dpi=300)
            plt.close()
    
    print(f"\n3. Distribuição por Fonte:")
    if 'source' in df.columns:
        source_counts = df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"   {source.capitalize()}: {count:,} ({count/len(df)*100:.2f}%)")
        
        if save_plots:
            plt.figure(figsize=(8, 6))
            source_counts.plot(kind='bar', color=['#9b59b6', '#f39c12'])
            plt.title('Distribuição por Fonte de Dados')
            plt.xlabel('Fonte')
            plt.ylabel('Frequência')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(eda_dir / "source_distribution.png", dpi=300)
            plt.close()
    
    print(f"\n4. Análise de Texto:")
    if 'text' in df.columns:
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        
        print(f"   Comprimento médio do texto: {df['text_length'].mean():.1f} caracteres")
        print(f"   Comprimento mediano: {df['text_length'].median():.1f} caracteres")
        print(f"   Número médio de palavras: {df['word_count'].mean():.1f}")
        print(f"   Número mediano de palavras: {df['word_count'].median():.1f}")
        
        if save_plots:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].hist(df['text_length'], bins=50, color='#3498db', edgecolor='black')
            axes[0].set_title('Distribuição do Comprimento do Texto')
            axes[0].set_xlabel('Número de Caracteres')
            axes[0].set_ylabel('Frequência')
            
            axes[1].hist(df['word_count'], bins=50, color='#e74c3c', edgecolor='black')
            axes[1].set_title('Distribuição do Número de Palavras')
            axes[1].set_xlabel('Número de Palavras')
            axes[1].set_ylabel('Frequência')
            
            plt.tight_layout()
            plt.savefig(eda_dir / "text_length_distribution.png", dpi=300)
            plt.close()
    
    print(f"\n5. Análise de Texto Limpo:")
    if 'clean_text' in df.columns:
        df['clean_text_length'] = df['clean_text'].str.len()
        df['clean_word_count'] = df['clean_text'].str.split().str.len()
        
        print(f"   Comprimento médio do texto limpo: {df['clean_text_length'].mean():.1f} caracteres")
        print(f"   Número médio de palavras (limpo): {df['clean_word_count'].mean():.1f}")
    
    print(f"\n6. Valores Ausentes:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        for col, count in missing[missing > 0].items():
            print(f"   {col}: {count} ({count/len(df)*100:.2f}%)")
    else:
        print("   Nenhum valor ausente encontrado.")
    
    if 'label' in df.columns and 'source' in df.columns and save_plots:
        print(f"\n7. Distribuição de Classes por Fonte:")
        cross_tab = pd.crosstab(df['source'], df['label'])
        print(cross_tab)
        
        plt.figure(figsize=(10, 6))
        cross_tab.plot(kind='bar', stacked=False, color=['#3498db', '#e74c3c'])
        plt.title('Distribuição de Classes por Fonte')
        plt.xlabel('Fonte')
        plt.ylabel('Frequência')
        plt.xticks(rotation=45)
        plt.legend(['Sem Risco (0)', 'Risco (1)'])
        plt.tight_layout()
        plt.savefig(eda_dir / "class_by_source.png", dpi=300)
        plt.close()
    
    print(f"\n{'='*60}")
    print(f"Análise completa! Gráficos salvos em: {eda_dir}")
    print(f"{'='*60}\n")
    
    return df

