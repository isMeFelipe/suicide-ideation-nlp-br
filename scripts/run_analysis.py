#!/usr/bin/env python3
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from logger_config import setup_logger
from preprocess import load_datasets, preprocess
from eda import analyze_dataset
from validate_data import validate_merged_dataset

logger = setup_logger()

def main():
    logger.info("="*60)
    logger.info("ANÁLISE COMPLETA DO DATASET")
    logger.info("="*60)
    
    logger.info("\n1. Validando dataset...")
    if not validate_merged_dataset():
        logger.error("Validação falhou. Corrija os erros antes de continuar.")
        return
    
    logger.info("\n2. Carregando dataset...")
    df = load_datasets()
    logger.info(f"Dataset carregado: {len(df)} registros")
    
    logger.info("\n3. Pré-processando dados...")
    df_processed = preprocess(df)
    logger.info(f"Dataset processado: {len(df_processed)} registros")
    
    logger.info("\n4. Realizando análise exploratória...")
    df_analyzed = analyze_dataset(df_processed, save_plots=True)
    
    logger.info("\n" + "="*60)
    logger.info("ANÁLISE CONCLUÍDA!")
    logger.info("="*60)
    logger.info("Verifique os gráficos em: results/eda/")

if __name__ == "__main__":
    main()

