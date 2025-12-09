#!/usr/bin/env python3
"""
Script completo para treinar todos os modelos e gerar figuras para o TCC
"""

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import subprocess
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    logger.info(f"\n{'='*80}")
    logger.info(f"{description}")
    logger.info(f"{'='*80}")
    logger.info(f"Executando: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"Erro ao executar: {description}")
        logger.error(f"Stderr: {result.stderr}")
        return False
    else:
        logger.info(f"✓ {description} concluído com sucesso!")
        if result.stdout:
            logger.info(f"Output: {result.stdout[-500:]}")
        return True

def main():
    base_dir = Path(__file__).parent.parent
    
    logger.info("="*80)
    logger.info("TREINAMENTO DE MODELOS E GERAÇÃO DE FIGURAS PARA TCC")
    logger.info("="*80)
    
    steps = [
        {
            'description': 'Criando dataset mesclado em português (se necessário)',
            'cmd': ['python3', str(base_dir / 'src' / 'merge_translated_datasets.py')],
            'optional': True
        },
        {
            'description': 'Treinando modelos para todos os datasets em português',
            'cmd': ['python3', str(base_dir / 'src' / 'train_all_models.py'), 
                   '--datasets', 'reddit', 'twitter', 'merged',
                   '--languages', 'pt'],
            'optional': False
        },
        {
            'description': 'Treinando modelos para todos os datasets em inglês',
            'cmd': ['python3', str(base_dir / 'src' / 'train_all_models.py'), 
                   '--datasets', 'reddit', 'twitter', 'merged',
                   '--languages', 'en'],
            'optional': False
        },
        {
            'description': 'Gerando todas as figuras para o TCC (com dados reais)',
            'cmd': ['python3', str(base_dir / 'scripts' / 'update_figures_with_real_data.py')],
            'optional': False
        },
        {
            'description': 'Gerando figuras XAI (SHAP e LIME)',
            'cmd': ['python3', str(base_dir / 'scripts' / 'generate_xai_figures.py')],
            'optional': True
        },
        {
            'description': 'Gerando figuras adicionais do TCC',
            'cmd': ['python3', str(base_dir / 'scripts' / 'generate_tcc_figures.py')],
            'optional': True
        }
    ]
    
    success_count = 0
    failed_steps = []
    
    for step in steps:
        success = run_command(step['cmd'], step['description'])
        
        if success:
            success_count += 1
        else:
            if not step.get('optional', False):
                failed_steps.append(step['description'])
                logger.error(f"Falha crítica em: {step['description']}")
                logger.error("Interrompendo execução...")
                break
    
    logger.info("\n" + "="*80)
    logger.info("RESUMO DA EXECUÇÃO")
    logger.info("="*80)
    logger.info(f"Passos concluídos: {success_count}/{len(steps)}")
    
    if failed_steps:
        logger.error(f"\nPassos que falharam:")
        for step in failed_steps:
            logger.error(f"  - {step}")
        return 1
    else:
        logger.info("\n✓ Todos os passos foram concluídos com sucesso!")
        logger.info(f"\nModelos treinados em: {base_dir / 'models'}")
        logger.info(f"Resultados salvos em: {base_dir / 'results'}")
        logger.info(f"Figuras geradas em: {base_dir / 'tcc_figures'}")
        return 0

if __name__ == "__main__":
    sys.exit(main())

