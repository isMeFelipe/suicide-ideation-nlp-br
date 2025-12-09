import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

try:
    from logger_config import setup_logger
    from config import DATA_DIR
except ImportError:
    from src.logger_config import setup_logger
    from src.config import DATA_DIR

logger = setup_logger()

def merge_translated_datasets():
    logger.info("Carregando datasets traduzidos...")
    
    reddit_df = pd.read_csv(DATA_DIR / "suicidal_ideation_reddit_pt.csv")
    twitter_df = pd.read_csv(DATA_DIR / "suicidal_ideation_twitter_pt.csv")
    
    logger.info(f"Reddit: {len(reddit_df)} registros")
    logger.info(f"Twitter: {len(twitter_df)} registros")
    
    reddit_processed = pd.DataFrame({
        'text': reddit_df['usertext_pt'],
        'label': reddit_df['label'].astype(int),
        'source': 'reddit'
    })
    
    twitter_processed = pd.DataFrame({
        'text': twitter_df['Tweet_pt'],
        'label': twitter_df['Suicide'].apply(
            lambda x: 1 if 'Potential Suicide post' in str(x) else 0
        ),
        'source': 'twitter'
    })
    
    merged_df = pd.concat([reddit_processed, twitter_processed], ignore_index=True)
    
    merged_df = merged_df[merged_df['text'].notna() & (merged_df['text'] != "")]
    
    output_path = DATA_DIR / "merged_dataset_pt.csv"
    merged_df.to_csv(output_path, index=False)
    
    logger.info(f"Dataset mesclado em português criado com sucesso!")
    logger.info(f"Total de registros: {len(merged_df)}")
    logger.info(f"Reddit: {len(reddit_processed)} registros")
    logger.info(f"Twitter: {len(twitter_processed)} registros")
    logger.info(f"\nDistribuição de labels:")
    logger.info(merged_df['label'].value_counts())
    logger.info(f"\nDistribuição por fonte:")
    logger.info(merged_df['source'].value_counts())
    logger.info(f"\nArquivo salvo em: {output_path}")
    
    return merged_df

if __name__ == "__main__":
    merge_translated_datasets()

