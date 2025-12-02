import pandas as pd

def merge_datasets():
    reddit_df = pd.read_csv("data/suicidal_ideation_reddit.csv")
    
    twitter_df = pd.read_csv("data/suicidal_ideation_twitter.csv")
    
    reddit_processed = pd.DataFrame({
        'text': reddit_df['usertext'],
        'label': reddit_df['label'].astype(int),
        'source': 'reddit'
    })
    
    twitter_processed = pd.DataFrame({
        'text': twitter_df['Tweet'],
        'label': twitter_df['Suicide'].apply(
            lambda x: 1 if 'Potential Suicide post' in str(x) else 0
        ),
        'source': 'twitter'
    })
    
    merged_df = pd.concat([reddit_processed, twitter_processed], ignore_index=True)
    
    merged_df.to_csv("data/merged_dataset.csv", index=False)
    
    print(f"Dataset mesclado criado com sucesso!")
    print(f"Total de registros: {len(merged_df)}")
    print(f"Reddit: {len(reddit_processed)} registros")
    print(f"Twitter: {len(twitter_processed)} registros")
    print(f"\nDistribuição de labels:")
    print(merged_df['label'].value_counts())
    print(f"\nDistribuição por fonte:")
    print(merged_df['source'].value_counts())
    
    return merged_df

if __name__ == "__main__":
    merge_datasets()

