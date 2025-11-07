import sys, os
sys.path.append(os.path.abspath('./'))
import pandas as pd
from scripts.preprocessing import ensure_nltk
from scripts.sentiment_model import train_sentiment_model
from scripts.recommender import ItemRecommender

def main():
    ensure_nltk()
    df = pd.read_csv('data/sample.csv', low_memory=False)
    print('Columns found:', df.columns.tolist())
    print('Training sentiment model...')
    tfidf, clf = train_sentiment_model(df)
    print('\nFitting recommender...')
    rec = ItemRecommender()
    rec.fit(df)
    asin_col = next((c for c in ['asins','asin','product_id','productId','sku'] if c in df.columns), 'asins')
    example_asin = df[asin_col].iloc[0]
    print('Top recommendations for', example_asin, ':', rec.recommend(example_asin, topk=5))

if __name__ == '__main__':
    main()
