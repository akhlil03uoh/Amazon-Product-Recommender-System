import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scripts.preprocessing import clean_text

class ItemRecommender:
    def __init__(self):
        self.tfidf = None
        self.product_ids = None
        self.matrix = None

    def fit(self, df):
        df = df.copy()
        # possible asin/product id columns
        asin_cols = ['asins','asin','product_id','productId','sku']
        asin_col = next((c for c in asin_cols if c in df.columns), None)
        text_cols = ['reviews.text','reviewText','review_text','text','review']
        text_col = next((c for c in text_cols if c in df.columns), None)
        if asin_col is None or text_col is None:
            raise KeyError('asin or review text column not found.')
        grouped = df.groupby(asin_col)[text_col].apply(lambda texts: ' '.join(texts.fillna(''))).reset_index()
        grouped['clean'] = grouped[text_col].apply(clean_text)
        self.product_ids = grouped[asin_col].tolist()
        self.tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        self.matrix = self.tfidf.fit_transform(grouped['clean'])

    def recommend(self, asin, topk=5):
        if self.matrix is None:
            raise ValueError('Model not fitted. Call fit() first.')
        if asin not in self.product_ids:
            return []
        idx = self.product_ids.index(asin)
        sims = cosine_similarity(self.matrix[idx], self.matrix).flatten()
        sims[idx] = -1
        top_idx = sims.argsort()[::-1][:topk]
        return [(self.product_ids[i], float(sims[i])) for i in top_idx]
