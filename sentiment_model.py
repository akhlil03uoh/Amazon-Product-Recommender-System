import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from scripts.preprocessing import clean_text, ensure_nltk

def label_from_rating(r):
    try:
        r = float(r)
    except Exception:
        return 'neutral'
    if r >= 4:
        return 'positive'
    elif r == 3:
        return 'neutral'
    else:
        return 'negative'

def train_sentiment_model(df):
    ensure_nltk()
    df = df.copy()
    # try several possible text/rating column names
    text_cols = ['reviews.text','reviewText','review_text','text','review']
    rating_cols = ['reviews.rating','reviews.rating.1','rating','overall','reviews.rating']
    # find first existing
    text_col = next((c for c in text_cols if c in df.columns), None)
    rating_col = next((c for c in rating_cols if c in df.columns), None)
    if text_col is None:
        raise KeyError('No review text column found. Expected one of: ' + str(text_cols))
    if rating_col is None:
        print('Warning: no rating column found. Defaulting all to neutral.')
        df['__rating__'] = 3
        rating_col = '__rating__'
    df['clean'] = df[text_col].fillna('').apply(clean_text)
    df['label'] = df[rating_col].apply(label_from_rating)
    X = df['clean']
    y = df['label']
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_vec = tfidf.fit_transform(X)
    test_size = 0.2
    try:
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=test_size, random_state=42, stratify=y)
    except Exception:
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=test_size, random_state=42)
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('Classification Report:\n', classification_report(y_test, y_pred, zero_division=0))
    return tfidf, clf
