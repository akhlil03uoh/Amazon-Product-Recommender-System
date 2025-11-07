# Amazon Recommender - Using Your Uploaded Dataset

This project uses your uploaded Amazon dataset (placed at `data/sample.csv`) and runs:
- Preprocessing (NLTK)
- Naive Bayes sentiment classification (trained on review text and ratings)
- Item-based TF-IDF + cosine similarity recommender

**Important:** This version uses your dataset's columns: `reviews.text`, `reviews.rating`, `asins`, `reviews.username`, `reviews.title`.

## How to run (recommended)
1. Create and activate a Python virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Run the demo script (this will use the full dataset you uploaded):
   ```bash
   python demo_run_userdata.py
   ```
3. Or open and run `notebooks/enhanced_userdata.ipynb` in Jupyter.
