
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Pastikan nltk data seperti stopwords dan wordnet diunduh saat module di-import
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)


class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stop_words.update({'product', 'item', 'buy', 'set', 'pack', 'new', 'original'}) # Custom stop words
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text):
        if pd.isna(text) or str(text).strip() == '': return ''
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return ' '.join(tokens)


class HybridRetailRecommender:
    def __init__(self, alpha=0.8, beta=0.2): 
        self.alpha = alpha
        self.beta = beta
        self.df = None
        self.tfidf_matrix = None
        self.similarity_matrix = None 
        self.vectorizer = None
        self.name_to_idx = {}
       

    def fit(self, df, tfidf_model_path=None, similarity_path=None): 
        self.df = df
        if self.df is not None and 'products' in self.df.columns:
            for idx, name in enumerate(self.df['products']):
                self.name_to_idx.setdefault(str(name).lower().strip(), []).append(idx)
    

    def _find_query_index(self, query):
        q = str(query).lower().strip()
        if q in self.name_to_idx: return self.name_to_idx[q][0]

        local_cleaner = TextCleaner() # Menggunakan instance lokal dari TextCleaner
        cleaned_q = local_cleaner.clean(query)
        if not cleaned_q: return None

        if self.vectorizer is None:
            print("Error: TF-IDF vectorizer not loaded in recommender.")
            return None

        q_vec = self.vectorizer.transform([cleaned_q])
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        return np.argmax(sims) if sims.max() > 0 else None

    def recommend(self, query, top_n=10, min_sim=0.1, current_customer_id=None): 
        idx = self._find_query_index(query)
        if idx is None:
            print(f"Produk '{query}' tidak ditemukan.")
            return pd.DataFrame()

        content_sims = self.similarity_matrix[idx]
        pop_scores = self.df['popularity_score'].values
        
        
        
        # Pastikan bobot alpha, beta berjumlah 1 atau disesuaikan
        total_weight = self.alpha + self.beta
        if total_weight == 0: total_weight = 1 

        normalized_alpha = self.alpha / total_weight
        normalized_beta = self.beta / total_weight

        hybrid_scores = (normalized_alpha * content_sims) + \
                        (normalized_beta * pop_scores)

        temp_df = pd.DataFrame({
            'Hybrid_Score': hybrid_scores,
            'Content_Sim': content_sims,
            'Popularity': pop_scores,
            'Rating_Individu': self.df['Ratings'].values,
            'Index': np.arange(len(self.df))
        })

        temp_df = temp_df[temp_df['Index'] != idx]
        temp_df = temp_df[temp_df['Content_Sim'] >= min_sim]
        temp_df = temp_df.sort_values(by=['Hybrid_Score', 'Rating_Individu'], ascending=[False, False])

        top_results = temp_df.head(top_n)

        recs = []
        for i, row in top_results.iterrows():
            sim_idx = int(row['Index'])
            recs.append({
                'Product_Name': self.df.iloc[sim_idx]['products'],
                'Hybrid_Score': round(float(row['Hybrid_Score']), 4),
                'Popularity_Score': round(float(row['Popularity']), 4),
                'Category': self.df.iloc[sim_idx]['Product_Category'],
                'Brand': self.df.iloc[sim_idx]['Product_Brand'],
                'Price': f"{int(self.df.iloc[sim_idx]['Amount'] / 10000000):,.0f}.000".replace(',', '.'),
                'Rating': self.df.iloc[sim_idx]['Ratings'],
                'Total_Purchases': self.df.iloc[sim_idx]['Total_Purchases'],
                'Address': self.df.iloc[sim_idx]['Address'],
                'City': self.df.iloc[sim_idx]['City'],
                'State': self.df.iloc[sim_idx]['State'],
                'Country': self.df.iloc[sim_idx]['Country'],
                'Zipcode': self.df.iloc[sim_idx]['Zipcode'],
                'Gender': self.df.iloc[sim_idx]['Gender'],
                'Age': self.df.iloc[sim_idx]['Age'],
                'Internal_Index': sim_idx # Add internal index for better filtering
            })
        return pd.DataFrame(recs)

    def save_recommender(self, path='hybrid_retail_recommender.joblib'):
        print("This method is for saving, not needed when loading environment.")
