
# Import semua library yang dibutuhkan
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import re
import nltk

# Import classes from recommender_core.py
from recommender_core import TextCleaner, HybridRetailRecommender

# Pastikan nltk data seperti stopwords dan wordnet diunduh di lingkungan baru
for res in ['stopwords', 'wordnet', 'omw-1.4']:
    try: nltk.data.find(f'corpora/{res}')
    except LookupError: nltk.download(res, quiet=True)

# -- Mulai Loading --
print("Memuat recommender_environment.pkl...")
loaded_env = joblib.load('recommender_environment.pkl')

# Ekstrak komponen dari environment yang dimuat
loaded_df_clean = loaded_env['df_clean']
loaded_tfidf_matrix = loaded_env['tfidf_matrix']
loaded_vectorizer = loaded_env['vectorizer']
loaded_similarity_matrix = loaded_env['similarity_matrix']
loaded_recommender_object = loaded_env['recommender_object']
loaded_text_cleaner = loaded_env['text_cleaner']

# Penting: Setelah memuat, pastikan recommender_object memiliki referensi ke
# df, tfidf_matrix, vectorizer, dan similarity_matrix yang dimuat.
# Ini penting karena hanya instance objek yang di-pickle, bukan data frame besar yang terkait.
loaded_recommender_object.df = loaded_df_clean
loaded_recommender_object.tfidf_matrix = loaded_tfidf_matrix
loaded_recommender_object.vectorizer = loaded_vectorizer
loaded_recommender_object.similarity_matrix = loaded_similarity_matrix

print("Recommender environment berhasil dimuat dan siap digunakan.")

# Input produk yang dicari oleh pengguna
query_new_notebook = input("Masukkan nama produk yang ingin Anda cari: ")

print(f"\nMelakukan rekomendasi untuk: '{query_new_notebook}'")
recommendations_new_notebook = loaded_recommender_object.recommend(query_new_notebook, top_n=7)

if not recommendations_new_notebook.empty:
    print("Berikut adalah 7 produk rekomendasi:")
    print(recommendations_new_notebook.to_string())
else:
    print(f"Maaf, tidak ada rekomendasi yang ditemukan untuk produk '{query_new_notebook}'.")

print("\nApakah Anda ingin menampilkan produk terkait?")
user_choice = input("Ketik 'Ya' atau 'Tidak': ").strip().lower()

if user_choice == 'ya':
    print(f"\nMenampilkan 5 produk terkait untuk '{query_new_notebook}':")

    # Get the internal indices from the initial recommendations
    initial_recommended_indices = []
    initial_top_product_name = None
    if not recommendations_new_notebook.empty:
        initial_recommended_indices = recommendations_new_notebook['Internal_Index'].tolist()
        # Get the name of the top recommended product for stricter filtering
        initial_top_product_name = recommendations_new_notebook.iloc[0]['Product_Name']

    # Get a larger set of recommendations to find distinct related products
    # Increase top_n to get more candidates for related products
    all_related_candidates = loaded_recommender_object.recommend(query_new_notebook, top_n=100) # Fetch more candidates

    if not all_related_candidates.empty:
        # Filter out products that were already in the initial top 7 recommendations by Internal_Index
        filtered_related_products = all_related_candidates[
            ~all_related_candidates['Internal_Index'].isin(initial_recommended_indices)
        ]

        # Add an additional filter: exclude products with the same name as the top initial recommendation
        # This helps ensure diversity in 'related products'
        if initial_top_product_name:
            filtered_related_products = filtered_related_products[
                filtered_related_products['Product_Name'] != initial_top_product_name
            ]

        # Now take the top 5 from this more strictly filtered list
        filtered_related_products = filtered_related_products.head(5)

        if not filtered_related_products.empty:
            print(filtered_related_products.to_string())
        else:
            print("Tidak ada produk terkait yang berbeda ditemukan dari daftar yang lebih besar setelah memperhitungkan nama produk yang sama.")
    else:
        print("Tidak ada produk terkait yang ditemukan sama sekali.")
elif user_choice == 'tidak':
    print("Selesai.")
else:
    print("Pilihan tidak valid. Selesai.")
