import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load dataset (with fix for CSV issue)
df = pd.read_csv("tmdb_with_moods_enhanced_cleaned.csv")

# Preprocess
df['overview'] = df['overview'].fillna('')
df['genres'] = df['genres'].fillna('')
df['release_date'] = df['release_date'].fillna('')
df['mood'] = df['mood'].fillna('')
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
df['combined_features'] = df['overview'] + " " + df['genres'] + " " + df['mood']

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# Train K-Means
kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans.fit_predict(tfidf_matrix)

# Recommendation function using K-Means
def recommend_with_kmeans(input_year, input_mood, top_n=10):
    input_vec = vectorizer.transform([input_mood])
    cluster_label = kmeans.predict(input_vec)[0]

    print(f"🎯 Closest cluster to your mood: {cluster_label}")
    
    # Filter by cluster and year
    cluster_df = df[(df['cluster'] == cluster_label) & (df['year'] == input_year)]
    
    if cluster_df.empty:
        print(f"😕 No movies found in {input_year} for this cluster. Searching nearby years...")
        prev_df = df[(df['cluster'] == cluster_label) & (df['year'] == input_year - 1)]
        next_df = df[(df['cluster'] == cluster_label) & (df['year'] == input_year + 1)]
        cluster_df = pd.concat([prev_df, next_df])
    
    if cluster_df.empty:
        return "❌ No similar movies found in nearby years either."

    cluster_df = cluster_df.copy()
    cluster_df['cosine_similarity'] = cosine_similarity(input_vec, tfidf_matrix[cluster_df.index]).flatten()
    top_movies = cluster_df.sort_values(by='cosine_similarity', ascending=False).head(top_n)

    return top_movies[['title', 'genres', 'year', 'overview']].reset_index(drop=True)

# 🌟 Get user input and recommend
try:
    year_input = int(input("📅 Enter the year of movie you want to watch: "))
    mood_input = input("🧠 Enter your current mood (e.g., happy, romantic, thrilling): ")

    print("\n🔍 Recommending movies based on your mood and year...\n")
    results = recommend_with_kmeans(year_input, mood_input)

    if isinstance(results, str):
        print(results)
    else:
        for idx, row in results.iterrows():
            print(f"{idx+1}. {row['title']} ({row['year']})")
            print(f"   Genre: {row['genres']}")
            print(f"   Overview: {row['overview'][:250]}...\n")

except ValueError:
    print("⚠ Please enter a valid year.")
from sklearn.metrics import silhouette_score

# Evaluate clustering performance
sil_score = silhouette_score(tfidf_matrix, df['cluster'])
print(f"📊 Silhouette Score: {sil_score:.4f}")
print(f"🌀 K-Means Inertia: {kmeans.inertia_:.2f}")