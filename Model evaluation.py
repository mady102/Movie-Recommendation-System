import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("tmdb_with_moods.csv")

# Preprocess
df['overview'] = df['overview'].fillna('')
df['genres'] = df['genres'].fillna('')
df['release_date'] = df['release_date'].fillna('')
df['mood'] = df['mood'].fillna('')
df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
df['combined_features'] = df['overview'] + " " + df['genres'] + " " + df['mood']

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_features'])

# KNN Model Training
def train_knn_model(df, n_neighbors=5):
    # Splitting data into features and target
    X = tfidf_matrix
    y = df['mood']  # Assuming mood is a classification target
    
    # Splitting data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize KNN
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = knn.predict(X_test)
    
    # Model evaluation
    print("Model Evaluation Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    return knn

# Train the KNN model
knn_model = train_knn_model(df)

# Movie recommendation function using KNN
def recommend_movies_knn(input_year, input_mood, top_n=10):
    # Get mood prediction for the input mood
    input_vec = vectorizer.transform([input_mood])
    mood_prediction = knn_model.predict(input_vec)

    print(f"Predicted Mood for Input: {mood_prediction[0]}")
    
    # Find movies with the predicted mood
    filtered_df = df[df['year'] == input_year]
    if filtered_df.empty:
        print(f"😕 No movies found in {input_year}. Looking for nearby years...")

        # Try previous year
        prev_df = df[df['year'] == input_year - 1]
        next_df = df[df['year'] == input_year + 1]
        extended_df = pd.concat([prev_df, next_df])

        if extended_df.empty:
            return "❌ No movies found in nearby years either."

        print(f"✅ Found {len(extended_df)} movies in year {input_year - 1} or {input_year + 1}")
        top_movies = extended_df[extended_df['mood'] == mood_prediction[0]].head(top_n)
        return top_movies[['title', 'genres', 'year', 'overview']].reset_index(drop=True)

    # Found in exact year
    year_indices = filtered_df.index
    filtered_df['cosine_similarity'] = cosine_similarity(input_vec, tfidf_matrix[year_indices]).flatten()
    top_movies = filtered_df[filtered_df['mood'] == mood_prediction[0]].sort_values(by='cosine_similarity', ascending=False).head(top_n)
    return top_movies[['title', 'genres', 'year', 'overview']].reset_index(drop=True)

# 🌟 Get user input
try:
    year_input = int(input("📅 Enter the year of movie you want to watch: "))
    mood_input = input("🧠 Enter your current mood (e.g., happy, romantic, thrilling): ")

    print("\n🔍 Recommending movies based on your mood and year...\n")
    results = recommend_movies_knn(year_input, mood_input)

    if isinstance(results, str):
        print(results)
    else:
        for idx, row in results.iterrows():
            print(f"{idx+1}. {row['title']} ({row['year']})")
            print(f"   Genre: {row['genres']}")
            print(f"   Overview: {row['overview'][:250]}...\n")

except ValueError:
    print("⚠️ Please enter a valid year.")

