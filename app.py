from flask import Flask, request, jsonify, render_template
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Step 1: Load the data
df = pd.read_csv('playlist.csv', low_memory=False)

# Step 2: Filter out relevant columns
relevant_columns = ['danceability', 'energy', 'key', 'loudness', 'mode',
                    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                    'valence', 'tempo', 'title']
df_filtered = df[relevant_columns].copy()

# Step 3: Drop rows with missing values
df_filtered = df_filtered.dropna()

# Step 4: Reset index
df_filtered.reset_index(drop=True, inplace=True)

# Step 5: Vectorize titles
title_vectorizer = TfidfVectorizer(stop_words='english')
title_matrix = title_vectorizer.fit_transform(df_filtered['title'])

# Step 6: Scale audio features
audio_features = ['danceability', 'energy', 'key', 'loudness', 'mode',
                  'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                  'valence', 'tempo']
scaler = StandardScaler()
df_filtered[audio_features] = scaler.fit_transform(df_filtered[audio_features])

# Step 7: Combine title vector and audio features
X_combined = scipy.sparse.hstack([title_matrix, df_filtered[audio_features]])

# Step 8: Train a KNN model
knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
knn_model.fit(X_combined)

# Step 9: Define recommendation function
def recommend_song_titles(title, audio_features):
    try:
        # Vectorize title
        title_vector = title_vectorizer.transform([title])

        # Scale audio features
        audio_features_scaled = scaler.transform([audio_features])

        # Combine title vector and audio features
        combined_input = scipy.sparse.hstack([title_vector, audio_features_scaled])

        # Find nearest neighbors
        distances, indices = knn_model.kneighbors(combined_input)

        # Get recommended song titles
        recommended_titles = df_filtered.iloc[indices[0][1:]]['title'].tolist()

        return recommended_titles
        
    except Exception as e:
        return str(e)

# Route to serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')

# Define the recommendation function
@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        data = request.get_json()
        title = data['title']
        audio_features = data['audio_features']
        
        recommended_titles = recommend_song_titles(title, audio_features)
        
        return jsonify({'recommended_titles': recommended_titles})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
