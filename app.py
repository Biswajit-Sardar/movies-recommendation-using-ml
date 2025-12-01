from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import requests  


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

app = Flask(__name__)
API_KEY = "da6fad4a"  

print("Loading dataset & Training model... Please wait.")
try:
    df = pd.read_csv("movies.csv")
    
 
    df = df[['title', 'overview', 'genres']].dropna().reset_index(drop=True)
    df['combined'] = df['genres'] + ' ' + df['overview']

    def preprocess_text(text):
        text = str(text).lower()
        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        return " ".join(tokens)

    df['cleaned_text'] = df['combined'].apply(preprocess_text)


    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['cleaned_text'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    print("Model Ready! System is online.")

except Exception as e:
    print(f"Error: {e}")


def fetch_poster(movie_title):
    try:

        url = f"http://www.omdbapi.com/?t={movie_title}&apikey={API_KEY}"
        response = requests.get(url, timeout=2)
        data = response.json()
        
        if data.get('Response') == 'True' and data.get('Poster') != 'N/A':
            return data.get('Poster')
    except:
        pass
  
    return "https://via.placeholder.com/300x450?text=No+Image+Available"

def get_recommendations(movie_name, top_n=5):
 
    matches = df[df['title'].str.lower() == movie_name.lower()]
    
    if len(matches) == 0:
        return []

    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
  
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    
    results = []
  
    for i in movie_indices:
        title = df['title'].iloc[i]
        poster_url = fetch_poster(title) 
        
        results.append({
            "title": title,
            "combined_info": df['combined'].iloc[i],
            "poster": poster_url
        })
    return results

# 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie = data.get('movie_name')
    if not movie:
        return jsonify([])
        
    recommendations = get_recommendations(movie)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)