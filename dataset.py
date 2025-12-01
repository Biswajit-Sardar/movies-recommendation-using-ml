import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# loading the dataset to a pandas dataframe
df = pd.read_csv("movies.csv")
df.info()



#filter the required columns
required_columns=['title','overview','genres']
df = df[required_columns] 


df=df.dropna().reset_index(drop=True)

df['combined'] =  df['genres'] + ' ' + df['overview'] 

data=df[['title','combined']]

combined_text = " ".join(df['combined'])


WordCloud=WordCloud(width=800, height=400, background_color='white').generate(combined_text)


#visualizetion
plt.figure(figsize=(10,5))
plt.imshow(WordCloud, interpolation='bilinear')
plt.axis('off')
plt.title('most common movie content in the world')
plt.show()



# download nltk data
import nltk

nltk.download('punkt')
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)



# Apply preprocessing to the movie content
data['cleaned_text'] = df['combined'].apply(preprocess_text)


# Vectorization with TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['cleaned_text'])




# Compute Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(cosine_sim[0])



# Recommendation Function
def recommend_movies(movie_name, cosine_sim=cosine_sim, df=data, top_n=5):
    # Find the index of the movie
    idx = df[df['title'].str.lower() == movie_name.lower()].index
    if len(idx) == 0:
        return "Movie not found in the dataset!"
    idx = idx[0]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return top n similar movies
    return df[['title']].iloc[movie_indices]

# row_index = df[df['title'] == "Avengers: Age of Ultron"].index
row_index = df[df['title'] == "Batman v Superman: Dawn of Justice"].index
print(row_index)

movie_name = data["title"][10]
print(movie_name)

# Example Recommendation

print(f"Recommendations for the Movie {movie_name}")
recommendations = recommend_movies(movie_name)
print(recommendations)