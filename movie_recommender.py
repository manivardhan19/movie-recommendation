import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")
movies["genres"] = movies["genres"].fillna("")

vectorizer = CountVectorizer(tokenizer=lambda x: x.split())
genre_matrix = vectorizer.fit_transform(movies["genres"])

cosine_sim = cosine_similarity(genre_matrix, genre_matrix)

def recommend_movies(title, num_recommendations=5):
    indices = pd.Series(movies.index, index=movies['title'].str.lower())
    title_lower = title.lower()
    if title_lower not in indices:
        return f"Movie '{title}' not found in the database."

    idx = indices[title_lower]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

if __name__ == "__main__":
    while True:
        movie_name = input("Enter a Telugu movie title (or 'exit' to quit): ")
        if movie_name.lower() == 'exit':
            break
        recommendations = recommend_movies(movie_name, 5)
        print("Recommended movies:", recommendations)
