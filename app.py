import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# Load and prepare data
# ---------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/rashida048/Datasets/master/movie_dataset.csv"
    df = pd.read_csv(url)

    features = ['keywords', 'cast', 'genres', 'director']
    for feature in features:
        df[feature] = df[feature].fillna('')

    df["combined_features"] = df.apply(
        lambda row: row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director'],
        axis=1
    )

    return df

@st.cache_resource
def create_similarity(df):
    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(df["combined_features"])
    return cosine_similarity(feature_matrix)

df = load_data()
cosine_sim = create_similarity(df)

# ---------------------------
# Recommendation function
# ---------------------------
def recommend_movies(title):
    if title not in df['title'].values:
        return []

    idx = df[df.title == title].index[0]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    sorted_movies = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    return [df.iloc[i[0]].title for i in sorted_movies[1:11]]

# ---------------------------
# UI Design
# ---------------------------
st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommender System")
st.write("Get similar movie suggestions instantly!")

# Dropdown instead of typing (better UX)
movie_list = df['title'].sort_values().unique()
selected_movie = st.selectbox("Choose a movie:", movie_list)

# Button
if st.button("Recommend"):
    recommendations = recommend_movies(selected_movie)

    if recommendations:
        st.subheader("Recommended Movies:")
        for i, movie in enumerate(recommendations, 1):
            st.write(f"{i}. {movie}")
    else:
        st.error("Movie not found!")