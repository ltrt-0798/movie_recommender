import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Movie Recommendation System")

# ---------------------------
# LOAD DATA (SAFE + CACHED)
# ---------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("movie_dataset.csv")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df = load_data()

if df is None:
    st.stop()

st.success("Data loaded successfully ✅")

# ---------------------------
# BUILD MODEL (CACHED)
# ---------------------------
@st.cache_resource
def create_model(df):
    df = df.copy()

    features = ['keywords','cast','genres','director']
    for feature in features:
        if feature in df.columns:
            df[feature] = df[feature].fillna('')
        else:
            st.error(f"Missing column: {feature}")
            st.stop()

    df["combined_features"] = df.apply(
        lambda row: row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director'],
        axis=1
    )

    df["combined_features"] = df["combined_features"].str.lower()

    vectorizer = TfidfVectorizer()
    feature_matrix = vectorizer.fit_transform(df["combined_features"])
    similarity = cosine_similarity(feature_matrix)

    return similarity

similarity = create_model(df)

st.success("Model ready ✅")

# ---------------------------
# POSTER FUNCTION (SAFE)
# ---------------------------
API_KEY = "2dbd05c4e48926f61a8482f6afb3e6d3"  

def fetch_poster(movie_name):
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}"
        data = requests.get(url).json()

        if "results" in data and len(data["results"]) > 0:
            poster_path = data["results"][0].get("poster_path")
            if poster_path:
                return "https://image.tmdb.org/t/p/w500/" + poster_path
    except:
        pass

    return "https://via.placeholder.com/150?text=No+Image"

# ---------------------------
# RECOMMEND FUNCTION
# ---------------------------
def recommend(movie_name):
    if movie_name not in df['title'].values:
        return [], []

    movie_index = df[df['title'] == movie_name].index[0]
    similar_movies = list(enumerate(similarity[movie_index]))

    sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:11]

    names = []
    posters = []

    for i in sorted_movies:
        title = df.iloc[i[0]].title
        names.append(title)
        posters.append(fetch_poster(title))

    return names, posters

# ---------------------------
# UI
# ---------------------------
movie_list = sorted(df['title'].values)

selected_movie = st.selectbox("🔍 Search or select a movie:", movie_list)

if st.button("🎯 Recommend"):
    st.write(f"### Showing results for: **{selected_movie}**")

    with st.spinner("Finding best movies..."):
        names, posters = recommend(selected_movie)

    if names:
        st.subheader("Top Recommendations")

        cols = st.columns(5)

        for i in range(len(names)):
            with cols[i]:
                st.image(posters[i])
                st.markdown(f"**{names[i]}**")
    else:
        st.error("Movie not found!")
  

