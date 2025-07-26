import streamlit as st
import pandas as pd
import re
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, LangDetectException
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import pyLDAvis.gensim as gensimvis


warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('wordnet')
import streamlit as st
import base64


# To set the background color of the home page, use CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFA500; /**/
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add image to sidebar
st.sidebar.image(
    "/Users/ecemzeynepiscanli/PycharmProjects/SMA_STREAMLIT/amazon_logo.jpg",
    use_column_width=True
)



# Changing the sidebar color with CSS
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Page selection

st.markdown("""
    <style>
    [data-testid="stSidebar"] h1 {
        background: linear-gradient(to right, black, orange);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Recommender Options")

page = st.sidebar.radio("Choose a method:", ["Content-Based Recommender", "Personalised Recommender System"])

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("bookdata.csv")
    return df

df = load_data()

# Visualization settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
df.head()

# Page 1: Content-Based Recommender
if page == "Content-Based Recommender":

    st.title("📘 Content-Based Book Recommender")

    # Prepare unique books and lowercase key text fields
    df_unique = df[['Title', 'description', 'categories']].drop_duplicates(subset='Title').reset_index(drop=True)
    df_unique['Title'] = df_unique['Title'].str.lower()
    df_unique['description'] = df_unique['description'].fillna('').str.lower()
    df_unique['categories'] = df_unique['categories'].astype(str).str.lower()

    # Combine description and category
    df_unique['combined'] = df_unique['description'] + " " + df_unique['categories']

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df_unique['combined'])

    # Cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Recommendation function
    def content_based_recommender(title, cosine_sim, dataframe):
        indices = pd.Series(dataframe.index, index=dataframe['Title'])
        indices = indices[~indices.index.duplicated(keep='last')]
        if title not in indices:
            return None, f"'{title}' not found. Please check the spelling."
        book_index = indices[title]
        similarity_scores = pd.DataFrame(cosine_sim[book_index], columns=["score"])
        book_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
        results = dataframe.iloc[book_indices][['Title', 'description']].copy()
        results['Similarity Score'] = similarity_scores.iloc[book_indices].values
        results = results[['Title', 'Similarity Score', 'description']]
        results.rename(columns={"Title": "Title", "description": "Description"}, inplace=True)
        return results.reset_index(drop=True), None

    # User input
    title_input = st.text_input("Enter a book title to get similar recommendations:")

    if title_input:
        recommendations, error = content_based_recommender(title_input.lower(), cosine_sim, df_unique)

        if error:
            st.warning(error)
        else:
            st.success(f"Top recommendations similar to '{title_input.title()}':")

            # Show each result in a clean card-like layout
            for i, row in recommendations.iterrows():
                st.markdown(f"### 📖 {row['Title'].title()}")
                st.markdown(f"**Similarity Score:** {round(row['Similarity Score'], 3)}")

                # Limit description to first 100 words
                full_desc = row["Description"]
                short_desc = " ".join(full_desc.split()[:50])

                # If longer than 100 words, show expandable
                if len(full_desc.split()) > 100:
                    st.markdown(short_desc + "...")
                    with st.expander("Show full description"):
                        st.write(full_desc)
                else:
                    st.markdown(full_desc)

                st.markdown("---")  # line separator between results


