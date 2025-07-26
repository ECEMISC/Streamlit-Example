import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Arayüz ayarları
st.markdown("""
    <style>
    .stApp { background-color: #FFA500; }
    [data-testid="stSidebar"] { background-color: white; }
    [data-testid="stSidebar"] h1 {
        background: linear-gradient(to right, black, orange);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar görsel
st.sidebar.image(
    "https://raw.githubusercontent.com/ECEMISC/Streamlit-Example/main/amazon_logo.JPG",
    use_column_width=True
)

st.sidebar.title("Recommender Options")
page = st.sidebar.radio("Choose a method:", ["Content-Based Recommender"])

# Veri yükleme
@st.cache_data
def load_data():
    return pd.read_csv("bookdata.csv")

df = load_data()

# Content-Based Recommender
if page == "Content-Based Recommender":
    st.title("📘 Content-Based Book Recommender")

    df_unique = df[['Title', 'description', 'categories']].drop_duplicates(subset='Title').reset_index(drop=True)
    df_unique['Title'] = df_unique['Title'].str.lower()
    df_unique['description'] = df_unique['description'].fillna('').str.lower()
    df_unique['categories'] = df_unique['categories'].astype(str).str.lower()
    df_unique['combined'] = df_unique['description'] + " " + df_unique['categories']

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df_unique['combined'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def content_based_recommender(title, cosine_sim, dataframe):
        indices = pd.Series(dataframe.index, index=dataframe['Title'])
        indices = indices[~indices.index.duplicated(keep='last')]
        if title not in indices:
            return None, f"'{title}' not found."
        book_index = indices[title]
        similarity_scores = pd.DataFrame(cosine_sim[book_index], columns=["score"])
        book_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
        results = dataframe.iloc[book_indices][['Title', 'description']].copy()
        results['Similarity Score'] = similarity_scores.iloc[book_indices].values
        return results[['Title', 'Similarity Score', 'description']].reset_index(drop=True), None

    title_input = st.text_input("Enter a book title to get similar recommendations:")

    if title_input:
        recommendations, error = content_based_recommender(title_input.lower(), cosine_sim, df_unique)

        if error:
            st.warning(error)
        else:
            st.success(f"Top recommendations similar to '{title_input.title()}':")
            for _, row in recommendations.iterrows():
                st.markdown(f"### 📖 {row['Title'].title()}")
                st.markdown(f"**Similarity Score:** {round(row['Similarity Score'], 3)}")
                desc = row["description"]
                short = " ".join(desc.split()[:50])
                st.markdown(short + "..." if len(desc.split()) > 100 else desc)
                if len(desc.split()) > 100:
                    with st.expander("Show full description"):
                        st.write(desc)
                st.markdown("---")
