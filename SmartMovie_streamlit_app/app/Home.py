"""Page d'accueil — lancer avec : streamlit run app/Home.py"""
import streamlit as st

from utils import load_movies, show_source_status

st.set_page_config(page_title="SmartMovie", page_icon="🎬", layout="wide")

df, source = load_movies()
show_source_status(source)

st.title("🎬 SmartMovie")
st.markdown(
    "Application d'analyse et de recommandation de films, construite sur les données "
    "**TMDB** (1000 films les plus populaires) stockées dans **MongoDB**."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Films", f"{len(df):,}")
c2.metric("Note moyenne", f"{df['vote_average'].mean():.2f}")
c3.metric("Durée médiane", f"{df['runtime'].median():.0f} min")
c4.metric("Période", f"{int(df['release_year'].min())} – {int(df['release_year'].max())}")

st.divider()
st.subheader("Navigation (menu de gauche)")
st.markdown(
    """
| Page | Contenu |
|---|---|
| **Dashboard** | Distributions, genres, années, durée, budget/revenus, votes/popularité, boxplots et corrélations (avec filtres) |
| **Classification** | Prédiction de `high_engagement` : comparaison Logistic Regression / Random Forest / Linear SVM, et test sur un film fictif |
| **Clusters** | K-Means sur variables structurées, TF-IDF ou les deux : choix de K par Silhouette Score, visualisation 2D, interprétation |
| **Recommandations** | Films similaires (TF-IDF + similarité cosinus), ou recherche à partir d'une description libre |
"""
)

with st.expander("Comment les données sont traitées"):
    st.markdown(
        """
- **Source** : MongoDB (`smartmovie.movies`). Si la connexion échoue, l'application se replie sur `data/raw.json`.
- **Nettoyage** : les `0` de `budget`, `revenue` et `runtime` sont traités comme des valeurs manquantes.
- **Feature engineering** : année, mois, décennie, saisonnalité, nombre de genres/mots-clés, catégories de durée et de budget (identique à l'Étape 4).
- **Anti-leakage** : la classification n'utilise que des variables connues *avant* la sortie. `vote_count` (qui définit la cible), `vote_average`, `popularity` et `revenue` sont exclus des features.
"""
    )
