"""Recommandations de films basées sur le contenu (TF-IDF + similarité cosinus)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    build_recommender,
    load_movies,
    recommend_from_text,
    recommend_similar,
    show_source_status,
)

st.set_page_config(page_title="Recommandations · SmartMovie", page_icon="🍿", layout="wide")

df, source = load_movies()
show_source_status(source)
rec = build_recommender(df, len(df))

st.title("🍿 Recommandations")
st.markdown(
    "Moteur **basé sur le contenu** : chaque film est décrit par son synopsis, ses genres (pondérés ×2) "
    "et ses mots-clés, vectorisés en **TF-IDF**. Deux films sont proches si leurs vecteurs ont un "
    "**cosinus** élevé (1 = vocabulaire identique, 0 = rien en commun)."
)

min_rating = st.sidebar.slider("Note minimale des films recommandés", 0.0, 10.0, 0.0, 0.5)
n_results = st.sidebar.slider("Nombre de recommandations", 3, 20, 8)


def label_for(i: int) -> str:
    year = df.loc[i, "release_year"]
    year = f"{int(year)}" if pd.notna(year) else "?"
    return f"{df.loc[i, 'title']} ({year})"


def show_results(indices: np.ndarray, sims: np.ndarray) -> None:
    """Affiche les recommandations (filtrées par note minimale)."""
    keep = [(i, s) for i, s in zip(indices, sims) if df.loc[i, "vote_average"] >= min_rating][:n_results]
    if not keep:
        st.warning("Aucun résultat avec ces critères. Baisse la note minimale ou reformule la recherche.")
        return

    table = pd.DataFrame({
        "Film": [label_for(i) for i, _ in keep],
        "Similarité": [s for _, s in keep],
        "Genres": [df.loc[i, "genres"] for i, _ in keep],
        "Note": [df.loc[i, "vote_average"] for i, _ in keep],
    })
    st.dataframe(
        table,
        column_config={
            "Similarité": st.column_config.ProgressColumn("Similarité", min_value=0.0, max_value=1.0, format="%.2f"),
            "Note": st.column_config.NumberColumn("Note", format="%.1f"),
        },
        hide_index=True,
    )
    for i, s in keep:
        with st.expander(f"{label_for(i)} — similarité {s:.2f}"):
            st.write(df.loc[i, "overview"] or "Synopsis non disponible.")
            st.caption(f"Genres : {df.loc[i, 'genres']} · Mots-clés : {df.loc[i, 'keywords'][:150]}")


tab_similar, tab_text = st.tabs(["Films similaires à…", "Décris le film que tu cherches"])

# ---------------------------------------------------------------- Films similaires
with tab_similar:
    labels = [label_for(i) for i in df.index]
    dup = pd.Series(labels).duplicated(keep=False)
    labels = [f"{lab} #{df.loc[i, 'movie_id']}" if d else lab for i, (lab, d) in enumerate(zip(labels, dup))]
    label_to_idx = dict(zip(labels, df.index))

    chosen = st.selectbox("Choisis un film (tape pour rechercher)", sorted(labels), index=None,
                          placeholder="Ex. : un titre que tu as aimé")
    if chosen:
        idx = label_to_idx[chosen]
        with st.expander("Film choisi", expanded=True):
            st.write(df.loc[idx, "overview"] or "Synopsis non disponible.")
            st.caption(f"Genres : {df.loc[idx, 'genres']} · Note : {df.loc[idx, 'vote_average']:.1f}")
        # On récupère plus de candidats que nécessaire pour absorber le filtre de note
        indices, sims = recommend_similar(rec, idx, n=len(df) - 1)
        show_results(indices, sims)
        st.info(
            "**Lecture.** Les similarités restent souvent modestes (< 0.3) : un synopsis court partage peu de mots "
            "avec un autre, même du même genre. Ce qui compte est le **classement**, pas la valeur absolue."
        )

# ---------------------------------------------------------------- Recherche libre
with tab_text:
    query = st.text_area(
        "Décris le film (en anglais, comme les synopsis TMDB)",
        placeholder="a detective investigates a murder in a dark city",
        height=100,
    )
    if st.button("Recommander"):
        if not query.strip():
            st.warning("Écris d'abord une description.")
        else:
            indices, sims = recommend_from_text(rec, query, n=len(df))
            if len(indices) == 0:
                st.warning("Aucun mot de ta description n'est connu du vocabulaire. Essaie avec d'autres termes.")
            else:
                show_results(indices, sims)
