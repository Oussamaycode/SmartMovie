"""Clustering K-Means : variables structurées, TF-IDF ou combiné."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from utils import (
    combined_matrix,
    fit_kmeans,
    load_movies,
    project_2d,
    show_source_status,
    silhouette_scan,
    structured_matrix,
    text_matrix,
)

st.set_page_config(page_title="Clusters · SmartMovie", page_icon="🧩", layout="wide")
sns.set_theme(style="whitegrid")

df, source = load_movies()
show_source_status(source)


def show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


st.title("🧩 Clusters (K-Means)")

APPROACHES = {
    "Variables structurées": "struct",
    "TF-IDF (synopsis)": "text",
    "Combiné (structuré + TF-IDF)": "combo",
}
choice = st.radio("Variables utilisées pour le clustering", list(APPROACHES), horizontal=True)
approach = APPROACHES[choice]
key = f"{approach}-{len(df)}"

terms = None
if approach == "struct":
    X, metric, projection = structured_matrix(df, len(df)), "euclidean", "pca"
    st.caption("Budget, durée, saisonnalité, genres/mots-clés (nombre), catégories… standardisés. Distance euclidienne.")
elif approach == "text":
    (X, terms), metric, projection = text_matrix(df, len(df)), "cosine", "svd"
    st.caption("TF-IDF du synopsis (1000 termes, unigrammes + bigrammes). Distance cosinus, plus adaptée au texte.")
else:
    X, metric, projection = combined_matrix(df, len(df)), "euclidean", "pca"
    st.caption("Variables structurées + 20 composantes SVD du TF-IDF, re-standardisées ensemble.")

# ---------------------------------------------------------------- Choix de K
scores = silhouette_scan(X, key, metric)
best_k = max(scores, key=scores.get)

left, right = st.columns([3, 2])
with left:
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(list(scores), list(scores.values()), marker="o", color="#DD8452")
    ax.scatter([best_k], [scores[best_k]], s=160, facecolors="none", edgecolors="red", linewidths=2,
               label=f"Meilleur K = {best_k}")
    ax.set(title="Silhouette Score selon K", xlabel="K (nombre de clusters)", ylabel="Silhouette")
    ax.legend()
    fig.tight_layout()
    show(fig)
with right:
    k = st.slider("Nombre de clusters K", 2, 10, best_k)
    s = scores[k]
    quality = "forte" if s > 0.5 else "raisonnable" if s > 0.25 else "faible"
    st.metric(f"Silhouette (K={k})", f"{s:.3f}")
    st.info(
        f"Structure **{quality}** (>0.5 forte, 0.25–0.5 raisonnable, <0.25 faible). "
        + ("Sur du texte, un score bas est normal : les synopsis forment un continuum thématique plutôt que des groupes nets."
           if approach == "text" else
           "Le K retenu par défaut maximise le silhouette score.")
    )

labels, centers = fit_kmeans(X, key, k)
fdf = df.copy()
fdf["cluster"] = labels
palette = plt.get_cmap("tab10")

# ---------------------------------------------------------------- Projection 2D
coords, explained = project_2d(X, key, projection)
fig, ax = plt.subplots(figsize=(9, 6))
for c in range(k):
    mask = labels == c
    ax.scatter(coords[mask, 0], coords[mask, 1], color=palette(c % 10), alpha=0.65, s=30,
               label=f"Cluster {c} ({mask.sum()})")
name = "PCA" if projection == "pca" else "TruncatedSVD"
ax.set(title=f"Projection 2D ({name}) — variance expliquée : {explained:.0%}",
       xlabel="Composante 1", ylabel="Composante 2")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
fig.tight_layout()
show(fig)
st.caption(
    "La projection 2D sert uniquement à l'affichage : le clustering est fait dans l'espace complet. "
    "Plus la variance expliquée est faible, plus ce graphique est approximatif ; des clusters qui se "
    "chevauchent ici peuvent être bien séparés dans les dimensions non affichées."
)

# ---------------------------------------------------------------- Interprétation
st.subheader("Interprétation des clusters")


def top_genres(sub: pd.DataFrame, n: int = 3) -> str:
    counts = sub["genres_list"].explode().dropna().value_counts()
    return ", ".join(counts.head(n).index) or "n/a"


summary_lines = []
for c in range(k):
    sub = fdf[fdf["cluster"] == c]
    parts = [f"**Cluster {c}** ({len(sub)} films, {len(sub) / len(fdf):.0%})"]
    if sub["budget"].notna().any():
        parts.append(f"budget médian {sub['budget'].median() / 1e6:.0f} M$")
    if sub["runtime"].notna().any():
        parts.append(f"durée médiane {sub['runtime'].median():.0f} min")
    parts.append(f"genres dominants : {top_genres(sub)}")
    if approach == "text":
        top_idx = centers[c].argsort()[::-1][:8]
        parts.append("termes clés : " + ", ".join(terms[top_idx]))
    summary_lines.append("- " + " · ".join(parts))
st.markdown("\n".join(summary_lines))

profile_cols = ["budget", "runtime", "n_genres", "n_keywords", "overview_length",
                "is_summer_release", "is_holiday_release", "is_original_language_en"]
tab_profile, tab_success, tab_box, tab_films = st.tabs(
    ["Profil moyen", "Indicateurs de succès", "Boxplots", "Explorer les films"]
)

with tab_profile:
    profile = fdf.groupby("cluster")[profile_cols].mean()
    profile["budget"] = profile["budget"] / 1e6
    profile = profile.rename(columns={"budget": "budget (M$)"}).round(2)
    st.dataframe(profile.style.background_gradient(cmap="Blues", axis=0))
    st.caption("Moyennes par cluster. Colonnes 0/1 (`is_*`) = part de films concernés. Dégradé de couleur par colonne.")

with tab_success:
    success = fdf.groupby("cluster")[["vote_average", "vote_count", "popularity", "revenue"]].median()
    success["revenue"] = success["revenue"] / 1e6
    success = success.rename(columns={"revenue": "revenue (M$)"}).round(2)
    st.dataframe(success.style.background_gradient(cmap="Greens", axis=0))
    st.caption(
        "Médianes par cluster. Ces indicateurs post-sortie ne sont **pas** utilisés pour former les clusters : "
        "ils servent seulement à décrire a posteriori si un profil de production va de pair avec plus de succès. "
        "(Le clustering est non supervisé : il n'y a pas de cible, donc pas de leakage au sens de la classification.)"
    )

with tab_box:
    var = st.selectbox("Variable", ["budget", "runtime", "vote_average", "popularity", "overview_length", "n_keywords"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=fdf, x="cluster", y=var, hue="cluster", palette="tab10", legend=False, ax=ax)
    if var in ("budget", "popularity"):
        ax.set_yscale("log")
    ax.set(title=f"{var} par cluster", xlabel="Cluster")
    fig.tight_layout()
    show(fig)

with tab_films:
    chosen = st.selectbox("Cluster", list(range(k)))
    sub = fdf[fdf["cluster"] == chosen].sort_values("popularity", ascending=False)
    st.dataframe(sub[["title", "release_date", "genres", "runtime", "budget",
                      "vote_average", "popularity"]].head(100))
