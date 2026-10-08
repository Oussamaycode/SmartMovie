"""Dashboard : visualisations de l'EDA, avec filtres interactifs."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st

from utils import load_movies, show_source_status

st.set_page_config(page_title="Dashboard · SmartMovie", page_icon="📊", layout="wide")
sns.set_theme(style="whitegrid")

df, source = load_movies()
show_source_status(source)


def show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------- Filtres
st.sidebar.header("Filtres")
years = df["release_year"].dropna().astype(int)
year_range = st.sidebar.slider(
    "Années de sortie", int(years.min()), int(years.max()), (int(years.min()), int(years.max()))
)
all_genres = sorted({g for gl in df["genres_list"] for g in gl})
selected_genres = st.sidebar.multiselect("Genres (au moins un)", all_genres)
languages = sorted(df["original_language"].dropna().unique())
selected_langs = st.sidebar.multiselect("Langue originale", languages)

fdf = df[df["release_year"].between(*year_range)]
if selected_genres:
    fdf = fdf[fdf["genres_list"].apply(lambda gl: any(g in gl for g in selected_genres))]
if selected_langs:
    fdf = fdf[fdf["original_language"].isin(selected_langs)]
fdf = fdf.reset_index(drop=True)

st.title("📊 Dashboard")
if fdf.empty:
    st.warning("Aucun film ne correspond à ces filtres.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Films", f"{len(fdf):,}", f"{len(fdf) / len(df):.0%} du total")
k2.metric("Note moyenne", f"{fdf['vote_average'].mean():.2f}")
k3.metric("Durée médiane", f"{fdf['runtime'].median():.0f} min" if fdf["runtime"].notna().any() else "n/a")
k4.metric("Budget médian", f"{fdf['budget'].median() / 1e6:.0f} M$" if fdf["budget"].notna().any() else "n/a")

tabs = st.tabs([
    "Notes & popularité", "Genres", "Années", "Durée",
    "Budget & revenus", "Votes & popularité", "Boxplots & corrélations",
])

# ---------------------------------------------------------------- 1. Notes & popularité
with tabs[0]:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(fdf["vote_average"].dropna(), bins=20, kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set(title="Distribution des notes", xlabel="Note moyenne")
    sns.histplot(fdf["popularity"].dropna(), bins=30, kde=True, ax=axes[1], color="#DD8452")
    axes[1].set(title="Distribution de la popularité", xlabel="Popularité")
    fig.tight_layout()
    show(fig)
    skew = fdf["popularity"].mean() / fdf["popularity"].median()
    st.info(
        f"**Interprétation.** Note moyenne {fdf['vote_average'].mean():.2f} (médiane {fdf['vote_average'].median():.2f}) : "
        "les notes forment une cloche resserrée, car le jeu de données ne contient que des films déjà populaires. "
        f"La popularité est très asymétrique : sa moyenne ({fdf['popularity'].mean():.0f}) vaut {skew:.1f}× sa médiane "
        f"({fdf['popularity'].median():.0f}), tirée vers le haut par quelques blockbusters."
    )

# ---------------------------------------------------------------- 2. Genres
with tabs[1]:
    genre_counts = fdf.explode("genres_list").reset_index(drop=True)["genres_list"].dropna().value_counts()
    if genre_counts.empty:
        st.info("Aucun genre renseigné pour cette sélection.")
    else:
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(x=genre_counts.values, y=genre_counts.index, hue=genre_counts.index,
                    palette="viridis", legend=False, ax=ax)
        ax.set(title="Nombre de films par genre", xlabel="Nombre de films", ylabel="")
        fig.tight_layout()
        show(fig)
        st.info(
            f"**Interprétation.** Le genre le plus fréquent est **{genre_counts.index[0]}** "
            f"({genre_counts.iloc[0]} films). Un film pouvant avoir plusieurs genres, la somme des barres "
            "dépasse le nombre de films."
        )

# ---------------------------------------------------------------- 3. Années
with tabs[2]:
    year_counts = fdf["release_year"].dropna().astype(int).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(year_counts.index, year_counts.values, marker="o", color="#55A868")
    ax.set(title="Nombre de films sortis par année", xlabel="Année", ylabel="Films")
    fig.tight_layout()
    show(fig)
    recent_share = (fdf["release_year"] >= fdf["release_year"].max() - 9).mean()
    st.info(
        f"**Interprétation.** Pic en **{year_counts.idxmax()}** ({year_counts.max()} films) ; "
        f"{recent_share:.0%} des films datent des 10 dernières années de la sélection. "
        "Le jeu de données est trié par popularité actuelle : il reflète l'attention du moment, "
        "pas la production réelle par année."
    )

# ---------------------------------------------------------------- 4. Durée
with tabs[3]:
    runtime = fdf["runtime"].dropna()
    if runtime.empty:
        st.info("Durée non renseignée pour cette sélection.")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        sns.histplot(runtime, bins=30, kde=True, ax=axes[0], color="#8172B2")
        axes[0].axvline(runtime.median(), color="red", linestyle="--", label=f"Médiane {runtime.median():.0f} min")
        axes[0].set(title="Distribution de la durée", xlabel="Minutes")
        axes[0].legend()
        cat_counts = fdf["runtime_category"].value_counts().reindex(
            fdf["runtime_category"].cat.categories
        )
        sns.barplot(x=cat_counts.values, y=cat_counts.index.astype(str), hue=cat_counts.index.astype(str),
                    palette="mako", legend=False, ax=axes[1])
        axes[1].set(title="Catégories de durée", xlabel="Films", ylabel="")
        fig.tight_layout()
        show(fig)
        st.info(
            f"**Interprétation.** Durée médiane {runtime.median():.0f} min ; "
            f"{runtime.between(90, 130).mean():.0%} des films durent entre 90 et 130 minutes, "
            "le format standard du long métrage."
        )

# ---------------------------------------------------------------- 5. Budget & revenus
with tabs[4]:
    both = fdf[["budget", "revenue"]].dropna()
    c1, c2 = st.columns(2)
    for col, ctn, color, label in [("budget", c1, "#C44E52", "Budget (M$)"), ("revenue", c2, "#4C72B0", "Revenu (M$)")]:
        s = fdf[col].dropna() / 1e6
        with ctn:
            if s.empty:
                st.info(f"{label} non renseigné.")
            else:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.histplot(s, bins=30, kde=True, ax=ax, color=color)
                ax.set(title=f"Distribution — {label}", xlabel=label)
                fig.tight_layout()
                show(fig)
    if len(both) > 2:
        log_scale = st.checkbox("Échelle logarithmique (budget vs revenu)", value=True)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(both["budget"], both["revenue"], alpha=0.5, color="#55A868")
        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set(title="Budget vs revenu", xlabel="Budget ($)", ylabel="Revenu ($)")
        fig.tight_layout()
        show(fig)
        corr = both.corr().iloc[0, 1]
        st.info(
            f"**Interprétation.** Corrélation budget/revenu : **{corr:.2f}** (sur {len(both)} films où les deux sont connus). "
            "Les deux variables sont très asymétriques (d'où l'échelle log) ; un gros budget "
            "ne garantit pas un gros revenu. Les valeurs à 0 de TMDB sont traitées comme manquantes."
        )
    else:
        st.info("Pas assez de films avec budget et revenu connus pour tracer le nuage de points.")

# ---------------------------------------------------------------- 6. Votes & popularité
with tabs[5]:
    log_vp = st.checkbox("Échelle logarithmique (votes vs popularité)", value=True)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sc = ax.scatter(fdf["vote_count"], fdf["popularity"], c=fdf["vote_average"], cmap="coolwarm", alpha=0.6)
    fig.colorbar(sc, ax=ax, label="Note moyenne")
    if log_vp:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set(title="Popularité vs nombre de votes", xlabel="Nombre de votes", ylabel="Popularité")
    fig.tight_layout()
    show(fig)
    corr_vp = fdf[["vote_count", "popularity"]].corr(method="spearman").iloc[0, 1]
    st.info(
        f"**Interprétation.** Corrélation de Spearman votes/popularité : **{corr_vp:.2f}**. "
        "Plus un film est voté, plus il tend à être populaire ; la couleur montre que popularité "
        "et note ne vont pas toujours ensemble : la popularité mesure l'attention, pas la qualité perçue."
    )

# ---------------------------------------------------------------- 7. Boxplots & corrélations
with tabs[6]:
    exploded = fdf.explode("genres_list").reset_index(drop=True)
    top_genres = exploded["genres_list"].value_counts().head(8).index.tolist()
    top_df = exploded[exploded["genres_list"].isin(top_genres)].reset_index(drop=True)

    if top_df.empty:
        st.info("Aucun genre renseigné pour cette sélection.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.boxplot(data=top_df, x="genres_list", y="vote_average", hue="genres_list",
                        palette="Set2", legend=False, ax=ax)
            ax.set(title="Note par genre (8 plus fréquents)", xlabel="", ylabel="Note moyenne")
            ax.tick_params(axis="x", rotation=40)
            fig.tight_layout()
            show(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.boxplot(data=top_df, x="genres_list", y="runtime", hue="genres_list",
                        palette="Set3", legend=False, ax=ax)
            ax.set(title="Durée par genre (8 plus fréquents)", xlabel="", ylabel="Minutes")
            ax.tick_params(axis="x", rotation=40)
            fig.tight_layout()
            show(fig)

    num_cols = ["vote_average", "vote_count", "popularity", "runtime", "budget", "revenue"]
    corr_matrix = fdf[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, ax=ax)
    ax.set_title("Matrice de corrélation")
    fig.tight_layout()
    show(fig)

    pairs = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)).stack()
    if not pairs.empty:
        best = pairs.abs().idxmax()
        st.info(
            f"**Interprétation.** Les boxplots comparent la dispersion (et pas seulement la moyenne) "
            f"d'un genre à l'autre. Dans la matrice, la paire la plus corrélée est "
            f"**{best[0]} / {best[1]}** ({pairs[best]:.2f}) ; la note moyenne reste en général peu liée "
            "aux variables financières."
        )

with st.expander("Voir les données filtrées"):
    st.dataframe(fdf[["title", "release_date", "genres", "runtime", "budget", "revenue",
                      "popularity", "vote_average", "vote_count"]])
