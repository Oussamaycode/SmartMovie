"""Classification : prédire high_engagement avant la sortie d'un film."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve

from utils import (
    budget_edges,
    load_movies,
    make_input_row,
    rf_feature_importance,
    show_source_status,
    train_models,
)

st.set_page_config(page_title="Classification · SmartMovie", page_icon="🤖", layout="wide")
sns.set_theme(style="whitegrid")

df, source = load_movies()
show_source_status(source)
bundle = train_models(df, len(df))
results = bundle["results"]


def show(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


st.title("🤖 Classification — `high_engagement`")
st.markdown(
    f"**Cible** : `high_engagement = 1` si `vote_count` > **{bundle['threshold']:.0f} votes** "
    "(médiane de l'échantillon : seuil relatif, classes équilibrées).  \n"
    "**Anti-leakage** : `vote_count` (qui définit la cible), `vote_average`, `popularity` et `revenue` "
    "sont exclus des features. Seules les variables connues *avant* la sortie sont utilisées."
)
balance = bundle["class_balance"]
c1, c2, c3 = st.columns(3)
c1.metric("Films d'entraînement", bundle["n_train"])
c2.metric("Films de test", bundle["n_test"])
c3.metric("Part « high engagement »", f"{balance.get(1, 0):.0%}")

tab_perf, tab_detail, tab_imp, tab_try = st.tabs(
    ["Performances", "Matrices de confusion & ROC", "Importance des variables", "Tester un film"]
)

# ---------------------------------------------------------------- Performances
with tab_perf:
    st.dataframe(results.style.format("{:.3f}").highlight_max(axis=0, color="#2e7d32"))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    results.plot(kind="bar", ax=ax, colormap="viridis")
    ax.set(ylim=(0, 1), ylabel="Score", xlabel="")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    show(fig)
    best_f1 = results["F1-score"].idxmax()
    best_auc = results["ROC-AUC"].idxmax()
    st.info(
        f"**Lecture.** Meilleur F1 : **{best_f1}** ({results.loc[best_f1, 'F1-score']:.3f}) ; "
        f"meilleur ROC-AUC : **{best_auc}** ({results.loc[best_auc, 'ROC-AUC']:.3f}). "
        "L'accuracy seule ne dit rien du type d'erreur : la *precision* mesure la fiabilité des films "
        "annoncés « high », le *recall* la part des vrais « high » retrouvés. "
        "Un ROC-AUC proche de 0.5 signifierait que les features pré-sortie séparent mal les deux classes."
    )

# ---------------------------------------------------------------- Détails
with tab_detail:
    cols = st.columns(3)
    for col, (name, y_pred) in zip(cols, bundle["predictions"].items()):
        with col:
            fig, ax = plt.subplots(figsize=(4, 3.6))
            ConfusionMatrixDisplay(
                confusion_matrix(bundle["y_test"], y_pred), display_labels=["Low", "High"]
            ).plot(ax=ax, cmap="Blues", colorbar=False)
            ax.set_title(name, fontsize=11)
            fig.tight_layout()
            show(fig)
    st.caption("Diagonale = bonnes prédictions. Hors diagonale : faux positifs (haut-droite) et faux négatifs (bas-gauche).")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for name, y_score in bundle["scores"].items():
        fpr, tpr, _ = roc_curve(bundle["y_test"], y_score)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {results.loc[name, 'ROC-AUC']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Hasard")
    ax.set(title="Courbes ROC", xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs")
    ax.legend()
    fig.tight_layout()
    show(fig)
    st.info(
        "**Lecture.** Plus une courbe se rapproche du coin supérieur gauche, mieux le modèle sépare les classes "
        "*à tous les seuils*, pas seulement au seuil par défaut utilisé pour les métriques."
    )

# ---------------------------------------------------------------- Importance
with tab_imp:
    importance = rf_feature_importance(bundle["pipelines"]["Random Forest"])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.barplot(x=importance.values, y=importance.index, hue=importance.index,
                palette="rocket", legend=False, ax=ax)
    ax.set(title="Importance des variables — Random Forest", xlabel="Importance", ylabel="")
    fig.tight_layout()
    show(fig)
    st.info(
        f"**Lecture.** La variable la plus utilisée est **{importance.index[0]}**. "
        "Attention : l'importance mesure l'usage par le modèle, pas une relation causale ; "
        "les variables numériques continues (budget, année) ont tendance à être favorisées par les forêts."
    )

# ---------------------------------------------------------------- Tester un film
with tab_try:
    st.markdown("Décris un film **avant sa sortie** et vois ce que prédit chaque modèle.")
    edges = budget_edges(df["budget"])
    year_min, year_max = int(df["release_year"].min()), int(df["release_year"].max())

    with st.form("predict_form"):
        model_name = st.selectbox("Modèle", list(bundle["pipelines"].keys()), index=1)
        f1, f2, f3 = st.columns(3)
        budget_musd = f1.number_input("Budget (M$)", min_value=0.0, max_value=1000.0, value=50.0, step=5.0)
        unknown_budget = f1.checkbox("Budget inconnu")
        is_english = f1.checkbox("Film en anglais", value=True)
        runtime = f2.slider("Durée (min)", 40, 240, 110)
        release_year = f2.slider("Année de sortie", year_min, year_max, year_max)
        release_month = f2.selectbox("Mois de sortie", list(range(1, 13)), index=5)
        n_genres = f3.slider("Nombre de genres", 1, 6, 2)
        n_keywords = f3.slider("Nombre de mots-clés", 0, 40, 8)
        overview_length = f3.number_input("Longueur du synopsis (mots)", 0, 300, 40)
        submitted = st.form_submit_button("Prédire")

    if submitted:
        row = make_input_row(
            budget=np.nan if unknown_budget else budget_musd * 1e6,
            runtime=runtime, release_year=release_year, release_month=release_month,
            n_genres=n_genres, n_keywords=n_keywords, is_english=is_english,
            overview_length=overview_length, edges=edges,
        )
        pipe = bundle["pipelines"][model_name]
        label = int(pipe.predict(row)[0])
        r1, r2 = st.columns(2)
        r1.metric("Prédiction", "High engagement" if label else "Low engagement")
        if hasattr(pipe, "predict_proba"):
            r2.metric("Probabilité « high »", f"{pipe.predict_proba(row)[0, 1]:.0%}")
        else:
            r2.metric("Score de décision (SVM)", f"{pipe.decision_function(row)[0]:+.2f}")
            st.caption("Le Linear SVM ne fournit pas de probabilité : un score > 0 correspond à « high ».")
        st.caption(
            "Prédiction indicative : le modèle n'utilise que des métadonnées de production, "
            "sans le contenu du film ni son marketing."
        )
        with st.expander("Variables envoyées au modèle"):
            st.dataframe(row.T.rename(columns={0: "valeur"}).astype(str))
