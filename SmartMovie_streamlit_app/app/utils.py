"""Fonctions partagées par toutes les pages de l'application SmartMovie.

Ce module regroupe :
- le chargement des données (MongoDB, avec repli sur data/raw.json) ;
- le feature engineering (identique à l'Étape 4 des notebooks) ;
- la cible, les modèles de classification (Étape 6) ;
- le clustering K-Means (Étape 8) ;
- le moteur de recommandation basé sur TF-IDF (Étape 5).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

# ---------------------------------------------------------------------------
# Chemins et configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_JSON_PATH = PROJECT_ROOT / "data" / "raw.json"
load_dotenv(PROJECT_ROOT / ".env")

RUNTIME_BINS = [0, 90, 120, 150, np.inf]
RUNTIME_LABELS = ["Court (<90min)", "Standard (90-120min)", "Long (120-150min)", "Très long (>150min)"]
BUDGET_LABELS = ["Faible", "Moyen", "Élevé", "Très élevé"]

# Variables disponibles AVANT la sortie du film (cf. Étape 4 : pas de leakage)
NUMERIC_FEATURES = [
    "release_year", "release_month", "release_decade",
    "is_summer_release", "is_holiday_release",
    "n_genres", "n_keywords", "has_keywords",
    "runtime", "overview_length", "is_original_language_en", "budget",
]
CATEGORICAL_FEATURES = ["runtime_category", "budget_category"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# Chargement des données
# ---------------------------------------------------------------------------
def _load_raw() -> tuple[pd.DataFrame, dict]:
    """MongoDB en priorité (source de vérité), repli sur data/raw.json."""
    try:
        import certifi
        from pymongo import MongoClient

        uri = os.getenv("MONGO_URI")
        if not uri:
            raise RuntimeError("variable MONGO_URI absente du fichier .env")
        client = MongoClient(uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=8000)
        docs = list(client["smartmovie"]["movies"].find({}, {"_id": 0}))
        if not docs:
            raise RuntimeError("la collection 'movies' est vide")
        return pd.DataFrame(docs), {"name": "MongoDB", "error": None}
    except Exception as exc:  # noqa: BLE001 - on veut capter toute erreur de connexion
        if not RAW_JSON_PATH.exists():
            raise RuntimeError(
                f"MongoDB inaccessible ({exc}) et {RAW_JSON_PATH} est introuvable."
            ) from exc
        return pd.read_json(RAW_JSON_PATH), {"name": "data/raw.json (repli)", "error": str(exc)}


def _split_csv(value) -> list[str]:
    return [x.strip() for x in str(value or "").split(",") if x.strip()]


def clean_text(text) -> str:
    """Nettoyage identique à l'Étape 5 : minuscules, lettres uniquement."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def budget_edges(budget: pd.Series) -> np.ndarray:
    """Bornes des quartiles de budget (calculées sur les budgets connus)."""
    q = budget.dropna().quantile([0.25, 0.5, 0.75]).to_numpy()
    return np.unique(np.concatenate([[-np.inf], q, [np.inf]]))


def _budget_labels(edges: np.ndarray) -> list[str]:
    return BUDGET_LABELS[: len(edges) - 1]


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Reproduit le nettoyage (EDA) et le feature engineering (Étape 4)."""
    df = raw.copy()
    if "movie_id" in df.columns:
        df = df.drop_duplicates("movie_id")

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    for col in ["budget", "revenue", "runtime"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace(0, np.nan)
    for col in ["overview", "genres", "keywords"]:
        df[col] = df[col].fillna("").astype(str)

    df["genres_list"] = df["genres"].apply(_split_csv)
    df["keywords_list"] = df["keywords"].apply(_split_csv)

    # Temporelles
    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_decade"] = df["release_year"] // 10 * 10
    df["is_summer_release"] = df["release_month"].isin([6, 7, 8]).astype(int)
    df["is_holiday_release"] = df["release_month"].isin([11, 12]).astype(int)

    # Contenu
    df["n_genres"] = df["genres_list"].apply(len)
    df["n_keywords"] = df["keywords_list"].apply(len)
    df["has_keywords"] = (df["n_keywords"] > 0).astype(int)

    # Durée / budget
    df["runtime_category"] = pd.cut(df["runtime"], bins=RUNTIME_BINS, labels=RUNTIME_LABELS)
    edges = budget_edges(df["budget"])
    df["budget_category"] = pd.cut(df["budget"], bins=edges, labels=_budget_labels(edges))

    # Autres
    df["overview_length"] = df["overview"].apply(lambda x: len(x.split()))
    df["is_original_language_en"] = (df["original_language"] == "en").astype(int)

    return df.reset_index(drop=True)


@st.cache_data(show_spinner="Chargement des données…")
def load_movies() -> tuple[pd.DataFrame, dict]:
    raw, source = _load_raw()
    return engineer_features(raw), source


def show_source_status(source: dict) -> None:
    """Affiche dans la sidebar d'où viennent les données."""
    if source["error"]:
        st.sidebar.warning(f"Source : {source['name']}\n\nMongoDB indisponible : {source['error']}")
    else:
        st.sidebar.success(f"Source : {source['name']}")


# ---------------------------------------------------------------------------
# Classification (Étape 6)
# ---------------------------------------------------------------------------
def add_target(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """high_engagement = 1 si vote_count > médiane (seuil relatif à l'échantillon)."""
    data = df[df["vote_count"].notna()].copy()
    threshold = float(data["vote_count"].median())
    data["high_engagement"] = (data["vote_count"] > threshold).astype(int)
    return data, threshold


def get_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne uniquement les features pré-sortie (aucune colonne de leakage)."""
    X = df[FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(object)
    return X


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical = Pipeline(
        [("imputer", SimpleImputer(strategy="most_frequent")),
         ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)]
    )


@st.cache_resource(show_spinner="Entraînement des 3 modèles…")
def train_models(_df: pd.DataFrame, cache_key: int) -> dict:
    data, threshold = add_target(_df)
    X, y = get_model_frame(data), data["high_engagement"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Linear SVM": LinearSVC(random_state=42, max_iter=5000),
    }

    pipelines, rows, predictions, scores = {}, [], {}, {}
    for name, model in models.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("classifier", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_score = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X_test)
        pipelines[name], predictions[name], scores[name] = pipe, y_pred, y_score
        rows.append({
            "Modèle": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred),
            "ROC-AUC": roc_auc_score(y_test, y_score),
        })

    return {
        "pipelines": pipelines,
        "results": pd.DataFrame(rows).set_index("Modèle"),
        "y_test": y_test,
        "predictions": predictions,
        "scores": scores,
        "threshold": threshold,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "class_balance": y.value_counts(normalize=True).sort_index(),
    }


def rf_feature_importance(pipeline: Pipeline, top: int = 15) -> pd.Series:
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    names = [n.split("__", 1)[1] for n in names]
    importances = pipeline.named_steps["classifier"].feature_importances_
    return pd.Series(importances, index=names).sort_values(ascending=False).head(top)


def make_input_row(*, budget, runtime, release_year, release_month, n_genres,
                   n_keywords, is_english, overview_length, edges) -> pd.DataFrame:
    """Construit une ligne de features à partir des saisies de l'utilisateur."""
    budget_cat = np.nan
    if pd.notna(budget):
        budget_cat = pd.cut([budget], bins=edges, labels=_budget_labels(edges))[0]
    runtime_cat = pd.cut([runtime], bins=RUNTIME_BINS, labels=RUNTIME_LABELS)[0]
    row = {
        "release_year": release_year,
        "release_month": release_month,
        "release_decade": release_year // 10 * 10,
        "is_summer_release": int(release_month in (6, 7, 8)),
        "is_holiday_release": int(release_month in (11, 12)),
        "n_genres": n_genres,
        "n_keywords": n_keywords,
        "has_keywords": int(n_keywords > 0),
        "runtime": runtime,
        "overview_length": overview_length,
        "is_original_language_en": int(is_english),
        "budget": budget,
        "runtime_category": runtime_cat,
        "budget_category": budget_cat,
    }
    X = pd.DataFrame([row])[FEATURE_COLUMNS]
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(object)
    return X


# ---------------------------------------------------------------------------
# Clustering (Étape 8)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def structured_matrix(_df: pd.DataFrame, cache_key: int) -> np.ndarray:
    X = build_preprocessor().fit_transform(get_model_frame(_df))
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


@st.cache_data(show_spinner=False)
def text_matrix(_df: pd.DataFrame, cache_key: int) -> tuple[np.ndarray, np.ndarray]:
    texts = _df["overview"].apply(clean_text)
    vec = TfidfVectorizer(stop_words="english", max_features=1000, ngram_range=(1, 2))
    X = vec.fit_transform(texts)
    return X.toarray(), vec.get_feature_names_out()


@st.cache_data(show_spinner=False)
def combined_matrix(_df: pd.DataFrame, cache_key: int) -> np.ndarray:
    X_struct = structured_matrix(_df, cache_key)
    X_text, _ = text_matrix(_df, cache_key)
    reduced = TruncatedSVD(n_components=20, random_state=42).fit_transform(X_text)
    return StandardScaler().fit_transform(np.hstack([X_struct, reduced]))


@st.cache_data(show_spinner="Calcul des silhouette scores…")
def silhouette_scan(_X: np.ndarray, cache_key: str, metric: str, k_max: int = 10) -> dict[int, float]:
    scores = {}
    for k in range(2, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(_X)
        scores[k] = float(silhouette_score(_X, labels, metric=metric))
    return scores


@st.cache_data(show_spinner=False)
def fit_kmeans(_X: np.ndarray, cache_key: str, k: int) -> tuple[np.ndarray, np.ndarray]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(_X)
    return labels, km.cluster_centers_


@st.cache_data(show_spinner=False)
def project_2d(_X: np.ndarray, cache_key: str, method: str) -> tuple[np.ndarray, float]:
    reducer = PCA(n_components=2, random_state=42) if method == "pca" else TruncatedSVD(n_components=2, random_state=42)
    coords = reducer.fit_transform(_X)
    return coords, float(reducer.explained_variance_ratio_.sum())


# ---------------------------------------------------------------------------
# Recommandation (TF-IDF + similarité cosinus)
# ---------------------------------------------------------------------------
def _soup(df: pd.DataFrame) -> pd.Series:
    """Texte décrivant chaque film : synopsis + genres (pondérés x2) + mots-clés."""
    overview = df["overview"].apply(clean_text)
    genres = df["genres_list"].apply(
        lambda gl: " ".join(g.lower().replace(" ", "_") for g in gl)
    )
    keywords = df["keywords"].apply(clean_text)
    return overview + " " + (genres + " ") * 2 + keywords


@st.cache_resource(show_spinner="Construction du moteur de recommandation…")
def build_recommender(_df: pd.DataFrame, cache_key: int) -> dict:
    vec = TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2), min_df=2)
    matrix = vec.fit_transform(_soup(_df))
    return {"vectorizer": vec, "matrix": matrix}


def recommend_similar(rec: dict, idx: int, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    sims = linear_kernel(rec["matrix"][idx], rec["matrix"]).ravel()
    sims[idx] = -1  # on exclut le film lui-même
    top = np.argsort(sims)[::-1][:n]
    return top, sims[top]


def recommend_from_text(rec: dict, text: str, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    query = rec["vectorizer"].transform([clean_text(text)])
    if query.nnz == 0:  # aucun mot connu du vocabulaire
        return np.array([], dtype=int), np.array([])
    sims = linear_kernel(query, rec["matrix"]).ravel()
    top = np.argsort(sims)[::-1][:n]
    return top, sims[top]
