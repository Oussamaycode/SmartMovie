import requests
import os
import pandas as pd
import time
from dotenv import load_dotenv

load_dotenv()

keyapi = os.getenv("TMDB_TOKEN")

headers = {
    "Authorization": f"Bearer {keyapi}",
    "accept": "application/json"
}

discover_endpoint = "https://api.themoviedb.org/3/discover/movie"
MAX_MOVIES = 1000
movies = []

for page in range(1, 51):
    params = {
        "page": page,
        "sort_by": "popularity.desc",
        "language": "en-US"
    }

    response = requests.get(discover_endpoint, headers=headers, params=params)
    response.raise_for_status()
    results = response.json().get("results", [])

    for m in results:
        detail_url = f"https://api.themoviedb.org/3/movie/{m['id']}"
        detail_response = requests.get(
            detail_url,
            headers=headers,
            params={"language": "en-US", "append_to_response": "keywords"}
        )

        if detail_response.status_code != 200:
            print(f"Skipping {m['id']}: {detail_response.status_code}")
            continue

        movie = detail_response.json()

        movies.append({
            "movie_id": movie.get("id"),
            "title": movie.get("title"),
            "overview": movie.get("overview"),
            "release_date": movie.get("release_date"),
            "runtime": movie.get("runtime"),
            "original_language": movie.get("original_language"),
            "genres": ", ".join(g["name"] for g in movie.get("genres", [])),
            "keywords": ", ".join(
                k["name"] for k in movie.get("keywords", {}).get("keywords", [])
            ),
            "budget": movie.get("budget"),
            "revenue": movie.get("revenue"),
            "popularity": movie.get("popularity"),
            "vote_average": movie.get("vote_average"),
            "vote_count": movie.get("vote_count")
        })

        if len(movies) >= MAX_MOVIES:
            break

    if len(movies) >= MAX_MOVIES:
        break

df = pd.DataFrame(movies)
df.to_json("SmartMovie/data/raw.json")