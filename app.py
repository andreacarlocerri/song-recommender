"""Song Recommender for Gnod: pick a song, get songs that are popular right now or that sound like it.

Run the notebook song_recommender.ipynb once first (it creates data/songs_clustered.csv), then:

    streamlit run app.py
"""

import re

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler

FEATURES = ["danceability", "energy", "loudness", "speechiness", "acousticness",
            "instrumentalness", "liveness", "valence", "tempo"]

CLUSTER_DESCRIPTIONS = {
    "Loud & fast": "loud, energetic and fast, not acoustic (punk, rock, gospel)",
    "Soft & acoustic": "quiet, low energy and acoustic (folk, soul, jazz)",
    "Danceable & wordy": "very danceable, with lots of spoken words (reggae, flamenco, hip hop)",
    "Live recordings": "songs recorded live, whatever the style",
    "Quiet & instrumental": "very quiet, slow and without vocals (classical)",
    "Upbeat & happy": "danceable, energetic and happy (reggae, salsa, rock)",
    "Instrumental grooves": "danceable but mostly without vocals (hip hop, jazz, salsa)",
}

st.set_page_config(page_title="Song Recommender", page_icon="🎵")


def clean_title(title):
    """Lower-case title without the text in brackets: "Dracula (with JENNIE)" -> "dracula"."""
    return re.sub(r"\s*[\(\[].*?[\)\]]", "", str(title)).strip().lower()


@st.cache_data
def load_data():
    """Read the files once, scale the audio features and compute every distance between two songs."""
    songs = pd.read_csv("data/songs_clustered.csv", dtype={"deezer_id": "Int64"})
    popular = pd.read_csv("data/popular_now.csv", dtype={"deezer_id": "Int64"})
    popular["key"] = popular["title"].map(clean_title)
    X = StandardScaler().fit_transform(songs[FEATURES])
    return songs, popular, pairwise_distances(X)


def deezer_player(deezer_id):
    """Deezer's embedded player (30-second preview) for one song."""
    if pd.isna(deezer_id):
        return
    url = f"https://widget.deezer.com/widget/dark/track/{int(deezer_id)}"
    if hasattr(st, "iframe"):          # newer Streamlit versions
        st.iframe(url, height=160)
    else:
        components.iframe(url, height=160)


def is_popular(title, artist):
    """True if a song with this title (and an artist with the same first word) is in today's charts."""
    first_word = str(artist).lower().split()[0]
    same_title = popular[popular["key"] == clean_title(title)]
    return same_title["artist"].str.lower().str.contains(first_word, regex=False).any()


songs, popular, D = load_data()

# ---------------------------------------------------------------- sidebar: popular right now
with st.sidebar:
    st.header("Settings")
    n = st.slider("Number of recommendations", 3, 10, 5)
    st.header("🔥 Popular right now")
    for tab, chart in zip(st.tabs(["Worldwide", "US"]), ["Deezer worldwide", "Billboard Hot 100 (US)"]):
        with tab:
            st.dataframe(popular.loc[popular["chart"] == chart, ["rank", "title", "artist"]].head(10), hide_index=True)
    st.caption("Deezer worldwide chart (19 Nov 2026) and Billboard Hot 100 (16 Nov 2026).")

# ---------------------------------------------------------------- main page
st.title("🎵 Song Recommender")
st.write("Pick a song you like. If it's popular right now, we suggest other hits. "
         "Otherwise, we suggest songs that **sound** like it.")

choice = st.selectbox("Your song", options=songs.index, index=None,
                      format_func=lambda i: f"{songs.loc[i, 'song']} — {songs.loc[i, 'artist']}",
                      placeholder="Type a song or an artist...")

if choice is None:
    st.info(f"Our catalogue has {len(songs):,} songs from 15 genres, plus today's charts. "
            "Try *Bam Bam*, *Teenage Kicks* or *Dracula*.")
    st.stop()

picked = songs.loc[choice]
deezer_player(picked["deezer_id"])
picked_key = clean_title(picked["song"])

if is_popular(picked["song"], picked["artist"]):
    # Feature 1: popular right now
    st.subheader("🔥 It's popular right now! Other hits:")
    others = popular[popular["key"] != picked_key].drop_duplicates(subset="key")
    for _, hit in others.sample(n, random_state=choice).iterrows():
        st.markdown(f"**{hit['title']}** — {hit['artist']}  ·  *#{hit['rank']} {hit['chart']}*")
        deezer_player(hit["deezer_id"])
else:
    # Feature 2: sounds like your song = the closest songs of the same cluster
    st.subheader(f"🎧 It sounds *{picked['cluster_name']}*. Songs that sound like it:")
    st.caption(f"{picked['cluster_name']}: {CLUSTER_DESCRIPTIONS.get(picked['cluster_name'], '')}.")
    same_cluster = np.flatnonzero(songs["cluster"].values == picked["cluster"])
    same_cluster = same_cluster[same_cluster != choice]
    closest = same_cluster[np.argsort(D[choice, same_cluster])[:n]]
    for i in closest:
        song = songs.loc[i]
        st.markdown(f"**{song['song']}** — {song['artist']}  ·  *{song['genre']}*")
        deezer_player(song["deezer_id"])

    with st.expander("Why these songs?"):
        compared = songs.loc[[choice, *closest], ["song"] + FEATURES].set_index("song")
        st.write("The audio features of your song (first row) and of the recommendations:")
        st.dataframe(compared.round(2))

st.divider()
st.caption("Songs and players from Deezer, audio features from ReccoBeats, charts from Deezer and Billboard. "
           "7 clusters with K-Means; recommendations are the closest songs of the same cluster.")
