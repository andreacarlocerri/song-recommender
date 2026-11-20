# Song Recommender

A song recommender for **Gnod** (gnoosic.com), which today recommends bands only. It adds two kinds of **song** recommendations:

1. **Popular right now:** if the user's song is in today's charts, suggest other current hits, worldwide (Deezer) and in the US (Billboard).
2. **Sounds like your song:** otherwise, suggest the songs whose **audio features** (energy, tempo, acousticness...) are closest to it, inside the same cluster of similar-sounding songs.

*Ironhack Data Analytics bootcamp, Week 10 project (web scraping, APIs, unsupervised learning, Streamlit). Andrea Carlo Cerri, November 2026.*

**Try the app:** [gnod-song-recommender.streamlit.app](https://gnod-song-recommender.streamlit.app) · **Slides:** [presentation](https://docs.google.com/presentation/d/1VHsD9G1R6L50gKlNEP64UVT7xrKRtZXXa_JsjUNkZGs/edit?usp=sharing)

## Results

- **1,207 songs** from 15 genres plus the Billboard Hot 100, with 9 audio features each, and **200 chart songs** (Deezer worldwide top 100 + Billboard Hot 100).
- **7 clusters** of songs that sound alike (K-Means, chosen with the silhouette, Calinski-Harabasz and Davies-Bouldin scores and compared with agglomerative clustering and DBSCAN), from "Loud & fast" to "Quiet & instrumental".
- **30% hit rate** (recommendations with the same genre as the user's song) vs **6%** for random songs: about **5×** the baseline.
- A **Streamlit app** to try it, with a Deezer player for every song.

## Data sources

| Data | Source | How |
|---|---|---|
| Popular songs, worldwide | [Deezer chart API](https://developers.deezer.com/api) (`chart/0/tracks`) | API, no key needed |
| Popular songs, US | [Billboard Hot 100](https://www.billboard.com/charts/hot-100/) | web scraping (`requests` + `BeautifulSoup`) |
| Songs of 15 genres | Deezer public playlists | API |
| Audio features | [ReccoBeats API](https://reccobeats.com), matched by ISRC | API, no key needed |

Why not Spotify: since November 2024, new Spotify developer apps can no longer get audio features, so the project uses only free APIs that need no account.

## Repository

```
song-recommender/
├── song_recommender.ipynb        the whole analysis, from data collection to evaluation
├── app.py                        the Streamlit app
├── data/
│   ├── deezer_chart_worldwide.csv  Deezer worldwide top 100 (19 Nov 2026)
│   ├── billboard_hot100.csv        Billboard Hot 100, scraped (16 Nov 2026)
│   ├── popular_now.csv             both charts together, with Deezer ids
│   ├── gnod_songs_deezer.csv       1,207 songs with audio features
│   ├── hot100_songs_deezer.csv     the 47 Hot 100 songs with audio features
│   ├── deezer_ids.csv              Deezer id of every song (to play it in the app)
│   └── songs_clustered.csv         songs + cluster, created by the notebook, read by the app
└── requirements.txt
```

## Run it

```bash
pip install -r requirements.txt
jupyter notebook song_recommender.ipynb   # optional: run all cells to rebuild data/songs_clustered.csv
streamlit run app.py                      # opens http://localhost:8501
```

The notebook only calls the websites and APIs if the CSV files in `data/` are missing (the full collection takes about 30 minutes). The app needs an internet connection only for the Deezer players.

## Limitations and next steps

- **Small catalogue:** 1,207 songs. Next: millions of songs, and both charts refreshed every week.
- **Unknown songs:** the app only knows the songs in the catalogue. Next: look songs up live (Deezer → ReccoBeats) and assign them a cluster with `kmeans.predict`.
- **Genre is only a proxy** for "sounds similar". Next: A/B test the recommender with real users (clicks, songs played).
- **Weak clusters:** "Live recordings" groups songs by how they were recorded, not by style.
