import json
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import zipfile
import numpy as np
import pandas as pd
import os
from google import genai
import toml
import random
from sentence_transformers import SentenceTransformer, util
import time
import re
from google.genai.errors import APIError, ClientError, ServerError
import requests
from pathlib import Path
import joblib
import asyncio
import backoff
import gzip
from datetime import datetime, timedelta
import ast
from urllib.parse import urlparse
import io
import tempfile
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import hashlib
import spacy
import gc
import pickle
from BERT_update import (
    risk_weights, 
    load_full_topics, 
    load_articles_from_release, 
    atomic_write_csv, 
    atomic_write_pickle,
    upload_asset_to_release,
    load_model_bundle,
    upload_file,
    download_file,
    blob_exists,
    upload_bytes
)

Github_owner = 'ERSRisk'
Github_repo = 'Tulane-Sentiment-Analysis'
Release_tag = 'BERTopic_results'
GITHUB_TOKEN = os.getenv('TOKEN')

GEMINI_API_KEY = os.getenv("PAID_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def track_over_time(df, week_anchor="W-MON", out_csv="pipeline/resources/topic_trend.csv"):

    if 'Published' not in df.columns:
        print("⚠️ 'Published' column missing; skipping trend tracking.")
        return

    df = df.copy()

    # --- 1) Coerce to datetime, handle epoch numbers & strings, keep UTC then strip tz ---
    def _coerce_pub(x):
        if pd.isna(x): 
            return pd.NaT
        # epoch millis / seconds
        if isinstance(x, (int, float)):
            if x > 1e12:   # ms
                return pd.to_datetime(x, unit='ms', errors='coerce', utc=True)
            if x > 1e9:    # s
                return pd.to_datetime(x, unit='s', errors='coerce', utc=True)
        # general string/datetime
        return pd.to_datetime(x, errors='coerce', utc=True)

    df['Published'] = df['Published'].apply(_coerce_pub)
    df = df.dropna(subset=['Published'])

    # strip tz (Period ops are simplest on naive timestamps)
    if pd.api.types.is_datetime64tz_dtype(df['Published']):
        df['Published'] = df['Published'].dt.tz_convert('UTC').dt.tz_localize(None)
    else:
        # already naive or non-tz datetime64
        df['Published'] = df['Published'].dt.tz_localize(None)

    # --- 2) Week bucket (anchor Monday by default) ---
    # e.g. "W-SUN" if you prefer Sunday starts
    df['week'] = df['Published'].dt.to_period(week_anchor).apply(lambda p: p.start_time)

    # --- 3) Topic names (safe load) ---
    topic_name_map = {}
    try:
        with open('pipeline/resources/topics_BERT.json', 'r', encoding='utf-8') as f:
            topics_json = json.load(f)
            topic_name_map = {t['topic']: t['name'] for t in topics_json if 'topic' in t and 'name' in t}
    except FileNotFoundError:
        print("⚠️ topics_BERT.json not found; labeling as 'Unlabeled Topic'.")

    df['Topic_Name'] = df.get('Topic').map(topic_name_map) if 'Topic' in df.columns else "Unlabeled Topic"
    df['Topic_Name'] = df['Topic_Name'].fillna('Unlabeled Topic')

    # --- 4) Aggregate & save ---
    topic_trend = (
        df.groupby(['week', 'Topic_Name'], dropna=False)
          .size().reset_index(name='article_count')
          .sort_values(['week', 'article_count'], ascending=[True, False])
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    topic_trend.to_csv(out_csv, index=False)
    print(f"✅ Saved topic trend to {out_csv}")

def ensure_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    if 'Risk_Score' not in df.columns:
        print("Risk Score missing -- recomputing", flush = True)
        return risk_weights(df)
    if df['Risk_Score'].isna().all():
        print("Risk Score all NaN -- recomputing", flush = True)
        return risk_weights(df)
    nan_ratio = df['Risk_Score'].isna().mean()
    if nan_ratio > 0.2:
        print(f"Risk Score {nan_ratio: .1%} NaN -- recomputing", flush = True)
        return risk_weights(df)
    return df
def build_stories():
    
    model = SentenceTransformer("all-MiniLM-L6-v2")



    trash_topics = [95,94,76,75,52,44,17,10,7,0,559,527,515,503,481,474,469,462,
                461,452,450,445,438,434,395,389,354,349,345,323,315,301,299,
                258,257,254,249,236,234,228,224,208,198,191,188,186,178,177,
                174,172,167,164,156,154,140,136,135,130,125,110,101,90,84,73,
                60,59,56,54,50,24,22,18,568,565,550,526,518,505,484,477,458,
                456,387,245,239,226,196,155,144,123,117,109,105,85,61,33,28,
                25,16,14]
    df = load_full_topics(download_file('latest/BERTopic_results2.csv.gz', 'pipeline/resources/BERTopic_results2.csv.gz'))
    df = ensure_risk_scores(df)
    df['University Label'] = pd.to_numeric(df['University Label'], errors='coerce').fillna(0).astype(int)
    df = df[df['University Label'] == 1].copy()  # ← only cluster university-relevant articles
    df['Published_utc'] = pd.to_datetime(df['Published_utc'], errors='coerce', utc=True)
    df = df[df['Published_utc'].notna()]

    

    stories_df_exists = Path('pipeline/resources/Story_Clusters.csv.gz').exists()
    articles_with_stories_exists = Path('pipeline/resources/dashboard_dropdown.csv.gz').exists()

    if stories_df_exists and articles_with_stories_exists:
        old_df = pd.read_csv('pipeline/resources/dashboard_dropdown.csv.gz', compression='gzip')
        if old_df['story_id'].notna().sum() == 0:
            print("INVALID STATE: stories exist but no articles reference them. RESETTING.")
            Path("pipeline/resources/Story_Clusters.csv.gz").unlink(missing_ok = True)
            Path("pipeline/resources/Articles_with_Stories.csv.gz").unlink(missing_ok = True)
            old_df = pd.DataFrame(columns=df.columns.tolist() + ['story_id', '_key'])
    else:
        old_df = pd.DataFrame(columns=df.columns.tolist() + ['story_id'] + ['_key'])

    if 'Published_utc' in old_df.columns:
        old_df['Published_utc'] = pd.to_datetime(
            old_df['Published_utc'],
            errors = 'coerce',
            utc=True
        )
    



    df = df[df['Published_utc'].notna()]
    df['orig_idx'] = df.index
    already_labeled = old_df.dropna(subset=['story_id'])
    cutoff = old_df['Published_utc'].max()
    if pd.isna(cutoff):
        cutoff = pd.Timestamp.min.tz_localize("UTC")
    elif cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
        
    new_articles = df[df['Published_utc'] > cutoff].copy()
    new_articles['story_id'] = np.nan

    new_articles['_key'] = list(zip(new_articles['Title'], new_articles['Link']))
    already_labeled['_key'] = list(zip(already_labeled['Title'], already_labeled['Link']))

    new_articles = new_articles[
        ~new_articles['_key'].isin(already_labeled['_key'])
    ].drop(columns='_key')

    df = pd.concat([already_labeled, new_articles], ignore_index = True)

    if Path('pipeline/resources/Story_Clusters.csv.gz').exists():
        stories_df = pd.read_csv('pipeline/resources/Story_Clusters.csv.gz', compression='gzip')
    else:
        stories_df = pd.DataFrame(columns=[
        'story_id', 'canonical_title', 'canonical_link', 'canonical_published',
        'article_count', 'first_seen', 'last_seen'
    ])

    #if df['story_id'].notna().sum() == 0:
        #stories_df = pd.DataFrame(columns = stories_df.columns)

    if Path('pipeline/resources/Canonical_Stories_with_Summaries.csv').exists():
        canonical_titles = pd.read_csv('pipeline/resources/Canonical_Stories_with_Summaries.csv')
        canonical_titles['story_id'] = canonical_titles['story_id'].astype(int)
        stories_df = stories_df.merge(
            canonical_titles[['story_id', 'canonical_title', 'canonical_source']],
            on='story_id', how='left', suffixes=('', '_gemini'), validate='one_to_one'
        )
        if 'canonical_title_gemini' in stories_df.columns:
            stories_df['canonical_title'] = stories_df['canonical_title_gemini'].fillna(stories_df['canonical_title'])
        stories_df.drop(columns=[c for c in stories_df.columns if c.endswith('_gemini')], inplace=True, errors='ignore')
    
    stories_df['story_id'] = stories_df['story_id'].astype(int)
    
    # Prefer Gemini canonical_title when present, otherwise keep existing
    if 'canonical_title_gemini' in stories_df.columns:
        stories_df['canonical_title'] = stories_df['canonical_title_gemini'].fillna(stories_df['canonical_title'])
    
    # Keep canonical_source (new column)
    # (canonical_source is only in canonical_titles, so it won't conflict)
    stories_df.drop(columns=[c for c in stories_df.columns if c.endswith('_gemini')], inplace=True, errors='ignore')

    story_id_counter = int(stories_df['story_id'].max()) + 1 if not stories_df.empty else 1
    open_stories = []

    def norm_text(x):
        x = (x or '')
        x = re.sub(r'\s+', ' ', x).strip()
        return x

    open_stories = []

    already_labeled['story_embeddings'] = None

    if not new_articles.empty:
        new_embeddings = model.encode(new_articles['Title'].fillna('').tolist(), convert_to_numpy = True, normalize_embeddings = True, show_progress_bar = True)
        new_articles['story_embeddings'] = list(new_embeddings)

    else:
        new_articles['story_embeddings'] = []

    

    articles_by_story = {}

    if df['story_id'].notna().sum() > 0:
        articles_by_story = (df[df['story_id'].notna()].groupby('story_id').apply(lambda g: g.to_dict('records')).to_dict())

    if not articles_by_story:
        print("No existing story assignments - starting fresh clustering", flush = True)
        stories_df = stories_df.iloc[0:0]


    if Path('pipeline/resources/story_centroids.pkl').exists():
        with open('pipeline/resources/story_centroids.pkl', 'rb') as f:
            centroids_df = pickle.load(f)
        if isinstance(centroids_df, list):
            centroid_map = centroid_map = {int(row['story_id']): np.array(row['centroid']) for row in centroids_df}
        if isinstance(centroids_df, pd.DataFrame):
            centroid_map = {int(row['story_id']): np.array(row['centroid']) for _, row in centroids_df.iterrows()}
        elif isinstance(centroids_df, dict):
            centroid_map = {int(k): np.array(v) for k, v in centroids_df.items()}
        
    else:
        centroid_map = {}

    for _, row in stories_df.iterrows():
        sid = int(row['story_id'])
        rows = articles_by_story.get(sid, [])
        if not rows:
            continue

        if sid in centroid_map:
            centroid = centroid_map[sid]
        else:
            texts = [norm_text(f"{str(r.get('Title') or '')} {str(r.get('Summary') or '')}") 
                 for r in rows]
            centroid = model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            ).mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)

        first_seen = min(pd.to_datetime(r['Published_utc'], errors='coerce') for r in rows)
        last_seen = max(pd.to_datetime(r['Published_utc'], errors='coerce') for r in rows)

        open_stories.append({
            "id": sid,
            "centroid": centroid,
            "rows": rows,
            "n": len(rows),
            "first_seen": pd.to_datetime(first_seen, errors='coerce'),
            "last_seen": pd.to_datetime(last_seen, errors='coerce'),
            "canonical_title": row.get('canonical_title', None)
        })



    def build_story_clusters(df, open_stories, story_id_counter, stories_df, min_sim = 0.52):
        df = df.copy()


        df['text_for_embedding'] = (df['Title'].fillna('') + ' ' + df['Summary'].fillna('')).apply(norm_text)
        df['date_bucket'] = df['Published_utc'].dt.floor('D')
        df.sort_values(by='Published_utc', inplace=True)


        story_rows = []
        article_story_ids = []
    

    
        MAX_GAP_DAYS = 21
    

        df = df.copy()
        for pos, (idx, row) in enumerate(df.iterrows()):
            best_sim = -1
            best_story = None
            embed_i = row['story_embeddings']
            if not isinstance(embed_i, np.ndarray):
                embed_i = model.encode([row['Title'] or ''], convert_to_numpy=True, normalize_embeddings=True
                )[0]
            pub_i = row["Published_utc"]


            candidate_stories = [
                s for s in open_stories
                if pd.isna(pub_i)
                or pd.isna(s["last_seen"])
                or (pub_i - s["last_seen"]).days <= MAX_GAP_DAYS
            ]

        
            for s in candidate_stories:
                sim = float(np.dot(embed_i, s["centroid"]))
                if sim > best_sim:
                    best_sim = sim
                    best_story = s

            if best_sim >= min_sim:
                # assign
                n = best_story["n"]
                centroid = (best_story["centroid"] * n + embed_i) / (n + 1)
                best_story["centroid"] = centroid / np.linalg.norm(centroid)
                best_story["rows"].append(row.to_dict())
                best_story["n"] += 1
                best_story["first_seen"] = min(best_story["first_seen"], pub_i)
                best_story["last_seen"] = max(best_story["last_seen"], pub_i)
                article_story_ids.append((idx, best_story))
            else:
                # new story
            
                sid = story_id_counter
                story_id_counter += 1
                new_story = {
                    "id": sid,
                    "centroid": embed_i.copy(),
                    "rows": [row.to_dict()],
                    "n": 1,
                    "first_seen": pub_i,
                    "last_seen": pub_i
                }

                open_stories.append(new_story)
                article_story_ids.append((idx, new_story))
    
        best_story = None
        best_sim = -1
        MERGE_SIM = 0.65
        MERGE_GAP_DAYS = 14

        merged = True
        while merged:
            merged = False

            for i in range(len(open_stories)):
                if merged:
                    break

                s1 = open_stories[i]

                for j in range(i + 1, len(open_stories)):
                    s2 = open_stories[j]

                    # time constraint
                    if (
                        pd.notna(s1["last_seen"])
                        and pd.notna(s2["first_seen"])
                        and abs((s1["last_seen"] - s2["first_seen"]).days) > MERGE_GAP_DAYS
                    ):
                        continue
    
                    # semantic similarity
                    sim = float(np.dot(s1["centroid"], s2["centroid"]))
    
                    if sim >= MERGE_SIM:
                        # merge s2 into s1
                        n1, n2 = s1["n"], s2["n"]
                        centroid = (s1["centroid"] * n1 + s2["centroid"] * n2)
                        s1["centroid"] = centroid / np.linalg.norm(centroid)
                        s1["rows"].extend(s2["rows"])
                        s1["n"] = n1 + n2
                        s1["last_seen"] = max(s1["last_seen"], s2["last_seen"])
                        s1["first_seen"] = min(s1["first_seen"], s2["first_seen"])
    
                        # update article → story mapping
                        for k, (a_idx, sid) in enumerate(article_story_ids):
                            if article_story_ids[k][1] is s2:
                                article_story_ids[k] = (a_idx, s1)
    
                        open_stories.pop(j)

                    
                        merged = True
                        break
        story_id_map = {
        id(s): s["id"]
        for s in open_stories
        }
        for s in open_stories:
            rows = s["rows"]
            def canonical_key(r):
                title_ok = 1 if pd.notna(r.get("Title")) else 0
                content_len = len(str(r.get("Summary") or ""))
                return (title_ok, content_len)

            can_row = max(rows, key=canonical_key)

            canonical_title = f"Story {s['id']}" 

            story_rows.append({
                "story_id": s["id"],
                "canonical_title": canonical_title,
                "canonical_link": can_row["Link"],
                "canonical_published": can_row["Published_utc"],
                "article_count": len(rows),
                "first_seen": s["first_seen"],
                "last_seen": s["last_seen"]
            })
        new_stories_df = pd.DataFrame(story_rows)


        stories_df = stories_df.merge(
        new_stories_df, 
        on = 'story_id',
        how = 'outer',
        suffixes = ('', '_new')   
        )

        for col in ["canonical_title", "canonical_link", "canonical_published",
                "article_count", "first_seen", "last_seen"]:
            stories_df[col] = stories_df[col].fillna(stories_df[f"{col}_new"])
    
        stories_df = stories_df[
        [c for c in stories_df.columns if not c.endswith("_new")]
        ]
            

        aid = pd.DataFrame(
            [(idx, story_id_map[id(story_ref)])
            for idx, story_ref in article_story_ids],
            columns=["orig_idx", "story_id"]
        ).set_index("orig_idx")

        df_w = df.join(aid.rename(columns={"story_id": "story_id_new"}), how="left")
        df_w.loc[df_w['story_id'].isna(), 'story_id'] = df_w['story_id_new']
        df_w = df_w.drop(columns = ['story_id_new'])
        assert df_w['story_id'].notna().all()

        return df_w, new_stories_df, open_stories



    df['Published_utc'] = pd.to_datetime(df['Published_utc'], errors='coerce')
    #filtered_df = df.drop_duplicates(subset = 'Link', keep = 'last').reset_index(drop = True)
    articles_with_stories, stories_df, open_stories = build_story_clusters(new_articles, open_stories, story_id_counter, stories_df, min_sim = 0.6)
    articles_with_stories = pd.concat([already_labeled, articles_with_stories], ignore_index = True)

    #Saving centroids for future reference
    centroid_records = [{"story_id": s["id"], "centroid": s["centroid"].tolist()} for s in open_stories]
    atomic_write_pickle("pipeline/resources/story_centroids.pkl", centroid_records)
    
    assigned = articles_with_stories['story_id'].notna().sum()
    if assigned == 0:
        raise RuntimeError("CRITICAL: build_story_clusters produced zero story assignments.")
    articles_with_stories.to_csv("pipeline/resources/Articles_with_Stories.csv.gz", index = False, compression = 'gzip')

    df = articles_with_stories
    print(df.columns.tolist())
    merged = pd.merge(df, stories_df, on='story_id', how='left')
    merged = merged.sort_values(by='Published_utc', ascending=False)
    NUMERIC_COLS = [
    "Risk_Score",
    "Frequency_Score",
    "Acceleration_value",
    "Recency",
    "Source_Accuracy",
    "Impact_Score",
    "Industry_Risk",
    "Location"
]
    for c in NUMERIC_COLS:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors = 'coerce')
            
    grouped = merged.groupby('story_id')
    

    score_factors = []
    for story_id, group in grouped:
        if group.shape[0] >= 2:
            avg_risk_score = group['Risk_Score'].mean(skipna = True)
            avg_frequency = group['Frequency_Score'].mean(skipna = True)
            avg_acceleration = group['Acceleration_value'].mean(skipna = True)
            recency = (group.sort_values(by='Published_utc', ascending=False)['Recency'].dropna().iloc[0] if group['Recency'].notna().any() else np.nan)
            avg_source_acc = group['Source_Accuracy'].mean(skipna = True)
            avg_impact = group['Impact_Score'].mean(skipna = True)
            avg_industry = group['Industry_Risk'].mean(skipna = True) 
            avg_location = group['Location'].mean(skipna = True)

            if pd.isna(avg_risk_score):
                print(
                    f"Story {story_id} has no numeric Risk_Score:",
                    group["Risk_Score"].unique()[:5], flush = True
                )
    
            score_factors.append({
                "story_id": story_id,
                "avg_risk_score": avg_risk_score,
                "avg_frequency": avg_frequency,
                "avg_acceleration": avg_acceleration,
                "avg_recency":  recency,
                "avg_source_accuracy": avg_source_acc,
                "avg_impact_score": avg_impact,
                "avg_industry_risk": avg_industry,
                "avg_location": avg_location
            })
    
    if score_factors:
        score_df = pd.DataFrame(score_factors)
        stories_df = stories_df.merge(score_df, on="story_id", how="left")
        assert stories_df["canonical_title"].notna().all()
    
    stories_df.to_csv("pipeline/resources/Story_Clusters.csv.gz", index = False, compression = 'gzip')
    
    api_key = os.getenv('PAID_API_KEY')
    client = genai.Client(api_key = api_key)
    
    df = pd.read_csv('pipeline/resources/Articles_with_Stories.csv.gz', compression='gzip')
    df = df.drop_duplicates(subset=["Title", "Published_utc"], keep="last")
    if Path('pipeline/resources/dashboard_stories.csv.gz').exists():  
        df_stories = pd.read_csv('pipeline/resources/dashboard_stories.csv.gz', compression='gzip')
    else:
        df_stories = pd.DataFrame(columns=['story_id', 'canonical_title', 'canonical_source', 
                                            'last_seen', 'avg_risk_score', 'avg_recency'])
        df_stories.to_csv('pipeline/resources/Story_Clusters_backup.csv', index=False)
    
    
    if Path('pipeline/resources/Canonical_Stories_with_Summaries.csv').exists():
        canonical_titles = pd.read_csv('pipeline/resources/Canonical_Stories_with_Summaries.csv')
        canonical_titles['story_id'] = canonical_titles['story_id'].astype(int)
        stories_df = stories_df.merge(
            canonical_titles[['story_id', 'canonical_title', 'canonical_source']],
            on='story_id', how='left', suffixes=('', '_gemini'), validate='one_to_one'
        )
        if 'canonical_title_gemini' in stories_df.columns:
            stories_df['canonical_title'] = stories_df['canonical_title_gemini'].fillna(stories_df['canonical_title'])
        stories_df.drop(columns=[c for c in stories_df.columns if c.endswith('_gemini')], inplace=True, errors='ignore')
    
    if 'canonical_title_new' in df_stories.columns:
        df_stories['canonical_title'] = df_stories['canonical_title_new'].fillna(df_stories['canonical_title'])

    df_stories['canonical_title'] = df_stories['canonical_title'].astype('string')
    df_stories = (
        df_stories
        .sort_values("last_seen", ascending=False)
        .drop_duplicates(subset=["story_id"], keep="first")
    )
    
    
    df = df[['Title', 'story_id', 'Published_utc', 'Risk_Score', 'University Label']]
    df['Published_utc'] = pd.to_datetime(df['Published_utc'], errors='coerce', utc = True)
    cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=90)
    
    df = df[df['Published_utc'] > cutoff]
    
    
    
    STORY_AGG_COLS = [
        "avg_risk_score",
        "avg_frequency",
        "avg_acceleration",
        "avg_recency",
        "avg_source_accuracy",
        "avg_impact_score",
        "avg_industry_risk",
        "avg_location"
    ]
    df = df.drop(columns=STORY_AGG_COLS, errors="ignore")
    merged = df.merge(
        df_stories,
        on="story_id",
        how="left",
        validate="many_to_one"   
    )
    for col in ['canonical_source', 'canonical_title', 'avg_risk_score', 'avg_recency']:
        if col not in merged.columns:
            merged[col] = np.nan
    valid_story_ids = set(merged['story_id'].unique())
    df_stories = df_stories[df_stories['story_id'].isin(valid_story_ids)]
    grouped = merged.groupby('story_id')
    
    canonical_stories = []

    for story_id, group in grouped:
        canonical_source = group['canonical_source'].iloc[0]
        canonical_title = group['canonical_title'].iloc[0]
        if group.shape[0] >= 2:
            
            if pd.notna(canonical_title) and str(canonical_title).strip() not in ('', 'nan'):
                continue
            if 'University Label' in group.columns:
                if not (group['University Label'] == 1).any():
                    continue
            existing_title = group['canonical_title'].iloc[0]
            print(f"Story ID: {story_id}")
            titles = group['Title'].tolist()
            prompt = f"For each of these groups, read the titles of all the articles and generate ONLY ONE concise canonical title that summarizes the main topic of the story. No bullets. No preface. No list. Here's the titles: {titles}"
            prompt2 = f"Based on the following titles: {titles}, generate a 100 word summary that takes into account all the titles. No bullets. No headings."
            max_attempts = 6
            response = None
            response2 = None
            for attempt in range(max_attempts):
                try:
                    response = client.models.generate_content(
                        model = 'gemini-2.5-flash',
                        contents = prompt)
                    response2 = client.models.generate_content(
                        model = 'gemini-2.5-flash',
                        contents = prompt2
                        )
                    
        
                    title_txt = getattr(response, "text", None)
                    if not title_txt:
                        try:
                            title_txt = response.candidates[0].content.parts[0].text
                        except Exception:
                            title_txt = None
                    
                    summary_txt = getattr(response2, "text", None)
                    if not summary_txt:
                        try:
                            summary_txt = response2.candidates[0].content.parts[0].text
                        except Exception:
                            summary_txt = None
                    
                    # ---- validate ----
                    if title_txt and str(title_txt).strip():
                        canonical_title = str(title_txt).strip()
                        summary = str(summary_txt).strip() if (summary_txt and str(summary_txt).strip()) else ""
                        break
                    
                    raise RuntimeError("Gemini returned empty/None text")
                except Exception as e:
                    s = str(e).lower()
            
                    # quota / 429 handling
                    if ("resource_exhausted" in s) or ("quota" in s) or ("429" in s):
                        wait_time = 60
                        retry_delay_match = re.search(r"retrydelay['\"]?:\s*['\"]?(\d+)s", str(e), flags=re.I)
                        if retry_delay_match:
                            wait_time = int(retry_delay_match.group(1))
                        print(f"⚠️ Gemini quota hit. Retrying in {wait_time}s... (attempt {attempt+1}/{max_attempts})", flush=True)
                        time.sleep(wait_time)
                        continue
            
                    # transient-ish
                    wait = 2 ** attempt
                    print(f"⚠️ Gemini error: {e}. Retrying in {wait}s... (attempt {attempt+1}/{max_attempts})", flush=True)
                    time.sleep(wait)
            
            else:
    # hard fallback so you NEVER crash on .strip()
                canonical_title = f"Story {story_id}"
                summary = ""
                print(f"⚠️ Gemini failed after {max_attempts} attempts for story {story_id}. Using fallback title.", flush=True)
        
            print(f"Canonical Title: {canonical_title}")
            print(f"Summary: {summary}")
            print(f"Average Risk Score: {group['avg_risk_score'].iloc[0]:.2f}")
            print(f"Average Recency: {group['avg_recency'].iloc[0]:.2f}")
            print("Articles:")
            for _, row in group.iterrows():
                print(f" - {row['Published_utc']}: {row['Title']}")
            print("\n")
           
            canonical_stories.append({
                "story_id": story_id,
                "canonical_title": canonical_title,
                "summary": summary,
                "average_risk_score": group['avg_risk_score'].iloc[0],
                "average_recency": group['avg_recency'].iloc[0],
                "articles": [{row['Published_utc']: row['Title']} for _, row in group.iterrows()],
                "canonical_source": 'gemini'
            })
            print(f"Gemini Check story ={story_id}, title {canonical_title}, source {canonical_source}", flush = True)
        
    if canonical_stories:
        new_df = pd.DataFrame(canonical_stories)
        try:
            existing = pd.read_csv("pipeline/resources/Canonical_Stories_with_Summaries.csv")
            final = pd.concat([existing, new_df], ignore_index=True)
            final = final.drop_duplicates(subset=['story_id'], keep='last')
            final.to_csv("pipeline/resources/Canonical_Stories_with_Summaries.csv", index=False)
        except FileNotFoundError:
            final = new_df
            final.to_csv("pipeline/resources/Canonical_Stories_with_Summaries.csv", index=False)
    else:
        print("No new titles were generated.")
    
    return None

#Show the articles over time
stories = build_stories()
def safe_mode(series):
    s = series.dropna()
    return s.mode().iloc[0] if not s.empty else None
articles = pd.read_csv("pipeline/resources/Articles_with_Stories.csv.gz", compression = 'gzip')
score_cols = [
        "avg_risk_score",
        "avg_frequency",
        "avg_acceleration",
        "avg_recency",
        "avg_source_accuracy",
        "avg_impact_score",
        "avg_industry_risk",
        "avg_location"
    ]
existing = [c for c in score_cols if c in articles.columns]
articles[existing] = articles[existing].apply(pd.to_numeric, errors = 'coerce')
story_scores = (articles.groupby("story_id").agg(
    avg_frequency = ("Frequency_Score", "mean"),
    avg_acceleration = ("Acceleration_value", "max"),
    avg_source_accuracy = ("Source_Accuracy", "mean"),
    avg_impact_score = ("Impact_Score", "mean"),
    avg_industry_risk = ("Industry_Risk", "mean"),
    avg_location = ("Location", "mean"),
    risk_label = ("Predicted_Risks_new", safe_mode)).reset_index())
canonical = pd.read_csv("pipeline/resources/Canonical_Stories_with_Summaries.csv")
canonical = canonical.merge(story_scores, on = "story_id", how = 'left', validate= "one_to_one")
canonical.to_csv("pipeline/resources/Canonical_stories_with_Summaries.csv", index = False)
articles = load_full_topics(download_file('latest/BERTopic_results2.csv.gz', 'pipeline/resources/BERTopic_results2.csv.gz'))
articles = ensure_risk_scores(articles)
articles = articles.drop_duplicates(subset = ['Title', 'Link'], keep = 'last')
article_story_map = pd.read_csv("pipeline/resources/Articles_with_Stories.csv.gz", compression = 'gzip')
article_story_map = article_story_map.drop_duplicates(subset = ['Title', 'Link'], keep = 'last')
canonical = pd.read_csv("pipeline/resources/Canonical_Stories_with_Summaries.csv")
score_cols = ["avg_risk_score", "avg_frequency", "avg_recency"]
stories_df = pd.read_csv(
    "pipeline/resources/Story_Clusters.csv.gz",
    compression="gzip"
)
canonical = canonical.merge(stories_df[["story_id"]], on = "story_id", how = "left")
articles = articles.merge(article_story_map[['Title', 'Link', 'story_id']], on =['Title','Link'], how='left', validate='many_to_one')
story_sizes = (articles.groupby("story_id").size().rename("story_articles_count").reset_index())

articles = articles.merge(canonical, on = "story_id", how = 'left', validate = 'many_to_one')
articles = articles.merge(story_sizes, on = "story_id", how = 'left')

canonical_articles = articles[articles['story_articles_count'] >= 2].copy()




dashboard_stories = (
    canonical_articles
      .groupby("story_id")
      .agg(
          canonical_title = ("canonical_title", "first"),
          summary = ("summary", "first"),
          article_count = ("story_articles_count", "first"),
          avg_risk_score = ("Risk_Score", "mean"),
          avg_frequency = ("Frequency_Score", "mean"),
          avg_recency = ("Recency", "mean"),
          avg_acceleration = ("Acceleration_value", "max"),
          avg_source_accuracy = ("Source_Accuracy", "mean"),
          avg_impact_score = ("Impact_Score", "mean"),
          avg_industry_risk = ("Industry_Risk", "mean"),
          avg_location = ("Location", "mean"),

          risk_label = ("Predicted_Risks_new", safe_mode),
          last_seen = ("Published_utc", "max")
      )
      .reset_index()
)


dropdown_table = canonical_articles[["story_id", "Title","Topic", "Link", "Published_utc", "Risk_Score",'Recency', 'Source_Accuracy', 'Impact_Score', 'Acceleration_value', 'Location','Industry_Risk', 'Frequency_Score', "Predicted_Risks_new"]].sort_values("Published_utc", ascending = False)
standalone_articles = articles[articles["story_articles_count"] == 1].copy()

dashboard_stories.to_csv("pipeline/resources/dashboard_stories.csv.gz", compression = 'gzip')
dropdown_table.to_csv("pipeline/resources/dashboard_dropdown.csv.gz", compression = 'gzip')
standalone_articles.to_csv("pipeline/resources/dashboard_articles.csv.gz", compression = 'gzip')

articles_only = articles[articles['story_articles_count']<3].copy()

def build_subtopic_clusters(df, subtopics, model, min_sim=0.6, subtopic_centroids = None):
    df = df.copy()

    

    df['Published_utc'] = pd.to_datetime(df['Published_utc'], errors = 'coerce', utc = True)
    df = df.dropna(subset= ['Published_utc'])

    subtopics['Published_utc'] = pd.to_datetime(subtopics['Published_utc'], errors = 'coerce', utc = True)
    subtopics = subtopics.dropna(subset = ['Published_utc'])


    df['Window'] = (df['Published_utc'].dt.to_period('W-MON').dt.start_time)

    df['Window'] = df['Window'].dt.tz_localize('UTC')


    df = df.drop_duplicates(subset = ['Title'], keep = 'last')
    clusters_meta = (
        subtopics.groupby('Cluster').agg(
            Last_Seen = ('Published_utc', 'max'),
            Event_Label = ('Event_Label', 'first')
        )
        .reset_index()
    )

    current_time = df['Published_utc'].max()

    active_clusters = set(clusters_meta[
        (current_time - clusters_meta['Last_Seen']).dt.days <= 60
    ]['Cluster'])

    

    

    df = df.merge(subtopics[['Title', 'Link','Cluster','Event_Severity','Event_Label']], on =['Title', 'Link'], how = 'left')
    df['Embeddings'] = list(model.encode(
        (df['Title'].fillna('') + ' ' + df['Content'].fillna('')).tolist(),
        show_progress_bar=True,
        convert_to_numpy=True
    ))
    if subtopic_centroids is not None and len(subtopic_centroids) > 0:
        centroids = subtopic_centroids
    else:
        if not subtopics.empty:
            print("Building subtopic centroids from scratch...", flush=True)
            sub_embs = model.encode(
                (subtopics['Title'].fillna('') + ' ' + subtopics['Content'].fillna('')).tolist(),
                convert_to_numpy=True
            )
            subtopics = subtopics.copy()
            subtopics['_emb'] = list(sub_embs)
            centroids = (
                subtopics[subtopics['Cluster'].isin(active_clusters)]
                .groupby('Cluster')['_emb']
                .apply(lambda x: np.mean(np.vstack(x), axis=0))
            )
        else:
            centroids = pd.Series(dtype=object)

    if centroids.empty:
        df['Cluster'] = df['Cluster'].fillna(-1)
        return df, centroids

    
    def cluster_embeddings(embeddings, threshold=0.60):
        N = embeddings.shape[0]
        visited = [False] * N
        clusters = []

        chunks = 512

        for i in range(N):
            if not visited[i]:
                cluster = []
                queue = [i]
                visited[i] = True
            
                while queue:
                    node = queue.pop(0)
                    cluster.append(node)
                
                    node_vec = embeddings[node].reshape(1, -1)
                    unvisited_idx = [j for j in range(N) if not visited[j]]

                    for chunk_start in range(0, len(unvisited_idx), chunks):
                        chunk_idx = unvisited_idx[chunk_start:chunk_start + chunks]
                        chunk_vecs = embeddings[chunk_idx]
                        sims = (chunk_vecs @ node_vec.T).ravel()

                        for k, j in enumerate(chunk_idx):
                            if sims[k] > threshold:
                                visited[j] = True
                                queue.append(j)
                clusters.append(cluster)
        return clusters
    if len(centroids) == 0:
        df['Cluster'] = df['Cluster'].fillna(-1)
    else: 
        for idx, row in df[df['Cluster'].isna()].iterrows():
            if not isinstance(row['Embeddings'], np.ndarray):
                text = f"{row.get('Title','')} {row.get('Content','')}"
                emb = model.encode(text, convert_to_numpy=True)
                df.at[idx, 'Embeddings'] = emb
            else:
                emb = row['Embeddings']
            sims = cosine_similarity(emb.reshape(1, -1), np.vstack(centroids.values))[0]

            if sims.max() >= min_sim:
                assigned = centroids.index[sims.argmax()]
                df.at[idx, 'Cluster'] = assigned
                df.at[idx, 'Event_Label'] = subtopics.loc[subtopics['Cluster'] == assigned, 'Event_Label'].iloc[0]
            else:
                df.at[idx, 'Cluster'] = -1
        
    leftovers = df[df['Cluster'] == -1]

    if len(leftovers) > 1:
        bad = leftovers['Embeddings'].apply(lambda x: not isinstance(x, np.ndarray))
        if bad.any():
            raise ValueError("Some embeddings are not numpy arrays.")
        X = normalize(np.vstack(leftovers['Embeddings']))
        sub_clusters = cluster_embeddings(X, threshold=0.6)
        next_cluster_id = int(df['Cluster'].max()) + 1

        for group in sub_clusters:
            df.loc[leftovers.index[group], 'Cluster'] = next_cluster_id
            next_cluster_id = next_cluster_id + 1
    
    for cluster_id, group in df.groupby('Cluster'):
        has_university = (group['University Label'] == 1).any()
        if cluster_id == -1:
            continue
        existing_label = group['Event_Label'].dropna()
        existing_label = existing_label[existing_label.str.strip() != '']
        if len(existing_label) > 0:
            continue

        titles = group['Title'].dropna().tolist()

        # If only one article, just use its title
        if len(titles) == 1 or not has_university:
            label = titles[0]

        else:
            combined_text = ' | '.join(titles)  # cap for safety

            prompt = (
                "Given the following news article titles, provide ONE concise label "
                "that summarizes the shared event or topic. "
                "Do NOT include explanations.\n\n"
                f"{combined_text}"
            )

            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[prompt]
            )

            label = re.sub(r'[\n"]+', ' ', response.text).strip()[:120]

        df.loc[group.index, 'Event_Label'] = label

    for (window, risk, cluster), group in df.groupby(['Window', 'Predicted_Risks_new', 'Cluster']):
        risk_scores = pd.to_numeric(group['Risk_Score'], errors='coerce').fillna(0)
        event_severity = risk_scores.mean()
        df.loc[group.index, 'Event_Severity'] = event_severity
    for idx, row in df[df['Cluster'].notna() & (df['Cluster'] != -1)].iterrows():
        cid = row['Cluster']
        emb = row['Embeddings']
        if not isinstance(emb, np.ndarray):
            continue
        if cid in centroids.index:
            # Running mean update
            old = centroids[cid]
            centroids[cid] = (old + emb) / 2
        
    # Compute centroids for brand new leftover clusters
    new_cluster_ids = set(df['Cluster'].unique()) - set(centroids.index) - {-1}
    if new_cluster_ids:
        new_centroid_records = {}
        for cid in new_cluster_ids:
            cluster_embs = np.vstack(
                df[df['Cluster'] == cid]['Embeddings'].values
            )
            new_centroid_records[int(cid)] = cluster_embs.mean(axis=0)
        new_series = pd.Series(new_centroid_records)
        centroids = pd.concat([centroids, new_series])

    return df, centroids

articles = load_full_topics(download_file('latest/BERTopic_results2.csv.gz', 'pipeline/resources/BERTopic_results2.csv.gz'))

nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer('all-MiniLM-L6-v2')
if Path('pipeline/resources/subtopics.csv').exists():
    subtopics = pd.read_csv('pipeline/resources/subtopics.csv')
else:
    subtopics = pd.DataFrame(columns=['Title', 'Link', 'Cluster', 'Event_Severity', 'Event_Label'])


# Split into already labeled and new
already_clustered = articles[articles['Title'].isin(subtopics['Title'])]
already_clustered['Published_utc'] = pd.to_datetime(already_clustered['Published_utc'], errors='coerce', utc=True)
already_clustered['Window'] = (
    already_clustered['Published_utc']
    .dt.to_period('W-MON')
    .dt.start_time
    .dt.tz_localize('UTC')
)
new_only = articles[
    ~articles['Title'].isin(subtopics['Title']) & 
    (articles['University Label'] == 1)
].copy()

print(f"Already clustered: {len(already_clustered)}, New: {len(new_only)}", flush=True)
if Path('pipeline/resources/subtopic_centroids.pkl').exists():
    with open('pipeline/resources/subtopic_centroids.pkl', 'rb') as f:
        saved = pickle.load(f)
    subtopic_centroids = pd.Series({k: np.array(v) for k, v in saved.items()})
else:
    subtopic_centroids = None
articles, updated_centroids = build_subtopic_clusters(new_only, subtopics, model, subtopic_centroids=subtopic_centroids)

updated_subtopics = pd.concat([subtopics, articles[['Title', 'Link','Cluster', 'Event_Severity', 'Event_Label', 'Published_utc']]],
                             ignore_index = True).drop_duplicates(subset = ['Title', 'Link'], keep = 'last')

updated_subtopics.to_csv('pipeline/resources/subtopics.csv', index = False)
print(f"Saved {len(updated_subtopics)} total subtopics ({len(updated_subtopics) - len(subtopics)} new)", flush=True)

if subtopic_centroids is not None:
    merged_centroids = {int(k): v.tolist() for k, v in subtopic_centroids.items()}
else:
    merged_centroids = []

if updated_centroids is not None:
    for k, v in updated_centroids.items():
        merged_centroids[int(k)] = v.tolist() if isinstance(v, np.ndarray) else v
    

atomic_write_pickle('pipeline/resources/subtopic_centroids.pkl', merged_centroids)
already_clustered = already_clustered.merge(subtopics[['Title', 'Link', 'Cluster', 'Event_Label', 'Event_Severity']],
    on=['Title', 'Link'],
    how='left'
)
articles = pd.concat([already_clustered, articles], ignore_index=True)


atomic_write_csv("pipeline/resources/BERTopic_Streamlit.csv.gz", articles, compress = True)
upload_file('pipeline/resource/BERTopic_Streamlit.csv.gz', 'latest/BERTopic_Streamlit.csv.gz')


df = articles
df['Published_utc'] = pd.to_datetime(df['Published_utc'], errors = 'coerce')
df.sort_values('Published_utc', inplace = True)
df['Content_trunc'] = df['Content'].fillna('').str.slice(0, 500)
df = df[df['Predicted_Risks_new'] != 'No Risk'].copy()

bundle = load_model_bundle(Github_owner, Github_repo, 'regression')
risk_defs = {
  "Research Funding Disruption": "University research funding halted or withdrawn. Grant cuts, pauses, or canceled awards stop lab projects and furlough staff.",
  "Enrollment Pressure": "Fewer applications or lower student retention reduce tuition revenue. Admissions decline or FAFSA delays cause enrollment stress.",
  "Policy or Political Interference": "State or federal officials intervene in campus DEI or curriculum policies through mandates or funding threats.",
  "Institutional Alignment Risk": "University units pursue conflicting goals. Misaligned strategies or budgets stall progress on institutional plans.",
  "Mission Drift": "University focuses on revenue or prestige projects over teaching and research, weakening academic mission.",
  "Revenue Loss": "University faces financial shortfall due to budget cuts, declining tuition, or reduced auxiliary income.",
  "Insurance Market Volatility": "University insurance premiums rise or coverage is reduced after market hardening or claims disputes.",
  "Unexpected Expenditures": "Campus hit by sudden unplanned costs from facility failures, legal settlements, or emergency repairs.",
  "Endowment Risk": "University endowment loses value or liquidity, forcing payout cuts that affect scholarships or operations.",
  "Infrastructure Failure": "Campus systems fail. Power, HVAC, or network outages disrupt classes and research activity.",
  "Vendor Cyber Exposure": "University data exposed through a third-party SaaS or vendor security breach or SOC 2 gap.",
  "Unauthorized Access/Data Breach": "Hackers access internal university systems or personal records, requiring breach response.",
  "Artificial Intelligence Ethics & Governance": "Campus adoption of AI raises fairness, bias, or transparency issues needing governance policy.",
  "Rapid Speed of Disruptive Innovation": "Ability to develop a systematic approach to the identification of ever-increasing categories of AI risks; ability to adopt artificial intelligence systems to supplement existing equipment and people; Software systemfailure, cybersecurity issues with potential hacking of AI control systems",
  "Controversial Public Incident": "Campus statement, protest, or viral video sparks public backlash and reputational scrutiny.",
  "DEI Program Backlash": "Political or donor pressure challenges diversity and inclusion programs on campus.",
  "Leadership Missteps": "University leaders issue misleading statements or mishandle a crisis, prompting criticism or resignations.",
  "High-Profile Litigation": "University faces a lawsuit drawing major public or media attention, often around discrimination or research conduct.",
  "Violence or Threats": "Campus shooting, assault, or credible threat causes lockdowns or safety alerts for students and staff.",
  "Emergency Preparedness Gaps": "Campus emergency plans prove outdated or untested, delaying response during a crisis.",
  "Infectious Disease Outbreak": "Cluster of student or staff illness disrupts classes or triggers campus health measures.",
  "Lab Incident": "Chemical spill or fire in a university lab injures staff or halts research pending investigation.",
  "Environmental Exposure": "Hazardous materials like asbestos, lead, or mold found in campus buildings trigger closures.",
  "Hurricane/Flood/Wildfire": "Natural disaster damages campus property and displaces students or staff.",
  "Student Conduct Incident": "Fraternity hazing, fights, or misconduct lead to discipline, injuries, or suspension.",
  "Academic Disruption": "Classes canceled or delayed due to strikes, outages, or emergencies on campus.",
  "Mental Health Crises": "Campus counseling overwhelmed by student mental health emergencies or suicide risk.",
  "HR Complaint": "Employee alleges harassment, discrimination, or retaliation, leading to internal investigation.",
  "Labor Dispute": "Faculty or staff strike or protest disrupts classes and campus operations.",
  "Whistleblower Claims": "Employee reports internal fraud or safety cover-up, prompting investigation or retaliation concerns.",
  "Accreditation Risk": "Accreditor flags weaknesses in governance, finances, or academic outcomes, threatening status.",
  "No Risk": "Not relevant to higher education. Not relevant to Tulane University. Not relevant to university.",
  "Constant Inflation": "Sustained increases in operating costs, wages, or materials erode budget stability and financial planning.",
  "Morale challenges": "Persistent employee dissatisfaction, burnout, or disengagement reduces productivity and retention.",
  "Transportation/Access Disruption": "Interruptions to transit, parking, or campus access hinder attendance, operations, or service delivery.",
  "IT System Failure":  "Critical technology outage or infrastructure breakdown disrupts academic, financial, or administrative functions.",
  "Title IX/ADA Noncompliance": "Failure to meet federal civil rights obligations exposes institution to legal action and regulatory penalties.",
  "Housing/Food Insecurity": "Students lack stable housing or reliable nutrition, affecting retention, performance, and wellbeing.",
  "Policy Misapplication": "Inconsistent or improper enforcement of institutional policies creates fairness, legal, or reputational risks.",
  "Nepotism/Conflict of Interest": "Personal relationships or undisclosed interests influence hiring, contracts, or governance decisions.",
  "Grant Mismanagement": "Improper allocation, reporting, or oversight of grant funds jeopardizes funding and compliance status.",
  "Extreme Weather Events": "Severe storms, flooding, or climate events disrupt campus operations and damage infrastructure.",
  "FERPA/HIPAA Violations": "Unauthorized disclosure of protected student or health information triggers legal and regulatory consequences.",
  "Media campaigns": "Coordinated media coverage or social pressure amplifies reputational exposure and stakeholder scrutiny.",
  "Ransomware/Malware": "Malicious software encrypts or corrupts institutional systems, demanding payment or disrupting operations.",
  "Credential Phishing": "Deceptive communications obtain login credentials, enabling unauthorized system access or data breach.",
  "Workplace Safety Violation": "Failure to meet occupational safety standards results in injury, fines, or regulatory enforcement.",
  "Audit Findings": "Internal or external audit reveals financial irregularities, compliance gaps, misuse of funds, or control weaknesses in university operations.",
  "Supply Chain Disruption": "Inadequate supply chain management; Inability to comply with contract terms and conditions involving insurance requirements and termination clauses; Immature and decentralized process for vendor oversight; Lack of leverage for buying power; Lack of training to the campus community on supply chain processes and the importance of stewardship of the process.",
  "Cloud Misconfiguration": "Improperly configured cloud storage, access controls, or infrastructure settings expose sensitive university data or systems to unauthorized access.",
  "AI Ethics and Governance": "University adoption, use, or policy development around artificial raises concerns about academic integrity, data privacy, intellectual property.",
  "Climate Infrastructure Risks": "University buildings, utilities, or campus infrastructure are vulnerable to long-term climate related deterioration.",
  "Environmental Noncompliance": "University fails to meet federal, state, or local environmental regulations governing waste disposal, air quality, water discharge, hazardous materials handling, or emissions.",
  "Accessibility Barriers": "Barriers prevent students, faculty, or staff with disabilities from fully accessing campus facilities, online systems, academic programs, or university services, triggering ADA compliance concerns or student grievances.",
  "Faculty conflict": "Interpersonal or professional disputes among faculty members, including disagreements over governance, resource allocation, academic freedom, or departmental leadership that may disrupt operations, damage collegial relationships, or escalate into formal grievances or public controversy."
    
    




}
print(df.columns.tolist(), flush = True)

grouped = df.groupby(['Window', 'Cluster', 'Predicted_Risks_new']).agg({
    "Title": ' '.join,
    "Content_trunc": ' '.join,
    "Event_Severity": 'max',
    "Event_Label": 'first',
    "Acceleration_value": 'last',
    "Recency": "last",
    "Source_Accuracy": "mean",
    "Impact_Score": "mean",
    "Location": 'mean',
    "Industry_Risk": "mean",
    "Frequency_Score": "mean"
}).reset_index()

grouped['combined_text'] = (grouped['Title'].fillna('') + grouped['Content_trunc'].fillna(''))

model = SentenceTransformer('all-mpnet-base-v2')

risk_names = list(risk_defs.keys())
risk_labels = list(risk_defs.values())

risk_embeddings = model.encode(risk_labels, normalize_embeddings = True, show_progress_bar = True)
text_embeddings = model.encode(grouped['combined_text'].tolist(), normalize_embeddings = True, show_progress_bar = True)
risk_to_index = {name: i for i, name in enumerate(risk_defs.keys())}

similarities = []
for i, row in grouped.iterrows():
    risk_name = row['Predicted_Risks_new']
    risk_idx = risk_to_index[risk_name]

    sim = np.dot(text_embeddings[i], risk_embeddings[risk_idx])
    sim = np.clip(sim, 0, 1)
    similarities.append(sim)

grouped['similarity'] = similarities

s_min = grouped['similarity'].quantile(0.2)
s_max = grouped['similarity'].quantile(0.9)

grouped['rel_cos'] = np.clip((grouped['similarity'] - s_min)/(s_max - s_min), 0, 1)


sim_matrix = np.dot(text_embeddings, risk_embeddings.T)
pred_idx = grouped['Predicted_Risks_new'].map(risk_to_index).values
chosen_sim = sim_matrix[np.arange(len(grouped)), pred_idx]

sim_excluding_chosen = sim_matrix.copy()
sim_excluding_chosen[np.arange(len(grouped)), pred_idx] = -1
best_alt_sim = sim_excluding_chosen.max(axis = 1)

margin = chosen_sim - best_alt_sim

grouped['margin'] = margin

margin_cap = 0.2

tau = 0.05
grouped['rel_margin'] = 1/(1 + np.exp(-(grouped['margin']/tau)))


relevance = (0.7 * grouped['rel_cos']) + (0.3 * grouped['rel_margin'])

grouped['raw_score'] = grouped['Event_Severity'] * relevance

grouped = grouped.sort_values(['Window', 'Predicted_Risks_new', 'raw_score'], ascending = [True, True, False])

grouped['rank'] = grouped.groupby(['Window', 'Predicted_Risks_new']).cumcount() + 1

lam = 0.7

grouped['decay_weight'] = lam ** (grouped['rank'] - 1)

grouped['weighted_strength'] = (grouped['decay_weight'] * grouped['raw_score'])
grouped.to_csv('grouped_risk_scores1.csv', index = False)
K = 3
grouped_ranked = grouped.copy()
grouped_ranked = grouped_ranked[grouped_ranked['rank'] <= K]
grouped_ranked.to_csv('ranked_events_risks1.csv', index = False)

risk_scores = (grouped_ranked.groupby(['Window', 'Predicted_Risks_new'])['weighted_strength']
               .sum()
               .reset_index()
               .rename(columns = {'weighted_strength': 'raw_risk_score'})
)


c = 2.5
risk_scores['final_risk_score'] = (5 * risk_scores['raw_risk_score'] / (risk_scores['raw_risk_score'] + c))

risk_scores.to_csv('pipeline/resources/final_risk_scores1.csv', index = False)

def risk_weights_second_pass(df):
    base = df.copy()
    for col in ['Title','Content','Source']:
        if col not in base.columns:
            base[col] = ''
    base['Title'] = base['Title'].fillna('').astype(str)
    base['Content'] = base['Content'].fillna('').astype(str)
    base['Source'] = base['Source'].fillna('').astype(str)
    def _coerce_pub(x): 
        if pd.isna(x): 
            return pd.NaT 
        if isinstance(x, (int, float)): 
            if x > 1e12: # epoch ms 
                return pd.to_datetime(x, unit='ms', errors='coerce', utc=True) 
            if x > 1e9: # epoch s 
                return pd.to_datetime(x, unit='s', errors='coerce', utc=True) 
        sx = str(x) 
        sx = re.sub(r'\s(EST|EDT|PDT|CDT|MDT|GMT)\b', '', sx, flags=re.I) 
        return pd.to_datetime(sx, errors='coerce', utc=True) 
    if 'Published' not in base.columns: 
        base['Published'] = pd.NaT 
    base['Published'] = base['Published'].apply(_coerce_pub) 
    if pd.api.types.is_datetime64tz_dtype(base['Published']): 
        base['Published'] = base['Published'].dt.tz_convert('UTC').dt.tz_localize(None) 

    risk_half_life = { 
        "Research Funding Disruption": 60, 
        "Enrollment Pressure": 60, 
        "Policy or Political Interference": 90, 
        "Institutional Alignment Risk": 60, 
        "Mission Drift": 90, 
        "Revenue Loss": 90, 
        "Insurance Market Volatility": 90, 
        "Unexpected Expenditures": 15, 
        "Endowment Risk": 30, 
        "Constant Inflation": 15, 
        "Infrastructure Failure": 15, 
        "Transportation/Access Disruption": 7, 
        "Supply Chain Delay": 15, 
        "Emergency Preparedness Gaps": 15, 
        "Title IX/ADA Noncompliance": 30, 
        "Accreditation Risk": 120, 
        "FERPA/HIPAA Violations": 7, 
        "Grant Mismanagement": 7, 
        "Audit Findings": 30, 
        "Unauthorized Access/Data Breach": 7, 
        "Credential Phishing": 7, 
        "Vendor Cyber Exposure": 7, 
        "Cloud Misconfiguration": 7, 
        "Artificial Intelligence Ethics & Governance": 7, 
        "Rapid Speed of Disruptive Innovation": 90, 
        # --- Reputational and Social --- 
        "Controversial Public Incident": 30, 
        "DEI Program Backlash": 30, 
        "High-Profile Litigation": 90, 
        "Leadership Missteps": 30, 
        "Media Campaigns": 15, 
        # --- Health, Safety and Security --- 
        "Violence or Threats": 10, 
        "Infectious Disease Outbreak": 30, 
        "Lab Incident": 7, 
        "Workplace Safety Violation": 7, 
        "Environmental Exposure": 30, 
        # --- Environmental & Climate --- 
        "Hurricane/Flood/Wildfire": 30, 
        "Extreme Weather Events": 30, 
        "Climate Infrastructure Risks": 15, 
        "Environmental Noncompliance": 90, 
        "Insurance Withdrawal": 120, 
        # --- Student Experience & Welfare --- 
        "Mental Health Crises": 15, 
        "Housing/Food Insecurity": 15, 
        "Academic Disruption": 15, 
        "Student Conduct Incident": 7, 
        "Accessibility Barriers": 15, 
        # --- Internal Organization --- 
        "HR Complaint": 15, 
        "Labor Dispute": 30, 
        "Morale challenges": 30, 
        "Faculty conflict": 15, 
        "Executive Board conflicts": 30, 
        "Nepotism/Conflict of Interest": 15, 
        "Policy Misapplication": 15, 
        "Whistleblower Claims": 30 } 
    cand = base.get('Predicted_Risks_new', pd.Series('', index=base.index)).fillna('')
    cand = np.where(cand=='', base.get('Predicted_Risks_new', ''), cand)
    cand = pd.Series(cand, index=base.index).fillna('').astype(str)
    
    def _first_label(s):
        s = s.strip()
        if not s:
            return ''
        s = re.sub(r'^\[|\]$', '', s)
        s = re.split(r'[;,]', s)[0].strip().strip("'\"")
        return s
    
    if 'Cluster' not in base.columns:
        base['Cluster'] = -1
    base['Cluster'] = base['Cluster'].fillna(-1)
    def recency_features_topic_risk(df, now=None):
        fx = df.copy()

        required = {'Cluster', 'Predicted_Risks_new', 'Published_utc', 'Days_Ago'}
        if not required.issubset(fx.columns) or fx.empty:
            return pd.DataFrame(columns=['Cluster','Published_utc','last_seen_days','decayed_volume','recency_score_tr'])

        if now is None:
            now = pd.Timestamp.utcnow()

        art_w = 1.0
        if 'Impact_Score' in fx.columns:
            art_w = pd.to_numeric(fx['Impact_Score'], errors='coerce').fillna(0.0).clip(0, 1)

        def half_life(risk):
            return risk_half_life.get(risk, 30)

        hl  = fx['Predicted_Risks_new'].map(lambda r: max(1.0, half_life(r)))
        lam = np.log(2.0) / hl
        w_decay = np.exp(-lam * fx['Days_Ago'])
        fx['_w'] = w_decay * art_w

        grp = fx.groupby(['Cluster', 'Predicted_Risks_new'], dropna=False)
        out = grp.agg(
            last_seen=('Days_Ago', 'min'),
            decayed_volume=('_w', 'sum'),
            mentions=('Published', 'count')
        ).reset_index()

        out['hl'] = out['Predicted_Risks_new'].map(lambda r: max(1.0, half_life(r)))
        out['freshness'] = np.exp(-np.log(2.0) * (out['last_seen'] / out['hl']))

        def _safe_minmax(s):
            rng = s.max() - s.min()
            return (s - s.min()) / (rng + 1e-12)

        out['decayed_z'] = out.groupby('Predicted_Risks_new')['decayed_volume'].transform(_safe_minmax)

        w_fresh, w_vol = 0.6, 0.4
        out['recency_score_tr'] = (w_fresh * out['freshness'] + w_vol * out['decayed_z']).clip(0, 1)
        out = out.rename(columns={'last_seen': 'last_seen_days'})
        return out[['Cluster','Predicted_Risks_new','last_seen_days','decayed_volume','recency_score_tr']]


    def attach_topic_risk_recency(df):
        tr = recency_features_topic_risk(df)
        for c in ['last_seen_days','decayed_volume','recency_score_tr']:
            if c not in tr.columns:
                tr[c] = np.nan
        cols_to_drop = ["Predicted_Risks_new", "last_seen_days","decayed_volume",
                    "recency_score_tr","recency_score_tr_x","recency_score_tr_y"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

        tr_small = tr[["Cluster","Predicted_Risks_new","last_seen_days","decayed_volume","recency_score_tr"]].rename(
            columns={"recency_score_tr": "recency_score_tr_tr"}
        )
        overlap = [c for c in tr_small.columns if c in df.columns and c!= 'Cluster']
        if overlap:
            df = df.drop(columns = overlap)
        enriched = df.merge(tr_small, on="Cluster", how="left")

        days = pd.to_numeric(enriched.get('Days_Ago', np.nan), errors='coerce').astype(float)
        enriched['article_freshness'] = np.exp(-np.log(2.0) * (days / 14.0)).fillna(0.0)

        if 'recency_score_tr_tr' not in enriched.columns:
            enriched['recency_score_tr_tr'] = 0.0

        alpha = 0.7
        enriched['Recency_TR_Blended'] = (
            alpha * enriched['recency_score_tr_tr'].fillna(0.0)
            + (1 - alpha) * enriched['article_freshness']
        ).clip(0, 1)

        return enriched
    

    

    print("attach_topic_risk_recency() start", flush = True)
    base = attach_topic_risk_recency(base) 
    base['Recency'] = (base['Recency_TR_Blended'] * 5).round(2)

    return base

articles = df

risk_weights_second_pass(articles)
atomic_write_csv("pipeline/resources/BERTopic_Streamlit.csv.gz", articles, compress = True)
upload_file('pipeline/resource/BERTopic_Streamlit.csv.gz', 'latest/BERTopic_Streamlit.csv.gz')

print("Articles over time", flush = True)
#
track_over_time(df)
