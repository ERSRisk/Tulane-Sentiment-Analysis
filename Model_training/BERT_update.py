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
from google.genai.errors import APIError
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
import hashlib

rss_url = "https://github.com/ERSRisk/Tulane-Sentiment-Analysis/releases/download/rss_json/all_RSS.json.gz"

DIR_URL  = "https://github.com/ERSRisk/Tulane-Sentiment-Analysis/releases/download/rss_json/bertopic_dir.zip"
DIR_PATH = Path("Model_training/bertopic_dir")
Github_owner = 'ERSRisk'
Github_repo = 'Tulane-Sentiment-Analysis'
Release_tag = 'BERTopic_results'
Asset_name = 'BERTopic_results2.csv.gz'
GITHUB_TOKEN = os.getenv('TOKEN')

GEMINI_API_KEY = os.getenv("PAID_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

print(f"📥 Downloading all_RSS.json from release link...", flush=True)
response = requests.get(rss_url, timeout = 60)
response.raise_for_status()

data = gzip.decompress(response.content).decode('utf-8')
articles = json.loads(data)
# Now load it
df = pd.DataFrame(articles)

Path("Online_Extraction").mkdir(parents=True, exist_ok = True)
with gzip.open('Online_Extraction/all_RSS.json.gz', 'wb') as f:
    f.write(response.content)
df = df[~(df['Source']=="Economist")]
df['Text'] = df['Title'] + '. ' + df['Content']
def gh_headers():
    token = os.getenv('TOKEN')
    if not token:
        raise RuntimeError('Missing token')
    return {
        'Accept':"application/vnd.github+json",
        "Authorization": f"token {token if token else None}",
    }
def ensure_release(owner, repo, tag:str, token):
    headers = gh_headers()
    r = requests.get(f'https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}', headers=headers, timeout=60)
    if r.status_code == 404:
        r = requests.post(f'https://api.github.com/repos/{owner}/{repo}/releases', headers = headers, timeout = 60, json={
        "tag_name": tag, "name": tag, "draft": False, "prerelease": False
    })
    r.raise_for_status()
    rel = r.json()
    return rel

def upload_asset(owner, repo, release, asset_name, data_bytes, token, content_type = 'application/gzip'):
    assets_api = release['assets_url']
    r = requests.get(assets_api, headers = gh_headers())
    r.raise_for_status()
    for a in r.json():
        if a.get("name") == asset_name:
            del_r = requests.delete(a['url'], headers = gh_headers())
            del_r.raise_for_status()
            break
    upload_url = release['upload_url'].split('{')[0]
    params = {"name": asset_name}
    headers = gh_headers()
    headers['Content-Type'] = content_type
    up = requests.post(upload_url, headers = headers, params = params, data = data_bytes)
    if not up.ok:
        raise RuntimeError(f"Upload failed{up.status_code}: {up.text[:500]}")
    return up.json()

def upload_dir_model_zip(owner, repo, tag, token, dir_path=DIR_PATH, asset_name="bertopic_dir.zip"):
    # zip the directory model into memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        base = dir_path.name
        for root, _, files in os.walk(dir_path):
            for fn in files:
                full = Path(root) / fn
                rel  = Path(base) / full.relative_to(dir_path)
                zf.write(full, arcname=str(rel))
    buf.seek(0)
    # upload via your existing GitHub helper
    rel = ensure_release(owner, repo, tag, token)
    upload_asset(owner, repo, rel, asset_name, buf.getvalue(), token, content_type="application/zip")
    print(f"✅ Uploaded {asset_name} to release {tag}.")

def atomic_write_csv(path: str, df, compress: bool = False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    if compress:
        df.to_csv(tmp, index=False, compression="gzip")
    else:
        df.to_csv(tmp, index=False)
    os.replace(tmp, p)
    print(f"✅ Wrote {p} ({p.stat().st_size/1e6:.2f} MB)")

def load_dir_model():
    # load from disk if present
    if DIR_PATH.exists() and any(DIR_PATH.iterdir()):
        print("📦 Loading BERTopic from local directory model...")
        return BERTopic.load(str(DIR_PATH))
    # try Releases (zip)
    try:
        print("🌐 Fetching bertopic_dir.zip from Releases...")
        r = requests.get(DIR_URL, timeout=120)
        if r.ok and r.content[:2] == b"PK":  # zip magic
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                zf.extractall(DIR_PATH.parent)
            print("✅ Extracted bertopic_dir.zip.")
            return BERTopic.load(str(DIR_PATH))
        else:
            print(f"⚠️ No directory model at {DIR_URL} (status {r.status_code}).")
    except Exception as e:
        print("⚠️ Could not download dir model:", e)
    return None

def estimate_tokens(text):
    # Approx 4 chars per token (rough estimate for English, GPT-like models)
    return len(text) / 4


def save_to_json(topics, topic_names):
    topic_dict = []

    for i, topic in enumerate(topics):
        docs = topic_model.get_representative_docs()[topic]
        keywords = ', '.join([word for word, _ in topic_model.get_topic(topic)])
        topic_dict.append({
            "topic": topic,
            "name": topic_names[i] if i < len(topic_names) else f"Topic {topic}",
            "keywords": keywords,
            "documents": docs
        })
    with open('Model_training/topics_BERT.json', 'w') as f:
        json.dump(topic_dict, f, indent=4)

topic_blocks = []
#
topic_model = load_dir_model()

def get_topic(temp_model, topic_ids):
    print("✅ Preparing topic blocks for Gemini naming...", flush=True)
    topic_blocks = []
    rep = temp_model.get_representative_docs()
    rep_map = {}
    if isinstance(rep, dict):
        rep_map = {int(k): v for k, v in rep.items() if v is not None}
    for topic in topic_ids:
        words = temp_model.get_topic(topic)
        docs = rep_map.get(int(topic))
        if docs is None:
            try:
                docs = temp_model.get_representative_docs()[topic]
            except Exception:
                docs = []
        docs = docs[:2]
        keywords = ', '.join([word for word, _ in words])
        doc_list = '\n'.join([f"- {doc}" for doc in docs])
        block = (
            f"---\n"
            f"TopicID: {topic}\n"
            f"Keywords: {keywords}\n"
        )
        topic_blocks.append((topic, block))

    chunk_size = 1
    topic_name_pairs = []
    print(f"✅ Starting Gemini API calls on {len(topic_blocks)} topics...", flush=True)
    for i in range(0, len(topic_blocks), chunk_size):
        chunk = topic_blocks[i:i + chunk_size]
        print(f"🔹 Sending prompt chunk {i // chunk_size + 1}/{(len(topic_blocks) // chunk_size) + 1}", flush=True)

        prompt_blocks = "\n\n".join([b for (_, b) in chunk])
        prompt = (
            "You are helping analyze topics from BERTopic. Each topic includes keywords and representative documents.\n"
            "Your task is to return a short, clear name for each topic, based ONLY on the provided keywords and documents.\n"
            "Return your response as a list: one name per topic, in order, no explanations.\n"
            "Example: ['Erosion of Human Rights', 'University Funding Cuts', ...]\n\n"
            + prompt_blocks +
            "\nReturn your response as a JSON array of names."
        )

        tokens_estimate = estimate_tokens(prompt)  
        print(f"🔹 Sending prompt with approx {int(tokens_estimate)} tokens...")
        if tokens_estimate > 10000:
            print("⚠️ Prompt too large, consider lowering chunk_size!")
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                response = client.models.generate_content(model="gemini-2.0-flash", contents=[prompt])
                output_text = response.candidates[0].content.parts[0].text
                output_text = re.sub(r"^```(?:json)?\s*", "", output_text)
                output_text = re.sub(r"\s*```$", "", output_text)
                print(output_text)
                new_names = json.loads(output_text)
                topic_name_pairs.extend(zip([tid for (tid, _) in chunk], new_names))
                print(f"✅ Chunk {i // chunk_size + 1} processed and topic names extracted.")
                break
            except Exception as e:
                print(f"❌ Failed to parse Gemini response: {e}")
                print("Raw response:")
                print(response)

                break  # success!
            except APIError as e:
                error_str = str(e)
                if "RESOURCE_EXHAUSTED" in error_str or "quota" in error_str.lower():
                    retry_delay = 60
                    retry_match = re.search(r"'retryDelay': '(\d+)s'", error_str)
                    if retry_match:
                        retry_delay = int(retry_match.group(1))
                    print(f"⚠️ Quota exceeded, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    print(f"❌ Non-retryable API error: {e}")
                    return "❌ API error encountered."
            except Exception as e:
                wait_time = 2 ** attempt + random.uniform(0, 1)
                print(f"⚠️ Unexpected error: {e}. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
        else:
            print("❌ API failed after multiple attempts.")
            return "❌ API failed after multiple attempts."

    return topic_name_pairs

def label_model_topics(topic_model, path = 'Model_training/topics_BERT.json'):
    with open(path, 'r') as f:
        topics_json = json.load(f)
    topic_map = {int(t['topic']): t for t in topics_json}

    rep_docs = topic_model.get_representative_docs()
    print(rep_docs)
    patched = False
    for tid, entry in topic_map.items():
        docs = entry.get("documents", [])
        if not docs:  # only fill in if missing/empty
            new_docs = topic_model.get_representative_docs()
            entry["documents"] = (new_docs or [])[:5]  # cap at 5
            patched = True

    if patched:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(list(topic_map.values()), f, indent=4, ensure_ascii=False)
        print(f"✅ Patched documents for {path}")
    else:
        print("ℹ️ No missing documents to patch.")

#if topic_model:
#    label_model_topics(topic_model)
if topic_model is None:
    print("🧪 Training new BERTopic model (directory format)...", flush=True)
    topic_model = BERTopic(
        language="english",
        verbose=True,
        umap_model=UMAP(n_neighbors = 50, n_components = 10, min_dist = 0.0, metric = 'cosine', random_state = 42),
        hdbscan_model=HDBSCAN(min_cluster_size=15, min_samples = 10, cluster_selection_method = 'eom', prediction_data=True),
        vectorizer_model = CountVectorizer(ngram_range=(1,2), stop_words = 'english', min_df = 5, max_df = 0.8),
        calculate_probabilities=True,
        seed_topic_list=None,
    )
    topics, probs = topic_model.fit_transform(df['Text'].tolist())
    c_tf_idf = topic_model.c_tf_idf_.toarray()

# Cosine similarity between topics
    S = cosine_similarity(c_tf_idf)

    # Build groups to merge where similarity >= your_threshold
    thr = 0.80
    n_topics = S.shape[0]
    visited = set()
    groups = []
    for i in range(n_topics):
        if i in visited: 
            continue
        group = {i}
        for j in range(n_topics):
            if i != j and S[i, j] >= thr:
                group.add(j)
        visited.update(group)
        if len(group) > 1:
            groups.append(sorted(group))

    # Merge the groups
    if groups:
        topic_model.merge_topics(df["Text"].tolist(), groups)
    df['Topic'] = topics
    topics_arr = np.array(topics)
    df_prob = np.full(len(topics_arr), np.nan, dtype=float)
    if probs is not None:
        valid = topics_arr >= 0            # ignore outliers (-1)
        df_prob[valid] = probs[valid, topics_arr[valid]]
    df['Probability'] = df_prob

    # Save portable directory model
    DIR_PATH.parent.mkdir(parents=True, exist_ok=True)
    topic_model.save(
        str(DIR_PATH),
        serialization="pytorch",
        save_ctfidf=True,
        save_embedding_model=True
    )
    print("✅ Saved BERTopic directory model to", DIR_PATH)
    try:
        upload_dir_model_zip(Github_owner, Github_repo, "rss_json", os.getenv("TOKEN"))
    except Exception as e:
        print("⚠️ Skipped dir-model upload:", e)

    # (Optional) zip & upload the dir-model to your Release so future runs just download it
    #   -> see helper below; call after its definition if you want to publish now
else:
    print("✅ BERTopic directory model loaded.")


if 'Source' not in df.columns:
    df['Source'] = ''

# Convert NaN/None to empty string, keep as string dtype
df['Source'] = df['Source'].astype('string').fillna('')

def transform_text(texts):
    texts = texts.copy()
    print(f"Transforming {len(texts)} articles in batches...")
    all_topics, all_probs = [], []
    batch_size = 100
    texts_list = texts['Text'].tolist()

    for i in range(0, len(texts_list), batch_size):
        batch = texts_list[i:i+batch_size]
        topics, probs = topic_model.transform(batch)
        if probs is None:
            print("transform() return no probabilities", flush = True)
            probs = topic_model.approximate_distribution(batch)
        all_topics.extend(topics)
        all_probs.extend(probs)
        print(f"✅ Transformed batch {i//batch_size + 1}/{(len(texts_list) // batch_size) + 1}")
    if any(t == -1 for t in all_topics):
        all_topics = topic_model.reduce_outliers(texts_list, all_topics, strategy = 'embeddings', threshold = 0.4)

    remaining_idx = [i for i, t in enumerate(all_topics) if t==-1]
    def get_embedder(topic_model):
        if hasattr(topic_model, 'embedding_model_') and topic_model.embedding_model_ is not None:
            embedder = topic_model.embedding_model_
        elif hasattr(topic_model, 'embedding_model') and topic_model.embedding_model is not None:
            embedder = topic_model.embedding_model
        else:
            raise RuntimeError("Could not locate BERTopic's embedding model.")
        return embedder


    def embedding_dim(embedder):
        # Works for raw SentenceTransformer or BERTopic backends
        if hasattr(embedder, "get_sentence_embedding_dimension"):
            return embedder.get_sentence_embedding_dimension()
        if hasattr(embedder, "embedding_model") and hasattr(embedder.embedding_model, "get_sentence_embedding_dimension"):
            return embedder.embedding_model.get_sentence_embedding_dimension()
        # Last resort: probe once
        if hasattr(embedder, "encode"):
            v = np.asarray(embedder.encode(["x"], convert_to_numpy=True))
        elif hasattr(embedder, "embed"):
            v = np.asarray(embedder.embed(["x"]))
        else:
            raise RuntimeError("Unknown embedder type; cannot determine embedding dim.")
        return int(v.shape[-1])

    def encode(texts, embedder):
        # Normalize input
        if isinstance(texts, str):
            texts = [texts]
        elif texts is None:
            texts = []
        else:
            texts = [t.strip() if isinstance(t, str) else "" for t in texts]

        if len(texts) == 0:
            d = embedding_dim(embedder)
            return np.zeros((0, d), dtype=np.float32)

        # Support both raw SentenceTransformer and BERTopic backend
        if hasattr(embedder, "encode"):
            vecs = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=False,
                                   batch_size=64, show_progress_bar=False)
        elif hasattr(embedder, "embed"):
            vecs = np.asarray(embedder.embed(texts))
        else:
            raise RuntimeError("Embedder has neither .encode nor .embed")

        vecs = np.asarray(vecs)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms

    try:
        with open('Model_training/topics_BERT.json', 'r', encoding = 'utf-8') as f:
            topics_json = json.load(f)['topics']
    except Exception as e:
        topics_json = []
        print(f"[warn] Could not read {topics_json_path}: {e}")

    remaining_idx = [i for i, t in enumerate(all_topics) if t == -1]

    # Short-circuit if nothing to assign
    if not remaining_idx:
        texts['Topic'] = all_topics
        # ... continue assembling outputs as you already do
        return texts

    embedder = get_embedder(topic_model)

    centroids = []
    streamlit_topic_ids = []
    rep = topic_model.get_representative_docs()
    rep_map = {int(k): v for k, v in rep.items() if v} if isinstance(rep, dict) else {}

    for t in topics_json:
        if t.get('source') != 'Streamlit':
            continue
        reps = rep_map.get(int(t['topic'])) or [f"{t.get('name','')} ; {t.get('keywords','')}"]
        E_rep = encode(reps, embedder)
        if E_rep.shape[0] == 0:
            continue
        c = E_rep.mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        centroids.append(c)
        streamlit_topic_ids.append(int(t['topic']))

    if centroids:
        C = np.stack(centroids, axis=0)
        E = encode([texts_list[i] for i in remaining_idx], embedder)
        if E.shape[0] > 0:
            sims = E @ C.T
            j_best = np.argmax(sims, axis=1)
            s_best = sims[np.arange(sims.shape[0]), j_best]
            for row_pos, idx in enumerate(remaining_idx):
                if s_best[row_pos] >= 0.40:
                    all_topics[idx] = int(streamlit_topic_ids[j_best[row_pos]])

            st_cos = np.full(len(texts_list), np.nan, dtype=np.float32)
            for row_pos, idx in enumerate(remaining_idx):
                st_cos[idx] = float(s_best[row_pos])
            texts["StreamlitCosine"] = st_cos
    else:
        print("[info] No Streamlit topics with usable centroids found; skipping cosine assignment.")



    texts['Topic'] = all_topics
    
    assigned_probs = []
    for t, p in zip(all_topics, all_probs):
        if p is None or t < 0 or t >= len(p):
            assigned_probs.append(np.nan)
        else:
            assigned_probs.append(float(p[t]))
    texts['Probability'] = assigned_probs
    if assigned_probs is None:
        texts['Probability'] = texts.get('StreamlitCosine', pd.Series(0.0, index = texts.index)).astype(float)
    how = []
    for i, (t,p, sim) in enumerate(zip(texts['Topic'], texts['Probability'], texts.get('StreamlitCosine', [np.nan]*len(df)))):
        if t == -1:
            how.append('Unassigned')
        elif not np.isnan(p):
            how.append('bertopic')
        elif not np.isnan(sim):
            how.append('streamlit-cosine')
        else:
            how.append('other')
    texts['Assigned_how'] = how
    return texts
def load_articles_from_release(local_cache_path='Model_training/BERTopic_results2.csv.gz',
                               usecols=None, dtype=None):
    rel = get_release_by_tag(Github_owner, Github_repo, Release_tag)
    # 1) Try remote release asset (streamed)
    if rel:
        asset = next((a for a in rel.get('assets', []) if a['name'] == Asset_name), None)
        if asset:
            url = asset['browser_download_url']
            # Stream to disk instead of holding in RAM
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                Path(local_cache_path).parent.mkdir(parents=True, exist_ok=True)
                with open(local_cache_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                        if chunk:
                            f.write(chunk)
            # Read the gz file directly from disk
            return pd.read_csv(local_cache_path, compression='gzip',
                               low_memory=False, usecols=usecols, dtype=dtype)

    # 2) Fallback to local cache if present
    P = Path(local_cache_path)
    if P.exists():
        return pd.read_csv(local_cache_path, compression='gzip',
                           low_memory=False, usecols=usecols, dtype=dtype)

    # 3) Nothing available
    return pd.DataFrame()
def save_new_topics(existing_df, new_df, path = 'Model_training/BERTopic_results2.csv.gz'):
    if 'Link' in existing_df and 'Link' in new_df:
        unique_new = new_df[~new_df['Link'].isin(existing_df['Link'])]
    else:
        unique_new = new_df

    on_disk = load_articles_from_release()
    if on_disk is None or (isinstance(on_disk, pd.DataFrame) and on_disk.empty):
        on_disk = pd.read_csv(path, compression = 'gzip')

    pieces = [p for p in [existing_df, unique_new, on_disk] if not (isinstance(p, pd.DataFrame) and p.empty)]
    combined = pd.concat(pieces, ignore_index = True) if pieces else pd.DataFrame()

    if not combined.empty and {'Title', 'Content'}.issubset(combined.columns):
        combined = combined.drop_duplicates(subset = ['Title', 'Content'], keep = 'last')

    combined.to_csv(path, index = False, compression = 'gzip')
    return combined

def double_check_articles(df):
    double_check = df[df['Topic'] == -1]['Text'].dropna()
    double_check = [text for text in double_check if text.strip()]
    if not double_check:
        return None, []
    n = len(double_check)
    if n < 3:
        print(f"[info] Skipping double_check_articles: only {n} doc(s).")
        return None, []
    safe_neighbors = max(2, min(10, n-1))
    safe_components = max(1, min(5, n-1))
    min_cluster_size = max(2, min(15, max(2, n//2)))
    umap_small = UMAP(
        n_neighbors=safe_neighbors,
        n_components=safe_components,
        min_dist=0.0,
        metric='cosine',
        init='random',
        random_state=42,
    )
    hdbscan_small = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None,
        cluster_selection_method='eom',
        prediction_data=True,
    )

    temp_model = BERTopic(
        language='english',
        verbose=True,
        umap_model=umap_small,
        hdbscan_model=hdbscan_small,
        calculate_probabilities=True,
    )
    try:
        temp_model.fit_transform(double_check)
    except TypeError as e:
        # Catch the k >= N spectral error or any other tiny-graph hiccup
        print(f"[warn] Fallback in double_check_articles due to: {e}")
        return None, []
    except Exception as e:
        print(f"[warn] Could not double-check topics: {e}")
        return None, []

    topic_ids = temp_model.get_topic_info()
    topic_ids = topic_ids[topic_ids['Topic'] != -1]['Topic'].tolist()
    print(f"[double_check] Found {len(topic_ids)} valid topics (excluding -1).", flush = True)
    return temp_model, topic_ids
def fetch_release(owner, repo, tag:str, asset_name:str, token:str):
    headers = {'Authorization': f'token {token}',
              'Accept': 'application/vnd.github+json'}
    r = requests.get(f'https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}', headers=headers, timeout=60)
    if r.status_code == 404:
        r = requests.post(f'https://api.github.com/repos/{owner}/{repo}/releases', headers = headers, timeout = 60, json={
        "tag_name": tag, "name": tag, "draft": False, "prerelease": False
    })
    r.raise_for_status()
    rel = r.json()
    upload_url = rel['upload_url'].split('{', 1)[0]
    assets = requests.get(f"https://api.github.com/repos/{owner}/{repo}/releases/{rel['id']}/assets", headers=headers, timeout=60).json()
    a = next((x for x in assets if x.get("name") == asset_name), None)
    if not a:
        return []

    url = a.get('browser_download_url')
    b = requests.get(url, headers = headers, timeout = 120)
    b.raise_for_status()
    content = b.content
    return json.loads(content.decode('utf-8'))
def upload_asset_to_release(owner, repo, tag:str, asset_path:str, token:str):
    headers = {'Authorization': f'token {token}',
              'Accept': 'application/vnd.github+json'}
    r = requests.get(f'https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}', headers=headers, timeout=60)
    r.raise_for_status()
    rel = r.json()
    upload_url = rel["upload_url"].split("{", 1)[0]
    assets = requests.get(f"https://api.github.com/repos/{owner}/{repo}/releases/{rel['id']}/assets", headers=headers, timeout=60).json()
    name = os.path.basename(asset_path)
    for a in assets:
        if a.get("name") == name:
            requests.delete(f"https://api.github.com/repos/{owner}/{repo}/releases/assets/{a['id']}", headers=headers, timeout=60)
    with open(asset_path, "rb") as f:
        up = requests.post(
            f"{upload_url}?name={name}",
            headers={"Authorization": f"token {token}", "Content-Type": "application/octet-stream"},
            data=f.read(), timeout=300
        )
    up.raise_for_status()
    return up.json()
def upsert_single_big_json(owner, repo, tag: str, asset_name: str,
                       new_items: list, dedupe_key: str, token: str, mode = 'merge'):
    if mode == 'replace':
        current = []
    else:
        current = fetch_release(owner, repo, tag, asset_name, token)
        if not isinstance(current, list):
            current = []

    # 2) merge by key (new replaces old on same key)
    by_key = {}
    for it in (current if mode == "merge" else []):
        k = it.get(dedupe_key)
        if k is not None:
            by_key[k] = it
    for it in new_items:
        k = it.get(dedupe_key)
        if k is not None:
            by_key[k] = it

    merged = list(by_key.values())

    # 3) write to a temp gz and upload (same asset name → old is deleted then replaced)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, asset_name)  # e.g., "unmatched_topics.json.gz"
        raw = json.dumps(merged, ensure_ascii=False).encode("utf-8")
        if asset_name.endswith(".gz"):
            with gzip.open(path, "wb") as f:
                f.write(raw)
        else:
            with open(path, "wb") as f:
                f.write(raw)
        return upload_asset_to_release(owner, repo, tag, path, token)

def existing_risks_json(topic_name_pairs, topic_model):
    unmatched = list(topic_name_pairs)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load existing named topics (for matching to *known* topics)
    with open('Model_training/topics_BERT.json', 'r', encoding='utf-8') as f:
        topics = json.load(f)
    def doc_sig(doc:str) -> str:
        return hashlib.md5(doc.strip().lower().encode()).hexdigest()
        
    def is_mostly_seen(new_docs, known_doc_sigs, thresh=0.6):
        if not new_docs:
            return False
        hits = sum(1 for d in new_docs if doc_sig(d) in known_doc_sigs)
        return hits / max(1, len(new_docs)) >= thresh
        
    existing_topic_names = [t['name'] for t in topics if 'name' in t]


    try:
        existing_unmatched = fetch_release(
            "ERSRisk", "tulane-sentiment-app-clean",
            "unmatched-topics", "unmatched_topics.json",
            os.getenv('TOKEN')
            ) or []
    except Exception:
        existing_unmatched = []

    try:
        existing_discarded = fetch_release(
            "ERSRisk", "tulane-sentiment-app-clean",
            "discarded-topics", "discarded_topics.json",
            os.getenv('TOKEN')
            ) or []
    except Exception:
        existing_discarded = []
   
    unmatched_names = []
    index_map = [] 
    for i, item in enumerate(existing_unmatched):
        if isinstance(item, dict) and 'name' in item:
            unmatched_names.append(item['name'])
            index_map.append(i)

    discarded_names, discarded_index_map = [], []
    for i, item in enumerate(existing_discarded):
        if isinstance(item, dict) and 'name' in item:
            discarded_names.append(item['name'])
            discarded_index_map.append(i)

    if unmatched_names:
        unmatched_embeddings = model.encode(unmatched_names, convert_to_tensor=True)
    else:
        unmatched_embeddings = None

    discarded_embeddings = (
        model.encode(discarded_names, convert_to_tensor=True)
        if discarded_names else None
    )
    
    to_upsert_unmatched = [] 
    to_upsert_discarded = []  
    seen_changed_unmatched = set()
    seen_changed_discarded = set()

    to_check_discarded = []
    for topic_id, name in unmatched:
        new_emb = model.encode([name], convert_to_tensor=True)
        new_docs = topic_model.get_representative_docs().get(topic_id, [])
        new_keywords_pairs = topic_model.get_topic(topic_id) or []
        new_keywords = [w for (w, _) in new_keywords_pairs]

        if unmatched_embeddings is not None and len(unmatched_names) > 0:
            sims = util.cos_sim(new_emb, unmatched_embeddings)[0]
            best_score = float(sims.max())
            best_idx_in_names = int(sims.argmax())
            best_existing_idx = index_map[best_idx_in_names]
        else:
            best_score = 0.0
            best_existing_idx = None

        if best_score > 0.78 and best_existing_idx is not None:
            matched = existing_unmatched[best_existing_idx]
            # extend docs (dedupe)
            seen = set(matched.get('documents', []))
            for d in new_docs:
                if d not in seen:
                    matched.setdefault('documents', []).append(d)
                    seen.add(d)
            # merge keywords
            ek = matched.get('keywords', [])
            if isinstance(ek, str):
                ek = [kw.strip() for kw in ek.split(',') if kw.strip()]
            kw_seen = set(map(str.lower, ek))
            for kw in new_keywords:
                if kw.lower() not in kw_seen:
                    ek.append(kw)
                    kw_seen.add(kw.lower())
            matched['keywords'] = ek
            to_upsert_unmatched.append(matched)
            seen_changed_unmatched.add(matched.get('name', ''))
        else:
            to_check_discarded.append((topic_id, name))


    for topic_id, name in to_check_discarded:
        new_emb = model.encode([name], convert_to_tensor=True)       # (1, d)
        new_docs = topic_model.get_representative_docs().get(topic_id, [])
        new_keywords_pairs = topic_model.get_topic(topic_id) or []
        new_keywords = [w for (w, _) in new_keywords_pairs]

        if discarded_embeddings is not None and len(discarded_names) > 0:
            sims = util.cos_sim(new_emb, discarded_embeddings)[0]
            best_score = float(sims.max())
            best_idx_in_names = int(sims.argmax())
            best_existing_idx = discarded_index_map[best_idx_in_names]
        else:
            best_score = 0.0
            best_existing_idx = None

        if best_score > 0.78 and best_existing_idx is not None:
            matched = existing_discarded[best_existing_idx]
            # extend docs (dedupe)
            seen = set(matched.get('documents', []))
            for d in new_docs:
                if d not in seen:
                    matched.setdefault('documents', []).append(d)
                    seen.add(d)
            # merge keywords
            ek = matched.get('keywords', [])
            if isinstance(ek, str):
                ek = [kw.strip() for kw in ek.split(',') if kw.strip()]
            kw_seen = set(map(str.lower, ek))
            for kw in new_keywords:
                if kw.lower() not in kw_seen:
                    ek.append(kw)
                    kw_seen.add(kw.lower())
            matched['keywords'] = ek
            to_upsert_discarded.append(matched)
            seen_changed_discarded.add(matched.get('name',''))
        else:
            known_doc_sigs = set()
            for u in existing_unmatched:
                for d in u.get('documents', []):
                    known_doc_sigs.add(doc_sig(d))
            if not is_mostly_seen(new_docs, known_doc_sigs, thresh = 0.6):
                to_upsert_unmatched.append({
                    'topic': topic_id,
                    'name': name,
                    'keywords': new_keywords,
                    'documents': new_docs
                })
            else:
                print(f"skip Topic {name} looks already covered by existing unmatched topics", flush = True)
            
    if to_upsert_unmatched:
        resp = upsert_single_big_json(
                    owner="ERSRisk",
                    repo="tulane-sentiment-app-clean",
                    tag="unmatched-topics",
                    asset_name="unmatched_topics.json",
                    new_items=to_upsert_unmatched,
                    dedupe_key="name",
                    token = os.getenv('TOKEN')
                )
    if to_upsert_discarded:
        resp2 = upsert_single_big_json(
                    owner="ERSRisk",
                    repo="tulane-sentiment-app-clean",
                    tag="discarded-topics",
                    asset_name="discarded_topics.json",
                    new_items=to_upsert_discarded,
                    dedupe_key="name",
                    token = os.getenv('TOKEN')
        )
def risk_weights(df):
    t0 = time.perf_counter()
    print(f"[risk_weights] start: df = {df.shape}", flush = True)

    # ---------- Load config ----------
    with open('Model_training/risks.json', 'r', encoding='utf-8') as f:
        risks_cfg = json.load(f)

    json_all_labels = [r['name'] for block in risks_cfg.get('new_risks', []) for _, items in block.items() for r in items]

    print("risk labels loaded", flush = True)
    accuracy_map = {}
    for s in risks_cfg.get('sources', []):
        name = str(s.get('name', '') or '')
        acc = s.get('accuracy', 0) or 0
        accuracy_map[name] = acc
    level_name_map = {"low":1, "medium low":2, "medium":3, "medium high":4, "high":5}
    risks_map = {}
    for r in risks_cfg.get('risks', []):
        nm = (r.get('name') or '').strip().lower()
        lvl = r.get('level', 0)
        if isinstance(lvl, str):
            lvl = level_name_map.get(lvl.strip().lower(), 0)
        try:
            lvl = float(lvl)
        except Exception:
            lvl = 0.0
        if nm:
            risks_map[nm] = lvl

    higher_ed_dict = risks_cfg.get('HigherEdRisks', None)

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

    now_naive = datetime.utcnow() 
    base['Days_Ago'] = (now_naive - base['Published']).dt.days 
    base['Days_Ago'] = base['Days_Ago'].fillna(10_000).astype(int) 
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
    cand = np.where(cand=='', base.get('Predicted_Risks', ''), cand)
    cand = pd.Series(cand, index=base.index).fillna('').astype(str)
    
    def _first_label(s):
        s = s.strip()
        if not s:
            return ''
        s = re.sub(r'^\[|\]$', '', s)
        s = re.split(r'[;,]', s)[0].strip().strip("'\"")
        return s
    
    base['_RiskList'] = cand.apply(_first_label)
    base['Risk_item'] = np.where(base['_RiskList'].eq(''), 'No Risk', base['_RiskList'])

    if 'Topic' not in base.columns:
        base['Topic'] = -1
    base['Topic'] = base['Topic'].fillna(-1)
    def recency_features_topic_risk(df, now=None):
        fx = df.copy()

        required = {'Topic', '_RiskList', 'Published', 'Days_Ago'}
        if not required.issubset(fx.columns) or fx.empty:
            return pd.DataFrame(columns=['Topic','_RiskList','last_seen_days','decayed_volume','recency_score_tr'])

        if now is None:
            now = pd.Timestamp.utcnow()

        art_w = 1.0
        if 'Impact_Score' in base.columns:
            art_w = pd.to_numeric(base['Impact_Score'], errors='coerce').fillna(0.0).clip(0, 1)

        def half_life(risk):
            return risk_half_life.get(risk, 30)

        hl  = fx['_RiskList'].map(lambda r: max(1.0, half_life(r)))
        lam = np.log(2.0) / hl
        w_decay = np.exp(-lam * fx['Days_Ago'])
        fx['_w'] = w_decay * art_w

        grp = fx.groupby(['Topic', '_RiskList'], dropna=False)
        out = grp.agg(
            last_seen=('Days_Ago', 'min'),
            decayed_volume=('_w', 'sum'),
            mentions=('Published', 'count')
        ).reset_index()

        out['hl'] = out['_RiskList'].map(lambda r: max(1.0, half_life(r)))
        out['freshness'] = np.exp(-np.log(2.0) * (out['last_seen'] / out['hl']))

        def _safe_minmax(s):
            rng = s.max() - s.min()
            return (s - s.min()) / (rng + 1e-12)

        out['decayed_z'] = out.groupby('_RiskList')['decayed_volume'].transform(_safe_minmax)

        w_fresh, w_vol = 0.6, 0.4
        out['recency_score_tr'] = (w_fresh * out['freshness'] + w_vol * out['decayed_z']).clip(0, 1)
        out = out.rename(columns={'last_seen': 'last_seen_days'})
        return out[['Topic','_RiskList','last_seen_days','decayed_volume','recency_score_tr']]


    def attach_topic_risk_recency(df):
        tr = recency_features_topic_risk(df)
        for c in ['last_seen_days','decayed_volume','recency_score_tr']:
            if c not in tr.columns:
                tr[c] = np.nan
        cols_to_drop = ["_RiskList","last_seen_days","decayed_volume",
                    "recency_score_tr","recency_score_tr_x","recency_score_tr_y"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

        tr_small = tr[["Topic","_RiskList","last_seen_days","decayed_volume","recency_score_tr"]].rename(
            columns={"recency_score_tr": "recency_score_tr_tr"}
        )
        overlap = [c for c in tr_small.columns if c in df.columns and c!= 'Topic']
        if overlap:
            df = df.drop(columns = overlap)
        enriched = df.merge(tr_small, on="Topic", how="left")

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
    

    def _src_acc(row):
        src = str(row.get('Source','') or '')
        link = str(row.get('Link', '') or '')
        src_l = src.lower()
        dom = ''
        if link:
            try:
                dom = urlparse(link).netloc.lower()
            except Exception:
                dom = ''
        if dom.endswith('.gov') or dom.endswith('.mil') or dom.endswith('tulane.edu'):
            return 5
        best = 0.0
        for name, acc in accuracy_map.items():
            if name and name.lower() in src_l:
                try:
                    v = float(acc)
                except Exception:
                    v = 0.0
                best = max(best, v)

        return min(best, 4.0)
    base['Source_Accuracy'] = base.apply(_src_acc, axis=1)
    print('Source accuracy created', flush = True)

    def _loc_score(row):
        us_sources = ['foxnews','NIH', 'NOAA', 'FEMA', 'NASA', 'CISA', 'NIST', 'NCES', 'CMS', 'CDC', 'BEA', 'The Advocate', 'LA Illuminator', 'The Hill', 'NBC News', 'PBS', 'StatNews', 'NY Times', 'Washington Post', 'TruthOut', 'Politico', 'Inside Higher Ed', 'CNN', 'Yahoo News', 'FOX News', 'ABC News', 'Huffington Post', 'Business Insider', 'Bloomberg', 'AP News']
        raw = row.get('Entities', None)

        if isinstance(raw, list):
            entities = [str(e).lower() for e in raw if e is not None]
        elif isinstance(raw, str) and raw.strip():
            entities = [raw.strip().lower()]
        else:
            entities = []
        text = (row.get('Title','') + ' ' + row.get('Content','')).lower()
        def has_any(keys): 
            lk = [k.lower() for k in keys]
            return any(k in text for k in lk)
        if has_any(['tulane','tulane university']): return 5
        if has_any(['new orleans','louisiana','nola']): return 4
        if has_any(['baton rouge','governor landry','lafayette','lsu','university of louisiana']): return 3
        if has_any(['gulf coast','mississippi','texas','alabama']): return 2
        if has_any(['u.s.','united states','america','federal','washington dc','trump']) or str(row.get('Source','')) in us_sources: return 1
        return 0

    base['Location'] = base.apply(_loc_score, axis=1)
    base['Location'] = pd.to_numeric(base['Location'], errors = 'coerce').fillna(0).astype(int)
    print('Location created', flush = True)



    def _window_tag(d):
        if d <= 30: return 'recent'
        if d <= 60: return 'previous'
        return 'older'
    tmp = base[['Topic','Days_Ago']].copy()
    tmp['Time_Window'] = tmp['Days_Ago'].apply(_window_tag)
    topic_counts = tmp.groupby(['Topic','Time_Window']).size().unstack(fill_value=0)
    for c in ['recent','previous']:
        if c not in topic_counts.columns:
            topic_counts[c] = 0

    periods = base['Published'].dt.to_period('W-MON')
    base['Week'] = periods.dt.to_timestamp(how = 'start')
    ts = (
        base.loc[base['Week'].notna()]
        .groupby(['Risk_item','Week'])
        .size().rename('n').reset_index()
        .sort_values(['Risk_item','Week'])
    )
    if not ts.empty:
        ts['EMWA'] = ts.groupby('Risk_item')['n'].transform(
            lambda s: s.ewm(span = 4, adjust =False).mean()
        )

        ts['EMWA_Delta'] = ts.groupby('Risk_item')['EMWA'].diff().fillna(0.0)
        def slope(counts, k=6):
            x = np.arange(len(counts), dtype=float)
            out = np.zeros(len(counts), dtype=float)
            for i in range(len(counts)):
                lo = max(0, i - min(k, i) + 1) 
                xi = x[lo:i+1]; yi = counts[lo:i+1].astype(float)
                if len(xi) >= 2: 
                    m, _ = np.polyfit(xi, yi, 1)
                    out[i] = m
            return out

        ts['Slope'] = ts.groupby('Risk_item', group_keys = False)['n'].apply(lambda g: pd.Series(slope(g.values, k=6), index = g.index)).astype(float)

        def normalize_groupwise(s, by):
            return s.groupby(by, group_keys=False).rank(pct = True).fillna(0.0)

        ts['emwa_norm']  = normalize_groupwise(ts['EMWA'].clip(lower=0),  ts['Risk_item'])
        ts['slope_norm'] = normalize_groupwise(ts['Slope'].clip(lower = 0), ts['Risk_item'])

        w_emwa, w_slope = 0.6, 0.4
        ts['accel_score'] = (w_emwa*ts['emwa_norm'] + w_slope * ts['slope_norm']).clip(0,1)
        weeks_seen = ts.groupby('Risk_item')['Week'].transform('nunique')
        ts.loc[weeks_seen < 4, 'accel_score'] *= 0.6


        #if 'Sentiment Score' not in base.columns:
        #    base['Sentiment Score'] = 0.0
        #ts_sent = (
        #    base.loc[base['Week'].notna()]
        #    .groupby(['Risk_item','Week'])
        #    .agg(sent_mean=('Sentiment Score','mean'))
        #    .reset_index()
        #    .sort_values(['Risk_item','Week'])
        #)
        #ts_sent['sent_flipped'] = -ts_sent['sent_mean']
        #ts_sent['sent_ewma'] = ts_sent.groupby('Risk_item')['sent_flipped'].transform(
        #    lambda s: s.ewm(span=4, adjust=False).mean()
        #)
        #ts_sent['sent_delta'] = ts_sent.groupby('Risk_item')['sent_ewma'].diff().fillna(0.0)
        #ts_sent['sent_slope'] = ts_sent.groupby('Risk_item', group_keys=False)['sent_flipped'] \
        #    .apply(lambda g: pd.Series(slope(g.values, k=6), index=g.index)) \
        #    .astype(float)
        #ts_sent['sent_delta_norm'] = normalize_groupwise(ts_sent['sent_delta'], ts_sent['Risk_item'])
        #ts_sent['sent_slope_norm'] = normalize_groupwise(ts_sent['sent_slope'], ts_sent['Risk_item'])
        #w_sent_delta, w_sent_slope = 0.6, 0.4
        #ts_sent['accel_score_sent'] = (w_sent_delta*ts_sent['sent_delta_norm'] + w_sent_slope*ts_sent['sent_slope_norm']).clip(0,1)

        ts['accel_score'] = ts['accel_score'].fillna(0.0)



        def _acc_value(d):
            if d < 0.1: return 0
            if d < 0.25: return 1
            if d < 0.40: return 2
            if d < 0.60: return 3
            if d < 0.80: return 4
            return 5
        ts['Acceleration_value'] = ts['accel_score'].apply(_acc_value).astype(int)

        def cue_eta_from_text(t):
            t = str(t).lower()
            if re.search(r'\b(today|tonight|tomorrow|immediately|right now)\b', t):
                return 3
            if re.search(r'\b(this week|in\s*the\s*coming\s*days)\b', t):
                return 7
            # near-term windows
            if re.search(r'\b(next week|within\s*2\s*weeks|in\s*2\s*weeks)\b', t):
                return 14
            if re.search(r'\b(within\s*30\s*days|this month|in\s*\d+\s*days)\b', t):
                return 30
            return 45

        temp = base.loc[base['Week'].notna(), ['Risk_item', 'Week', 'Title','Content', 'Location','University Label']].copy()
        temp['cue_eta'] = (temp['Title'] + ' ' + temp['Content'].fillna('')).apply(cue_eta_from_text)

        def location_eta(l):
            try:
                loc = int(l)
            except Exception:
                loc = 0
            if loc == 5: return 7
            if loc == 1: return 21
            return 45
        temp['loc_eta'] = temp['Location'].apply(location_eta)
        def tulane_pull(u_label, eta):
            try:
                label = int(u_label)
            except Exception:
                label = 0
            return max(1, eta - (7 if label ==1 else 0))

        temp['eta_article'] = temp.apply(lambda x: tulane_pull(x['University Label'], min(x['cue_eta'], x['loc_eta'])), axis =1)

        eta_by_bucket = (
            temp.groupby(['Risk_item', 'Week'])['eta_article'].min()
            .rename('eta_days_proxy').reset_index()
        )

        ts = ts.merge(eta_by_bucket, on = ['Risk_item', 'Week'], how='left')
        ts['eta_days_proxy'] = ts['eta_days_proxy'].fillna(45).astype(float)

        ts['eta_days_proxy'] = (ts['eta_days_proxy']- (ts['accel_score'] * 5.0)).clip(lower =1)

        def cap_by_eta(val, eta_days):
            if eta_days <= 15:
                return(min(int(val), 5))
            elif eta_days <= 30:
                return (min(int(val),4))
            else: 
                return min(int(val), 3)

        ts['Acceleration_value'] = ts.apply(
            lambda r: cap_by_eta(r['Acceleration_value'], r['eta_days_proxy']),
            axis=1
        ).astype(int)

    base = base.drop(columns = ['Acceleration_value_x', 'Acceleration_value_y'], errors = 'ignore')
    right = ts[['Risk_item', 'Week', 'Acceleration_value']].rename(columns = {'Acceleration_value': 'Acceleration_value_new'})
    base = base.merge(right, on = ['Risk_item', 'Week'], how = 'left', validate = 'm:m')
    if 'Acceleration_value' in base.columns:
        base['Acceleration_value'] = base['Acceleration_value_new'].fillna(base['Acceleration_value'])
    else:
        base['Acceleration_value'] = base['Acceleration_value_new']


    base = base.drop(columns = ['Acceleration_value_new'])
    base['Acceleration_value'] =base['Acceleration_value'].fillna(0).astype(int)
    print('Acceleration value created', flush = True)


    if base.empty:
        base['Frequency_Score'] = 0
    else:
        counts = base['Risk_item'].value_counts().rename_axis('Risk_item').reset_index(name='Count')
        try:
            bins = pd.qcut(counts['Count'].rank(method='first'), 5, labels=[1,2,3,4,5])
            counts['Frequency_Score'] = bins.astype(int)
        except Exception:
            mn, mx = counts['Count'].min(), counts['Count'].max()
            if mx == mn:
                counts['Frequency_Score'] = 3
            else:
                scaled = 1 + 4 * (counts['Count'] - mn) / float(mx - mn)
                counts['Frequency_Score'] = scaled.round().clip(1,5).astype(int)
        freq_map = dict(zip(counts['Risk_item'], counts['Frequency_Score']))
        base['Frequency_Score'] = base['Risk_item'].map(freq_map).fillna(0).astype(int)

    hed = risks_cfg.get('HigherEdRisks') or {}
    hed_norm = {
        str(cat).strip().lower(): [str(p).strip().lower() for p in (phr_list or []) if str(p).strip()]
        for cat, phr_list in hed.items()
    }

    def phrase_to_pattern(phrase: str) -> str:
        tokens = re.split(r"\s+", phrase.strip())
        # allow any non-word chars between tokens; keep word boundaries at ends
        return r"(?<!\w)" + r"[\W_]+".join(map(re.escape, tokens)) + r"(?!\w)"

    cat_regex = {}
    for cat, phrases in hed_norm.items():
        if not phrases:
            continue
        pats = [phrase_to_pattern(p) for p in phrases]
        pats.append(phrase_to_pattern(cat))
        cat_regex[cat] = re.compile("(" + "|".join(pats) + ")", flags=re.I)
    
    text_all = (base['Title'].fillna('') + ' ' + base['Content'].fillna('')).astype(str)
    detected_cats = [{cat for cat, rx in cat_regex.items() if rx.search(t)} for t in text_all]
    base['Detected_HigherEd_Categories'] = detected_cats

    ul = base.get('University Label', 0)
    base['Industry_Risk_Presence'] = np.where(
        (base['Detected_HigherEd_Categories'].apply(len) > 0) | (pd.to_numeric(ul, errors='coerce').fillna(0).astype(int) == 1),
        3, 0
    ).astype(int)

    peers_list = risks_cfg.get('Peer_Institutions') or []
    peer_pat = re.compile(r'\b(' + '|'.join([re.escape(p) for p in peers_list]) + r')\b', flags=re.I) if peers_list else None

    moderate_impact = re.compile(r'\b(outage|closure|lawsuit|probation|sanction|breach|evacuation|investigation)\b', re.I)
    substantial_impact = re.compile(r'\b(widespread|catastrophic|shutdown|bankrupt|insolvenc\w*|fatalit\w*|revocation|accreditation\s+revoked)\b', re.I)
    _text_all = (base['Title'].fillna('') + ' ' + base['Content'].fillna('')).astype(str)
    base['_tulane_flag'] = (base.get('Location', 0).astype(int).eq(5)) | _text_all.str.contains(r'\btulane\b', case=False, regex=True)

    def find_peer(t):
        if not peer_pat:
            return ''
        m = peer_pat.search(t or '')
        return m.group(0) if m else ''

    def severity(text):
        if substantial_impact.search(text or ''):
            return 'substantial'
        if moderate_impact.search(text or ''):
            return 'moderate'
        return ''

    tmp_ind = base.loc[base['Week'].notna(), ['Week', 'Title', 'Content']].copy()
    tmp_ind['text_all'] = (tmp_ind['Title'] + ' ' + tmp_ind['Content']).fillna('')
    tmp_ind['peer'] = tmp_ind['text_all'].apply(find_peer)
    tmp_ind['sev'] = tmp_ind['text_all'].apply(severity)

    # Count unique peers by severity per week
    agg = (
        tmp_ind.groupby(['Week', 'sev'])['peer']
        .nunique()
        .unstack(fill_value=0)
        .rename(columns={'moderate': 'peers_mod', 'substantial': 'peers_sub'})
    )
    for c in ['peers_mod', 'peers_sub']:
        if c not in agg.columns:
            agg[c] = 0
    agg = agg.reset_index()

    
    tulane_week = (
        base.loc[base['_tulane_flag'] & base['Week'].notna()]
        .groupby('Week')
        .size()
        .rename('tulane_mentions')
        .reset_index()
    )
    agg = agg.merge(tulane_week, on='Week', how='left').fillna({'tulane_mentions': 0})


    if not agg.empty:
        week_max = agg['Week'].max()
        agg['days_ago'] = (week_max - agg['Week']).dt.days.clip(lower=0)
        lam = np.log(2.0) / 21.0
        agg['decay_w'] = np.exp(-lam * agg['days_ago'])

        
        agg['peer_index'] = agg['decay_w'] * (2 * agg['peers_sub'] + 1 * agg['peers_mod'])
        agg['sector_pressure'] = agg['peer_index'] / (1.0 + agg['tulane_mentions'])

        
        lo, hi = np.percentile(agg['sector_pressure'], [5, 95]) if agg['sector_pressure'].notna().any() else (0.0, 1.0)
        rng = max(1e-12, hi - lo)
        agg['Industry_Risk_Peer'] = (((agg['sector_pressure'] - lo) / rng).clip(0, 1) * 5).round().astype(int)
    else:
        agg['Industry_Risk_Peer'] = 0


    base = base.drop(columns=['Industry_Risk_Peer'], errors='ignore')
    base = base.merge(agg[['Week', 'Industry_Risk_Peer']], on='Week', how='left')
    base['Industry_Risk_Peer'] = base['Industry_Risk_Peer'].fillna(0).astype(int)


    base['Industry_Risk'] = np.maximum(base['Industry_Risk_Presence'], base['Industry_Risk_Peer']).astype(int)
    print('Industry risk created', flush = True)



    impact_weights = {"financial": 0.35, "reputational": 0.15, "academic": 0.25, "operational": 0.25}
    new_risks = risks_cfg.get('new_risks')
    risk_dims_map = {}
    for block in new_risks:
        for category, items in block.items():
            for r in items:
                name = re.sub(r'\s+', ' ', r.get('name', '')).strip().lower()
                dims = r.get('impact dims')
                risk_dims_map[name] = {
                    'financial': float(dims.get('financial', 0.0)),
                    'reputational': float(dims.get('reputational', 0.0)),
                    'academic': float(dims.get('academic', 0.0)),
                    'operational': float(dims.get('operational', 0.0))
                }

    existential_threat_patterns = [
    r'\b(permanent\s+closure|cease\s+operations|bankrupt|insolven|shut\s*down\s*permanent|revocation\s+of\s+accreditation)\b',
    r'\b(catastrophic\s+(damage|failure)|total\s+loss|existential\s+threat)\b'
]
    severe_patterns = [
    r'\b(university[-\s]*wide|campus[-\s]*wide|enterprise[-\s]*wide|entire\s+university|all\s+systems\s+down)\b',
    r'\b(ransomware|mass\s+evacuation|classes\s+canceled\s+across\s+campus|network\s+outage)\b',
    r'\b(federal\s+investigation|systemic\s+title\s*ix|major\s+scandal|fatalit(y|ies))\b',
    r'\b(executive\s+action\b(?:\s\w+){0,100}?\s(college|university|universities))\b',
    r'\b(regulatory\s+action|regulation\s\b(?:\s\w+){0,100}\s(higher\seducation|university|universities|college|colleges))\b'
]

    def find_patterns(text, patterns):
        for p in patterns:
            if re.search(p, text, flags=re.I):
                return True
        return False

    def impact_row(row):
        risk = re.sub(r'\s+', ' ', str(row['Risk_item'])).strip().lower()
        dims = risk_dims_map.get(risk)

        if dims:
            fin, rep, acad, oper = dims['financial'], dims['reputational'], dims['academic'], dims['operational']
        else:
            fin, rep, acad, oper = 1.0, 1.0, 1.0, 1.0

        base = (fin * impact_weights['financial'] +
                rep * impact_weights['reputational'] +
                acad * impact_weights['academic'] +
                oper * impact_weights['operational'])

        text = (str(row.get('Title','')) + ' ' + str(row.get('Content',''))).lower()
        existential = find_patterns(text, existential_threat_patterns)
        severe = find_patterns(text, severe_patterns) or (int(row.get('Location',0))==5 and int(row.get('University Label',0))==1)

        if existential:
            return min(5.0, max(5.0, base))
        if severe:
            return min(4.0, max(4.0, base))
        else:
            return min(base, 3.9)
    for col in ['Location', 'University Label']:
        if col not in base.columns:
            base[col] = 0
        base[col] = pd.to_numeric(base[col], errors = 'coerce').fillna(0).astype(int)
    base['Impact_Score'] = base.apply(impact_row, axis=1).astype(float)
    print('Impact score computed', flush = True)

    t_rec = time.perf_counter()
    print("attach_topic_risk_recency() start", flush = True)
    base = attach_topic_risk_recency(base) 
    base['Recency'] = (base['Recency_TR_Blended'] * 5).round(2)
    print("[recency] attached in {time.perf_counter()-t_rec:.1f}s", flush = True)


    w = {
        'Recency': 0.15,
        'Source_Accuracy': 0.10,
        'Impact_Score': 0.35,
        'Acceleration_value': 0.25,
        'Location': 0.05,
        'Industry_Risk': 0.05,
        'Frequency_Score': 0.05
    }
    weight_sum = sum(w.values()) 

    num = (
        base['Recency'] * w['Recency'] +
        base['Source_Accuracy'] * w['Source_Accuracy'] +
        base['Impact_Score'] * w['Impact_Score'] +
        base['Acceleration_value'] * w['Acceleration_value'] +
        base['Location'] * w['Location'] +
        base['Industry_Risk'] * w['Industry_Risk'] +
        base['Frequency_Score'] * w['Frequency_Score']
    )
    base['Risk_Score'] = (num / weight_sum).clip(0,5).round(3)
    base['Weights'] = base['Risk_Score']

    print(f"[risk_weights] done: base = {base.shape} elapsed = {time.perf_counter()- t0:.1f}s", flush = True)

    return base

def get_release_by_tag(owner, repo, tag):
    url = f'https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}'
    r = requests.get(url, headers = gh_headers())
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()

def predict_risks(df):
    def load_model_bundle(owner, repo, tag, asset_name = 'model_bundle.pkl', local_cache_path = 'Model_training/artifacts/model_bundle.pkl'):
        P = Path(local_cache_path)
        P.parent.mkdir(parents = True, exist_ok=True)

        if P.exists() and P.stat().st_size > 0:
            return joblib.load(P)
        rel = get_release_by_tag(owner, repo, tag)
        if not rel:
            raise RuntimeError(f"Release tag {tag} not found")
        assets = rel.get('assets', []) or []
        asset = next((a for a in assets if a.get('name')==asset_name), None)
        if not asset:
            raise RuntimeError(f"Asset {asset_name} not found")

        asset_url =asset['url']
        headers = gh_headers()
        headers['Accept'] = 'application/octet-stream'
        r = requests.get(asset_url, headers = headers, timeout = 120)
        r.raise_for_status()
        P.write_bytes(r.content)
        return joblib.load(P)

    def soft_cosine_probs(vecs, label_emb):
        cos = vecs @ label_emb.T
        cos = (cos + 1.0) / 2.0
        denom = cos.sum(axis = 1, keepdims = True) + 1e-12
        return cos/denom
    def rule_route(text, label): 
        t = text.lower() 

        if any(k in t for k in ["hazing", "pledge", "fraternity", "sorority"]) and ("student" in t or "chapter" in t or "greek" in t): 
            return "Student Conduct Incident"
        if any(k in t for k in ["D.E.I.", "DEI"]):
            return "DEI Program Backlash"
        if label == "Vendor Cyber Exposure" and not any(k in t for k in ["vendor", "third-party", "saas", "hosting", "soc 2", "breach", "dpi a", "dpa", "pii", "cybersecurity", "supplier"]):
            if "ai" in t or "artificial intelligence" in t: 
                return "Artificial Intelligence Ethics & Governance"
            return label
        return label
        
    def predict_with_fallback(proba_lr, cos_all, prob_cut, margin_cut, tau, tau_gray, trained_labels, all_labels):
        top_idx = proba_lr.argmax(axis=1)
        top_val = proba_lr[np.arange(len(proba_lr)), top_idx]
        tmp = proba_lr.copy()
        tmp[np.arange(len(tmp)), top_idx] = -1
        second = tmp.max(axis=1)
        margin = top_val - second
        lr_mask = (top_val >= prob_cut) & (margin >= margin_cut)
    
        cos_all_max = cos_all.max(axis=1)
        cos_all_idx = cos_all.argmax(axis=1)
    
        lr_names  = np.array(trained_labels)[top_idx]
        cos_names = np.array(all_labels)[cos_all_idx]

        route = np.full(len(top_val), 'norisk', dtype = object)
        final = np.array(['No Risk']*len(top_val), dtype = object)

        route[lr_mask] = 'lr'
        final[lr_mask] = lr_names[lr_mask]

        cos_hi = (~lr_mask) & (cos_all_max >= tau)
        route[cos_hi] = "cos"
        final[cos_hi] = cos_names[cos_hi]

        gray = (~lr_mask) & (cos_all_max >= tau_gray) & (cos_all_max < tau)
        route[gray] = "gray"

    
        return {
        "final_names": final,
        "route": route,
        "lr_top_prob": top_val,
        "lr_top_idx": top_idx,
        "cos_all_idx": cos_all_idx,
        "cos_all_max": cos_all_max,
        "cos_names": cos_names
    }
    with open('Model_training/risks.json', 'r', encoding='utf-8') as f:
        risks_cfg = json.load(f)

    json_all_labels = [r['name'] for block in risks_cfg.get('new_risks', []) for _, items in block.items() for r in items]
    bundle = load_model_bundle(Github_owner, Github_repo, 'regression')
    clf = bundle['clf']
    scaler = bundle['scaler']
    pca = bundle['pca']
    le = bundle['label_encoder']
    trained_labels = bundle['trained_label_names']
    risk_defs = bundle['risk_defs']
    model_name = bundle['sentence_model_name']
    prob_cut = 0.60
    margin_cut = 0.10
    tau = 0.55
    numeric_factors = list(bundle['numeric_factors'])
    trained_label_txt = list(bundle['trained_label_text'])
    all_labels = json_all_labels
    all_label_txt = list(bundle['all_label_text'])


    df = df.copy()
    df = df.sort_values('Published_utc').drop_duplicates('Title', keep = 'last').reset_index(drop=True)
    df['University Label'] = pd.to_numeric(df['University Label'], errors = 'coerce').fillna(0).astype(int)
    mask_he = df['University Label'] == 1
    
    df['Title'] = df['Title'].fillna('').str.strip()

    df['Content'] = df['Content'].fillna('').str.strip()
    df['Text'] = (df['Title'] + '. ' + df['Title'] + '. ' + df['Content']).str.strip()

    df = df.reset_index(drop = True)

    if 'Predicted_Risks_new' in df.columns:
        todo_mask = (df['Predicted_Risks_new'].isna()) | (df['Predicted_Risks_new'].eq('')) | (df['Predicted_Risks_new'].eq('No Risk'))
    else:
        todo_mask = pd.Series(True, index=df.index)
    recent_cut = pd.Timestamp.now(tz='utc') - pd.Timedelta(days=365)
    df['Published_utc'] = pd.to_datetime(df['Published'], errors='coerce', utc = True)
    recent_mask = df['Published_utc'] >= recent_cut
    todo_mask &= recent_mask.fillna(False)
    todo_mask &= mask_he
    sub = df.loc[todo_mask].copy()
    texts = df.loc[todo_mask, 'Text'].tolist()
    
    
    print(f"[dbg] total rows: {len(df)}", flush = True)
    print(f"[dbg] parsable Published: {df['Published_utc'].notna().sum()}", flush = True)
    print(f"[dbg] recent (<=30d): {recent_mask.fillna(False).sum()}", flush = True)
    print(f"[dbg] to score (todo_mask): {todo_mask.sum()}", flush = True)
    change = texts
    if not texts:
        return df


    all_risks = [risk['name'] for group in risks_cfg['new_risks'] for risks in group.values() for risk in risks]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-mpnet-base-v2', device = device)
    # Encode articles and risks
    #article_embeddings = model.encode(texts, convert_to_numpy = True, normalize_embeddings = True, show_progress_bar=True,  batch_size=256 if device=='cuda' else 32)
    article_embeddings = model.encode(texts, convert_to_numpy = True, normalize_embeddings = True, show_progress_bar=True,  batch_size=256 if device=='cuda' else 32)
    A = article_embeddings
    C = np.zeros_like(A)
    X_text = np.hstack([article_embeddings, C])
    X_text_red = pca.transform(X_text) if pca is not None else X_text
    n = len(texts)
    if len(numeric_factors) == 0:
        num_scaled = np.zeros((n, 0))
    else:
        means = np.asarray(bundle['scaler'].mean_, dtype = float)
        num_mat = np.tile(means, (n, 1))
        num_scaled = bundle['scaler'].transform(num_mat)

    topic_ids = pd.to_numeric(sub.get('Topic'), errors = 'coerce').fillna(-1).to_numpy().reshape(-1, 1)
    topic_probs = pd.to_numeric(sub.get('Probability'), errors = 'coerce').fillna(0.0).to_numpy().reshape(-1,1)
    topic_col_name = 'Topic'
    top_ids = bundle.get('topic_top_ids', [])
    ohe_cols_expected = bundle.get('topic_ohe_cols', [])

    topic_raw = pd.to_numeric(sub.get(topic_col_name), errors = 'coerce').fillna(-1).astype(int)
    topic_binned = np.where(np.isin(topic_raw, top_ids), topic_raw, -1)

    topic_ohe = pd.get_dummies(pd.Series(topic_binned), prefix = 'topic', dtype = int)

    for col in ohe_cols_expected:
        if col not in topic_ohe:
            topic_ohe[col] = 0
    topic_ohe = topic_ohe[ohe_cols_expected].to_numpy()

    X_all = np.hstack([X_text_red, num_scaled, topic_ohe])

    proba = clf.predict_proba(X_all)

    avg_emb = article_embeddings
    avg_emb = avg_emb / (np.linalg.norm(avg_emb, axis=1, keepdims=True) + 1e-12)
    
    lbl_emb_all = model.encode(all_label_txt, show_progress_bar=True, normalize_embeddings=True, batch_size=256)
    
    cos_all = avg_emb @ lbl_emb_all.T


    out = predict_with_fallback(proba, cos_all, prob_cut, margin_cut, tau, 0.3, trained_labels, all_labels)
    sub['pred_source'] = out['route']
    sub['Predicted_Risks_new'] = out['final_names']
    gray_mask = (out['route']=='gray')

    if gray_mask.any():
        gray_idx = sub.index[gray_mask]
        gray_texts = sub.loc[gray_idx, 'Text'].tolist()

        label_list = json.dumps(all_labels + ['No Risk'])

        adjudicated = []
        for txt in gray_texts:
            prompt = f"""
            You are labeling articles to assess the institutional risk they pose to a higher education institution.
Return strictly JSON with: label.
Choose label from this CLOSED LIST ONLY (no other strings allowed):
{label_list}

Article:
{txt[:3000]}

Rules:
- If the article is not clearly about a risk to a US higher-education institution, return "No Risk".
- If the article is about sports results or leadership, return "No Risk"
- Prefer the most specific risk (e.g., "Lab Incident" instead of "Environmental Exposure").
- If the title refers to federal funding, ALWAYS return "Research Funding Disruption".
- DO NOT use "Unauthorized Access/Dat Breach" if the article does not refer to the digital space or cloud systems
- If guns/lockdown/active shooter/bombs/explosions on educational institutions → "Violence or Threats".
- If it is any lab materials spilled or lab subjects on the loose, research materials mishandled -> "Lab Incident"
- If hazing/Greek life/student misconduct → "Student Conduct Incident".
- If third-party/SaaS vendor breach or supplier compromise → "Vendor Cyber Exposure".
- If open storage / IAM / exposed endpoint → "Cloud Misconfiguration".
- If any policies or political interference is affecting school curricula and acivities, or if gender or race are mentioned in the context of academic programs or political policy changes -> "Policy or Political Interference" and NOT "Title IX/ADA Noncompliance"
- If the event is general AI use on campus policy/teaching → "Artificial Intelligence Ethics & Governance". This topic should ONLY be used if AI/artificial intelligence is in the article
- If none match confidently → "No Risk".
"""
            resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
            )
            raw = getattr(resp, "text", "").strip()
            # be robust to fencing
            m = re.search(r"\{.*\}", raw, flags=re.S)
            try:
                obj = json.loads(m.group(0) if m else raw)
                label = obj.get("label", "No Risk")
                if label not in (all_labels + ["No Risk"]):
                    label = "No Risk"
            except Exception:
                label = "No Risk"
            adjudicated.append(label)
        sub.loc[gray_idx, 'Predicted_Risks_new'] = adjudicated
        sub.loc[gray_idx, 'pred_source'] = 'gemini'
        sub.loc[gray_idx, 'Pred_cos_label_all'] = np.array(all_labels)[out['cos_all_idx'][gray_mask]]
        sub.loc[gray_idx, 'Pred_cos_score_all'] = out['cos_all_max'][gray_mask]
   
            
    sub_texts = sub['Text'].tolist()
    sub['Predicted_Risks_new'] = [rule_route(txt, lbl) for txt, lbl in zip(sub['Text'].tolist(), sub['Predicted_Risks_new'].to_list())]
    
    
    sub['Pred_LR_label'] = out['lr_top_prob']
    sub['Pred_cos_label_all'] = np.array(all_labels)[out['cos_all_idx']]
    sub['Pred_cos_score_all'] = out['cos_all_max']
    for col in ['pred_source', 'Predicted_Risks_new', 'Pred_LR_label', 'Pred_cos_label_all', 'Pred_cos_score_all']:
        df.loc[sub.index, col] = sub[col]

    return df
def track_over_time(df, week_anchor="W-MON", out_csv="Model_training/topic_trend.csv"):

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
        with open('Model_training/topics_BERT.json', 'r', encoding='utf-8') as f:
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


def call_gemini(prompt):
    GEMINI_API_KEY = os.getenv('PAID_API_KEY')
    client = genai.Client(api_key=GEMINI_API_KEY)
    return client.models.generate_content(model="gemini-2.0-flash", contents=[prompt])

# 🧠 Async article processor
@backoff.on_exception(backoff.expo,
                      (genai.errors.ServerError, requests.exceptions.ConnectionError),
                      max_tries=6,
                      jitter=None,
                      on_backoff=lambda details: print(
                          f"Retrying after error: {details['exception']} (try {details['tries']} after {details['wait']}s)", flush=True))
async def process_article(article, sem, batch_number=None, total_batches=None, article_index=None):
    async with sem:
        try:
            if batch_number is not None and total_batches is not None and article_index is not None:
                print(f"📦 Processing Batch {batch_number} of {total_batches} | Article {article_index}", flush=True)

            content = article['Content']
            title = article['Title']
            if pd.isna(content) or pd.isna(title):
                return None

            prompt = f"""
            Read the following title and content from the following article: 
            Title: {title}
            Content: {" ".join(str(content).split()[:400])}
            Task: Decide if this is SPECIFICALLY about higher education/university news or university funding in the UNITED STATES ONLY.
            Return a compact valid JSON with exactly these keys and no explanations:
            {{
              "Title": "same title",
              "Content": "same content",
              "University Label": 1 or 0
            }}
            
            Labeling rules:
            - Return 1 ONLY if the article reports higher-ed institution news in the United States.
            - Return 1 if the article reports a significant U.S. federal acion--such as an executive order, new law, government shutdown, funding decision--that directly or plausibly affects higher-education institutions, even if no specific university is named.
            - Return 1 if the article mentions Tulane University or clearly affects Tulane operations, funding, leadership, policy, legal exposure, or reputation.
            - Return 1 if a US Federal/State policy or enforcement action applies to multiple universities and plausibly impacts peer institutions like Tulane.
            - Return 0 otherwise.
            
            Clauses (IMPORTANT!!):
            - If the article is an executive order from the White House that affects education and higher education, return 1
            - If the article comes from the Tulane Hullabaloo, return 1 if it reports any news that could be a risk to the organization
            - If the article is a professional/personal profile or staff/alumni spotlight (e.g., “Meet X…”, “X is a [role] at…”, bio pages, team/staff directory, “welcomes X to the team”, career journey, awards unrelated to institutional policy/funding) → return 0.
            - Return 1 for leadership announcements ONLY if they clearly indicate institutional impact (e.g., new president/provost with stated policy/strategy changes for the university). Otherwise return 0. (Hints that indicate a profile: “About [Name]”, “Meet [Name]”, “joined [org] as…”, “Biography/Profile”, “Our Team/Staff Directory”, CV-like education + roles with no institutional news.)
            - If the article is not in English, return 0.
            - If the article talks about general medical/healthcare advances that in no way impact university operations, return 0
            - If the article talks about sports news, matches, sports results, return 0
            - If the article is a news wrap, a podcast, or a video, return 0
            - If the article is a general scientific discovery, return 0
            
            Output must be exactly:
            {{
              "Title": "same title",
              "Content": "same content",
              "University Label": 0 or 1
            }}
            """

            response = await asyncio.to_thread(call_gemini, prompt)
            if hasattr(response, "text") and response.text:
                response_text = response.text
                json_str = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
                raw = json_str.group(1) if json_str else response_text
                raw = (raw.replace("“", '"').replace("”", '"')
                   .replace("’", "'")
                   .replace("\n", " "))
                raw = re.sub(r",\s*}", "}", raw)
                raw = re.sub(r",\s*]", "]", raw)


                try:
                    rec = json.loads(raw)
                except json.JSONDecodeError as e1:
                    try:
                        rec = ast.literal_eval(raw)
                    except Exception as e2:
                        print(f"⚠️ JSON decode fallback error: {e1} | Eval error: {e2}", flush=True)
                        return None
                title = rec.get('Title') or rec.get('title') or str(title)
                content = rec.get('Content') or rec.get('content') or str(content)

                ulabel = rec.get('University Label')

                if ulabel is None:
                    ulabel = rec.get('university_label') or rec.get('University_label') or rec.get('university label') or 0

                try:
                    ulabel = int(ulabel)
                    ulabel = 1 if ulabel ==1 else 0
                except Exception:
                    ulabel = 0
                return {'Title': str(title), "Content": str(content), 'University Label': ulabel}
        except Exception as e:
            print(f"🔥 Uncaught error in article {article_index} of batch {batch_number}: {e}", flush=True)
            return None

    # 🚀 Async batch runner
async def university_label_async(articles, batch_size=15, concurrency=10):
    sem = asyncio.Semaphore(concurrency)
    tasks = []

    total_articles = len(articles)
    total_batches = (total_articles + batch_size - 1) // batch_size
    for start in range(0, total_articles, batch_size):
        batch_number = (start // batch_size) + 1
        print(f"🚚 Starting Batch {batch_number} of {total_batches}", flush=True)
        batch = articles.iloc[start:start+batch_size]
        for i, (_, row) in enumerate(batch.iterrows()):
            tasks.append(process_article(row, sem,
                                         batch_number=batch_number,
                                         total_batches=total_batches,
                                         article_index=i+1))

    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

def load_university_label(new_label):
    all_articles = new_label.copy()
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=5)
    base_pub = all_articles.get('Published')
    all_articles['Published_utc'] = pd.to_datetime(base_pub, errors='coerce', utc=True)
    recent = all_articles[all_articles['Published_utc'] >= cutoff]

    try:
        existing = pd.read_csv('BERTopic_before.csv')
        labeled_titles = set(existing['Title']) if 'Title' in existing else set()
    except FileNotFoundError:
        existing = pd.DataFrame(columns=['Title', 'University Label'])
        labeled_titles = set()

    if not existing.empty and 'University Label' in existing.columns:
        existing_clean = (
            existing[['Title', 'University Label']]
            .dropna(subset=['University Label'])
            .drop_duplicates(subset=['Title'], keep='last')
        )
        all_articles = all_articles.merge(
            existing_clean[['Title', 'University Label']],
            on='Title', how='left',
            suffixes=('', '_prev')
        )

    new_articles = recent[~(recent['Title'].isin(labeled_titles))].copy()
    print(new_articles[['Title', 'Published_utc']].head())
    new_articles = new_articles[~(new_articles['University Label'] == 1)]
    print(f"🔎 Total articles: {len(recent)} | Unlabeled: {len(new_articles)}", flush=True)

    results = asyncio.run(university_label_async(new_articles))

    if results:
        labels_df = pd.DataFrame(results)[['Title', 'University Label']]
        labels_df['Title'] = labels_df['Title'].astype(str).str.strip()
        new_articles['Title'] = new_articles['Title'].astype(str).str.strip()
        missing_titles = set(new_articles['Title']) - set(labels_df['Title'])
        if missing_titles:
            missing_df = pd.DataFrame({
                'Title': list(missing_titles),
                'University Label': [0] * len(missing_titles)
            })
            labels_df = pd.concat([labels_df, missing_df], ignore_index=True)

        all_articles = all_articles.merge(labels_df, on='Title', how='left', suffixes=('', '_new'))


        prev_cols = [c for c in all_articles.columns
                     if c.startswith('University Label_prev')]
        if prev_cols:

            combined_prev = all_articles[prev_cols].bfill(axis=1).iloc[:, 0]

            all_articles.drop(columns=prev_cols, inplace=True, errors='ignore')

            all_articles['University Label_prev'] = combined_prev
        else:

            all_articles['University Label_prev'] = pd.NA


        label_cols = [c for c in all_articles.columns
                      if c.startswith('University Label') and not c.startswith('University Label_prev')]
        if not label_cols:

            all_articles['University Label'] = pd.NA
        else:

            combined_label = all_articles[label_cols].bfill(axis=1).iloc[:, 0]

            all_articles.drop(columns=label_cols, inplace=True, errors='ignore')

            all_articles['University Label'] = combined_label


        mask_have_prev = all_articles['University Label_prev'].notna()

        all_articles.loc[mask_have_prev & (all_articles['University Label'] == 0),
                         'University Label'] = all_articles.loc[mask_have_prev & (all_articles['University Label'] == 0),
                                                               'University Label_prev']


        all_articles.drop(columns=['University Label_prev'], inplace=True, errors='ignore')

        if not existing.empty:
            combined = pd.concat([existing, labels_df], ignore_index=True)
        else:
            combined = labels_df
    else:
        combined = existing

    combined.to_csv('BERTopic_before.csv',
                    columns=['Title', 'University Label'],
                    index=False)

    return all_articles








def save_dataset_to_releases(df:pd.DataFrame, local_cache_path = 'Model_training/BERTopic_results2.csv.gz'):
    buf= io.BytesIO()
    with gzip.GzipFile(fileobj = buf, mode = 'wb') as gz:
        gz.write(df.to_csv(index = False).encode('utf-8'))
    gz_bytes = buf.getvalue()

    Path(local_cache_path).parent.mkdir(parents = True, exist_ok = True)
    with open(local_cache_path, 'wb') as f:
        f.write(gz_bytes)

    rel = ensure_release(Github_owner, Github_repo, Release_tag, GITHUB_TOKEN)
    upload_asset(Github_owner, Github_repo, rel, Asset_name, gz_bytes, GITHUB_TOKEN)


def load_midstep_from_release(local_cache_path = 'Model_training/BERTopic_Streamlit.csv.gz'):
    rel = get_release_by_tag(Github_owner, Github_repo, Release_tag)
    if rel:
        asset = next((a for a in rel.get('assets', []) if a['name']=='BERTopic_Streamlit.csv.gz'), None)
        if asset:
            r = requests.get(asset['browser_download_url'], timeout = 60)
            if r.ok:
                return pd.read_csv(io.BytesIO(r.content), compression = 'gzip')
    P = Path(local_cache_path)
    if P.exists():
        return pd.read_csv(local_cache_path, compression='gzip')
    return pd.DataFrame()

##Assign topics and probabilities to new_df
#print("✅ Starting transform_text on new data...", flush=True)
#topic_model.calculate_probabilities = True
#new_df = transform_text(df)
##Fill missing topic/probability rows in the original df
#for c in ['Topic', 'Probability']:
#    if c not in df.columns:
#        df[c] = np.nan
#    
#mask = (df['Topic'].isna()) | (df['Probability'].isna())
#df.loc[mask, ['Topic', 'Probability']] = new_df[['Topic', 'Probability']]
#df[['Topic', 'Probability']] = new_df[['Topic', 'Probability']]
##Save only new, non-duplicate rows
#print("✅ Saving new topics to CSV...", flush=True)
#df_combined = save_new_topics(df, new_df)
#df_combined['Probability'] = pd.to_numeric(df_combined['Probability'], errors = 'coerce')
##
##Double-check if there are still unmatched (-1) topics and assign a temporary model to assign topics to them
#def coerce_pub_utc(x):
#    if pd.isna(x):
#        return pd.NaT
#    if isinstance(x, (int, float)):
#        if x > 1e12:  
#            return pd.to_datetime(x, unit="ms", errors="coerce", utc=True)
#        if x > 1e9: 
#            return pd.to_datetime(x, unit="s", errors="coerce", utc=True)
#    sx = str(x)
#    sx = re.sub(r'\s(EST|EDT|PDT|CDT|MDT|GMT)\b', '', sx, flags=re.I)
#    return pd.to_datetime(sx, errors="coerce", utc=True)
#print("✅ Running double-check for unmatched topics (-1)...", flush=True)
#cutoff_utc = pd.Timestamp(datetime.utcnow() - timedelta(days = 120), tz = 'utc')
#df_combined['Published'] = df_combined['Published'].apply(coerce_pub_utc)
#print(f"Length of dataset: {len(df_combined)}", flush = True)
#print(f"Length of recalculated topic names: {len(df_combined[df_combined['Probability'] < 0.15])}", flush = True)
#low_conf_mask = df_combined['Probability'] < 0.15
#df_combined.loc[low_conf_mask, 'Topic'] = -1
#

#atomic_write_csv('Model_training/Step0.csv.gz', df_combined, compress = True)
#upload_asset_to_release(Github_owner, Github_repo, Release_tag, 'Model_training/Step0.csv.gz', GITHUB_TOKEN)
##df_combined = load_midstep_from_release()
#recent_df = df_combined[df_combined['Published'].notna() & (df_combined['Published'] >= cutoff_utc)].copy()
#temp_model, topic_ids = double_check_articles(recent_df)
##If there are unmatched topics, name them using Gemini
#print("✅ Checking for unmatched topics to name using Gemini...", flush=True)
#if temp_model and topic_ids:
#    topic_name_pairs = get_topic(temp_model, topic_ids)
#    existing_risks_json(topic_name_pairs, temp_model)
###Assign weights to each article
##results_df = load_midstep_from_release()
#df_combined = load_university_label(df_combined)
#atomic_write_csv('Model_training/initial_label.csv.gz', df_combined, compress = True)
#upload_asset_to_release(Github_owner, Github_repo, Release_tag, 'Model_training/initial_label.csv.gz', GITHUB_TOKEN)
##df_combined = load_midstep_from_release()
#results_df = predict_risks(df_combined)
#results_df['Predicted_Risks'] = results_df.get('Predicted_Risks_new', '')
#print("✅ Applying risk_weights...", flush=True)
#atomic_write_csv('Model_training/Step1.csv.gz', results_df, compress = True)
#upload_asset_to_release(Github_owner, Github_repo, Release_tag, 'Model_training/Step1.csv.gz', GITHUB_TOKEN)
#results_df = load_midstep_from_release()

#results_df = results_df.drop(columns = ['Acceleration_value_x', 'Acceleration_value_y'], errors = 'ignore')

#results_df['Predicted_Risks'] = results_df.get('Predicted_Risks_new', results_df.get('Predicted_Risks', ''))
#df = risk_weights(results_df)
#print("Finished assigning risk weights", flush = True)
#df = df.drop(columns = ['University Label_x', 'University Label_y'], errors = 'ignore')
#print("Saving BERTopic_results2.csv.gz", flush = True)
#atomic_write_csv("Model_training/BERTopic_results2.csv.gz", df, compress=True)
#print('Uploading to releases', flush=True)
#upload_asset_to_release(Github_owner, Github_repo, Release_tag, 'Model_training/BERTopic_results2.csv.gz', GITHUB_TOKEN)
#print("Saving dataset for Streamlit", flush= True)
#df_streamlit = df[df['University Label'] == 1]
#atomic_write_csv("Model_training/BERTopic_Streamlit.csv.gz", df_streamlit, compress = True)
#upload_asset_to_release(Github_owner, Github_repo, Release_tag, 'Model_training/BERTopic_Streamlit.csv.gz', GITHUB_TOKEN)
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
    df = load_midstep_from_release()
    df = ensure_risk_scores(df)

    if Path('Model_training/Articles_with_Stories.csv.gz').exists():
        old_df = pd.read_csv('Model_training/Articles_with_Stories.csv.gz', compression='gzip')
    else:
        old_df = pd.DataFrame(columns=df.columns.tolist() + ['story_id'] + ['_key'])




    df = df[df['Published_utc'].notna()]
    df['orig_idx'] = df.index
    already_labeled = old_df.dropna(subset=['story_id'])
    cutoff = old_df['Published_utc'].max()
    new_articles = df[df['Published_utc'] > cutoff].copy()
    new_articles['story_id'] = np.nan

    new_articles['_key'] = list(zip(new_articles['Title'], new_articles['Link']))
    already_labeled['_key'] = list(zip(already_labeled['Title'], already_labeled['Link']))

    new_articles = new_articles[
        ~new_articles['_key'].isin(already_labeled['_key'])
    ].drop(columns='_key')

    df = pd.concat([already_labeled, new_articles], ignore_index = True)

    if Path('Model_training/Story_Clusters.csv.gz').exists():
        stories_df = pd.read_csv('Model_training/Story_Clusters.csv.gz', compression='gzip')
    else:
        stories_df = pd.DataFrame(columns=[
        'story_id', 'canonical_title', 'canonical_link', 'canonical_published',
        'article_count', 'first_seen', 'last_seen'
    ])

    if Path('Model_training/Canonical_Stories_with_Summaries.csv').exists():
        canonical_titles = pd.read_csv('Model_training/Canonical_Stories_with_Summaries.csv')
    else:
        canonical_titles = pd.DataFrame(columns=['story_id', 'canonical_title'])

    stories_df = stories_df.drop(columns = ['canonical_title'], errors = 'ignore')
    stories_df = stories_df.merge(
        canonical_titles[['story_id', 'canonical_title']],
        on = 'story_id',
        how = 'left'
    )

    story_id_counter = int(stories_df['story_id'].max()) + 1 if not stories_df.empty else 1
    open_stories = []

    def norm_text(x):
        x = (x or '')
        x = re.sub(r'\s+', ' ', x).strip()
        return x

    open_stories = []

    article_embeddings = model.encode(df['Title'].fillna('').tolist(), convert_to_numpy = True, normalize_embeddings = True)
    df['story_embeddings'] = list(article_embeddings)

    articles_by_story = (df[df['story_id'].notna()].groupby('story_id').apply(lambda g: g.to_dict('records')).to_dict())


    for _, row in stories_df.iterrows():
        sid = int(row['story_id'])

        rows = articles_by_story.get(sid, [])

        if not rows:
            print(f"Skipping story {sid}: no articles found")
            continue

        texts = [norm_text(f"{str(r.get('Title') or '')} {str(r.get('Summary') or '')}") for r in rows]

        first_seen = min(pd.to_datetime(r['Published_utc'], errors = 'coerce') for r in rows)
        last_seen = max(pd.to_datetime(r['Published_utc'], errors = 'coerce') for r in rows)
        centroid = model.encode(texts, convert_to_numpy = True, normalize_embeddings = True).mean(axis = 0)
        n = len(rows)
        open_stories.append({
            "id": sid,
            "centroid": centroid,
            "rows": rows,
            "n": n,
            "first_seen": pd.to_datetime(first_seen, errors = 'coerce'),
            "last_seen": pd.to_datetime(last_seen, errors = 'coerce'),
            "canonical_title": row.get('canonical_title', None)
        })



    def build_story_clusters(df, open_stories, story_id_counter, stories_df, min_sim = 0.52):
        df = df.copy()


        df['text_for_embedding'] = (df['Title'].fillna('') + ' ' + df['Summary'].fillna('')).apply(norm_text)
        df['date_bucket'] = df['Published_utc'].dt.floor('D')
        df.sort_values(by='Published_utc', inplace=True)

        embeddings = model.encode(df['text_for_embedding'].tolist(), convert_to_numpy=True, batch_size=64,
                    show_progress_bar=True,
                    normalize_embeddings=True)

        story_rows = []
        article_story_ids = []
    

    
        MAX_GAP_DAYS = 21
    

        df = df.copy()
        for pos, (idx, row) in enumerate(df.iterrows()):
            best_sim = -1
            best_story = None
            embed_i = embeddings[pos]
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

            existing = stories_df.loc[
                stories_df["story_id"] == s["id"], "canonical_title"
            ]

            canonical_title = (
                existing.values[0]
                if not existing.empty and pd.notna(existing.values[0])
                else can_row["Title"]
            )

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
            stories_df[col] = stories_df[f"{col}_new"].combine_first(stories_df[col])
    
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
    articles_with_stories, stories_df, open_stories = build_story_clusters(df, open_stories, story_id_counter, stories_df, min_sim = 0.6)
    articles_with_stories.to_csv("Model_training/Articles_with_Stories.csv.gz", index = False, compression = 'gzip')

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
    
    stories_df.to_csv("Model_training/Story_Clusters.csv.gz", index = False, compression = 'gzip')
    
    api_key = os.getenv('PAID_API_KEY')
    client = genai.Client(api_key = api_key)
    
    df = pd.read_csv('Model_training/Articles_with_Stories.csv.gz', compression='gzip')
    df = df.drop_duplicates(subset=["Title", "Published_utc"], keep="last")
    df_stories = pd.read_csv('Model_training/Story_Clusters.csv.gz', compression='gzip')
    df_stories.to_csv('Model_training/Story_Clusters_backup.csv', index=False)
    
    
    if Path('Model_training/Canonical_Stories_with_Summaries.csv').exists():
        canonical_titles = pd.read_csv('Model_training/Canonical_Stories_with_Summaries.csv')
        canonical_titles['canonical_source'] = canonical_titles['canonical_source'].fillna('unlabeled')
    else:
        canonical_titles = pd.DataFrame(columns=['story_id', 'canonical_title', 'summary', 'average_risk_score', 'average_recency', 'articles', 'canonical_source'])
    df_stories = df_stories.merge(
        canonical_titles[['story_id', 'canonical_title', 'canonical_source']],
        on='story_id',
        how='left',
        suffixes = ('', '_new'),
        validate='one_to_one'
    )
    df_stories['canonical_title'] = df_stories['canonical_title_new'].fillna(df_stories['canonical_title'])
    df_stories = df_stories.drop(columns=['canonical_title_new'], errors='ignore')
    df_stories['canonical_title'] = df_stories['canonical_title'].astype('string')
    df_stories = (
        df_stories
        .sort_values("last_seen", ascending=False)
        .drop_duplicates(subset=["story_id"], keep="first")
    )
    
    
    df = df[['Title', 'story_id', 'Published_utc']]
    df['Published_utc'] = pd.to_datetime(df['Published_utc'], errors='coerce', utc = True)
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=90)
    
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
    
    
    grouped = merged.groupby('story_id')
    
    canonical_stories = []
    for story_id, group in grouped:
        if group.shape[0] >= 2:
            canonical_source = group['canonical_source'].iloc[0]
            if (story_id in canonical_titles['story_id'].values
                and canonical_source == 'gemini'):
                continue
            existing_title = group['canonical_title'].iloc[0]
            print(f"Story ID: {story_id}")
            titles = group['Title'].tolist()
            prompt = f"For each of these groups, read the titles of all the articles and generate a concise canonical title that summarizes the main topic of the story. Here's the titles: {titles}"
            try:
                response = client.models.generate_content(
                    model = 'gemini-2.5-flash',
                    contents = prompt)
            except ClientError as e:
                 if "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = 60  
                    retry_delay_match = re.search(r"'retryDelay': '(\d+)s'", str(e))
                    if retry_delay_match:
                        wait_time = int(retry_delay_match.group(1))  # Use API's recommended delay#
    
                    print(f"⚠️ API quota exceeded. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
            prompt2 = f"Based on the following titles: {titles}, generate a 100 word summary that takes into account all the titles."
            response2 = client.models.generate_content(
                model = 'gemini-2.5-flash',
                contents = prompt2
            )
            canonical_title = response.text.strip()
            print(f"Canonical Title: {canonical_title}")
            summary = response2.text.strip()
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
    if canonical_stories:
        new_df = pd.DataFrame(canonical_stories)
        try:
            existing = pd.read_csv("Model_training/Canonical_Stories_with_Summaries.csv")
            final = pd.concat([existing, new_df], ignore_index=True)
            final = final.drop_duplicates(subset=['story_id'], keep='last')
            final.to_csv("Model_training/Canonical_Stories_with_Summaries.csv", index=False)
        except FileNotFoundError:
            final = new_df
            final.to_csv("Model_training/Canonical_Stories_with_Summaries.csv", index=False)
    else:
        print("No new titles were generated.")
    
    return None

##Show the articles over time
stories = build_stories()
def safe_mode(series):
    s = series.dropna()
    return s.mode().iloc[0] if not s.empty else None
articles = pd.read_csv("Model_training/Articles_with_Stories.csv.gz", compression = 'gzip')
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
canonical = pd.read_csv("Model_training/Canonical_Stories_with_Summaries.csv")

canonical = canonical.merge(story_scores, on = "story_id", how = 'left', validate= "one_to_one")
canonical.to_csv("Model_training/Canonical_stories_with_Summaries.csv", index = False)
articles = load_midstep_from_release()
articles = ensure_risk_scores()
articles = articles.drop_duplicates(subset = ['Title', 'Link'], keep = 'last')
article_story_map = pd.read_csv("Model_training/Articles_with_Stories.csv.gz", compression = 'gzip')
article_story_map = article_story_map.drop_duplicates(subset = ['Title', 'Link'], keep = 'last')
canonical = pd.read_csv("Model_training/Canonical_Stories_with_Summaries.csv")
score_cols = ["avg_risk_score", "avg_frequency", "avg_recency"]
stories_df = pd.read_csv(
    "Model_training/Story_Clusters.csv.gz",
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

dashboard_stories.to_csv("Model_training/dashboard_stories.csv.gz", compression = 'gzip')
dropdown_table.to_csv("Model_training/dashboard_dropdown.csv.gz", compression = 'gzip')
standalone_articles.to_csv("Model_training/dashboard_articles.csv.gz", compression = 'gzip')

articles_only = articles[articles['story_articles_count']<3].copy()
print("Articles over time", flush = True)
#
track_over_time(df)
