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

df = df[~(df['Source']=="Economist")]
df['Text'] = df['Title'] + '. ' + df['Content']
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
        docs = docs[:5]
        keywords = ', '.join([word for word, _ in words])
        doc_list = '\n'.join([f"- {doc}" for doc in docs])
        block = (
            f"---\n"
            f"TopicID: {topic}\n"
            f"Keywords: {keywords}\n"
            f"Representative Documents: {doc_list}\n"
        )
        topic_blocks.append((topic, block))

    # Chunk your topic blocks (e.g., 5 topics per call)
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

        tokens_estimate = estimate_tokens(prompt)  # ✅ Defined here BEFORE it's used
        print(f"🔹 Sending prompt with approx {int(tokens_estimate)} tokens...")
        if tokens_estimate > 10000:
            print("⚠️ Prompt too large, consider lowering chunk_size!")
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                response = client.models.generate_content(model="gemini-2.5-pro", contents=[prompt])
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







bert_csv = Path('BERTopic_results.csv')
if bert_csv.exists():
    bert_art = pd.read_csv(bert_csv, encoding='utf-8')
    df = pd.concat([df, bert_art], ignore_index=True).drop_duplicates(subset=['Title','Content'], keep='last')
else:
    print("ℹ️ BERTopic_results.csv not found; proceeding with current df.")


if 'Source' not in df.columns:
    df['Source'] = ''

# Convert NaN/None to empty string, keep as string dtype
df['Source'] = df['Source'].astype('string').fillna('')

def transform_text(texts):
    texts = texts.copy()
    texts = texts.drop(columns = 'Topic')
    print(f"Transforming {len(texts)} articles in batches...")
    all_topics, all_probs = [], []
    batch_size = 100  # or smaller
    texts_list = texts['Text'].tolist()

    for i in range(0, len(texts_list), batch_size):
        batch = texts_list[i:i+batch_size]
        topics, probs = topic_model.transform(batch)
        all_topics.extend(topics)
        all_probs.extend(probs)
        print(f"✅ Transformed batch {i//batch_size + 1}/{(len(texts_list) // batch_size) + 1}")
    for i, (t,p) in enumerate(zip(all_topics, all_probs)):
        if t == -1 and p is not None:
            best = p.argmax()
            if p[best] >= 0.1:
                all_topics[i] = int(best)
    if any(t == -1 for t in all_topics):
        all_topics = topic_model.reduce_outliers(texts_list, all_topics, strategy = 'embeddings', threshold = 0.3)

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

    # Build centroids from representative docs (NOT remaining_idx)
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
        E = encode([texts_list[i] for i in remaining_idx], embedder)  # <-- pass embedder
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
            assigned_probs.append(float(p[t]))   # pick prob of assigned topic
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
                               usecols=None, dtype=str):
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

    pieces = [p for p in [on_disk, existing_df, unique_new] if not (isinstance(p, pd.DataFrame) and p.empty)]
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
        init='random',          # <-- avoids spectral eigendecomposition
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

    existing_topic_names = [t['name'] for t in topics if 'name' in t]
    #existing_embeddings = model.encode(existing_topic_names, convert_to_tensor=True)

    # Compare each new name against known topics

    # Merge docs/keywords into matched existing topics

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
    # Build a **list** of names aligned with existing_unmatched indices
    unmatched_names = []
    index_map = []  # map from names-list index -> existing_unmatched index
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
        new_emb = model.encode([name], convert_to_tensor=True)       # (1, d)
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
            to_upsert_unmatched.append({
                'topic': topic_id,
                'name': name,
                'keywords': new_keywords,
                'documents': new_docs
            })
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

    # Sources accuracy map (string name -> numeric 0..5)
    accuracy_map = {}
    for s in risks_cfg.get('sources', []):
        name = str(s.get('name', '') or '')
        acc = s.get('accuracy', 0) or 0
        accuracy_map[name] = acc

    # Risk level map (risk name -> level 0..5), supports string levels too
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

    higher_ed_dict = risks_cfg.get('HigherEdRisks', None)  # optional spaCy phrase dict

    # ---------- Base sanitation ----------
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
    base['Published_raw'] = base['Published'] 
    base['Published'] = base['Published'].apply(_coerce_pub) 
    if pd.api.types.is_datetime64tz_dtype(base['Published']): 
        base['Published'] = base['Published'].dt.tz_convert('UTC').dt.tz_localize(None) 
    # ---------- Per-article features (computed once, broadcast to base rows) ---------- 
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
        # strip brackets if list-string like "['X', 'Y']"
        s = re.sub(r'^\[|\]$', '', s)
        # split on ; or , and strip quotes/spaces
        s = re.split(r'[;,]', s)[0].strip().strip("'\"")
        return s
    
    base['_RiskList'] = cand.apply(_first_label)
    base['Risk_item'] = np.where(base['_RiskList'].eq(''), 'No Risk', base['_RiskList'])

    #base = base.explode('_RiskList', ignore_index=False).rename(columns={'_RiskList':'Risk_item'})
    #mask = exploded['Risk_item'].notna() & (exploded['Risk_item'].astype(str).str.strip()!='')
    #exploded = exploded[mask].copy()
    #exploded['Risk_norm'] = exploded['Risk_item'].astype(str).str.strip().str.lower()


    # Published -> datetime (robust coercion)




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

        # article weight
        art_w = 1.0
        if 'Probability' in fx.columns:
            art_w = pd.to_numeric(fx['Probability'], errors='coerce').fillna(0.0).clip(0, 1)

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

        # ensure expected cols exist even if tr is empty
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

        if 'recency_score_tr' not in enriched.columns:
            enriched['recency_score_tr'] = 0.0

        alpha = 0.7
        enriched['Recency_TR_Blended'] = (
            alpha * enriched['recency_score_tr'].fillna(0.0)
            + (1 - alpha) * enriched['article_freshness']
        ).clip(0, 1)

        return enriched
    t_rec = time.perf_counter()
    print("attach_topic_risk_recency() start", flush = True)
    base = attach_topic_risk_recency(base) 
    base['Recency'] = (base['Recency_TR_Blended'] * 5).round(2)
    print("[recency] attached in {time.perf_counter()-t_rec:.1f}s", flush = True)

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

    ##acceleration calculation
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
                lo = max(0, i - min(k, i) + 1)  # smaller early windows allowed
                xi = x[lo:i+1]; yi = counts[lo:i+1].astype(float)
                if len(xi) >= 2:  # allow 2 points early
                    m, _ = np.polyfit(xi, yi, 1)
                    out[i] = m
            return out

        ts['Slope'] = ts.groupby('Risk_item', group_keys = False)['n'].apply(lambda g: pd.Series(slope(g.values, k=6), index = g.index)).astype(float)

        def normalize_groupwise(s, by):
            # per-risk 95th percentile cap
            return s.groupby(by, group_keys=False).rank(pct = True).fillna(0.0)

        ts['emwa_norm']  = normalize_groupwise(ts['EMWA'].clip(lower=0),  ts['Risk_item'])
        ts['slope_norm'] = normalize_groupwise(ts['Slope'].clip(lower = 0), ts['Risk_item'])

        w_emwa, w_slope = 0.6, 0.4
        ts['accel_score'] = (w_emwa*ts['emwa_norm'] + w_slope * ts['slope_norm']).clip(0,1)

        short = ts.groupby('Risk_item')['Week'].transform('count') < 4
        ts.loc[short, 'accel_score'] *= 0.6

        # ---- Sentiment acceleration (added) ----
        if 'Sentiment Score' not in base.columns:
            base['Sentiment Score'] = 0.0
        ts_sent = (
            base.loc[base['Week'].notna()]
            .groupby(['Risk_item','Week'])
            .agg(sent_mean=('Sentiment Score','mean'))
            .reset_index()
            .sort_values(['Risk_item','Week'])
        )
        ts_sent['sent_flipped'] = -ts_sent['sent_mean']
        ts_sent['sent_ewma'] = ts_sent.groupby('Risk_item')['sent_flipped'].transform(
            lambda s: s.ewm(span=4, adjust=False).mean()
        )
        ts_sent['sent_delta'] = ts_sent.groupby('Risk_item')['sent_ewma'].diff().fillna(0.0)
        ts_sent['sent_slope'] = ts_sent.groupby('Risk_item', group_keys=False)['sent_flipped'] \
            .apply(lambda g: pd.Series(slope(g.values, k=6), index=g.index)) \
            .astype(float)
        ts_sent['sent_delta_norm'] = normalize_groupwise(ts_sent['sent_delta'], ts_sent['Risk_item'])
        ts_sent['sent_slope_norm'] = normalize_groupwise(ts_sent['sent_slope'], ts_sent['Risk_item'])
        w_sent_delta, w_sent_slope = 0.6, 0.4
        ts_sent['accel_score_sent'] = (w_sent_delta*ts_sent['sent_delta_norm'] + w_sent_slope*ts_sent['sent_slope_norm']).clip(0,1)

        ts = ts.merge(ts_sent[['Risk_item','Week','accel_score_sent']], on=['Risk_item','Week'], how='left')
        ts['accel_score_sent'] = ts['accel_score_sent'].fillna(0.0)

        # blend volume accel with sentiment accel (keep same thresholds below)
        w_vol, w_sent = 0.7, 0.3
        ts['accel_score'] = (w_vol*ts['accel_score'] + w_sent*ts['accel_score_sent']).clip(0,1)
        # ---- end sentiment acceleration (added) ----


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




    # Location (entities preferred; fallback text)


    # ---------- Explode to one row per risk ----------


    # ---------- Frequency_Score per risk (qcut over counts) ----------
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

    # ---------- Industry_Risk via spaCy matches (per article, then applied per risk) ----------
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

    # Tulane mentions per week (for lag pressure)
    tulane_week = (
        base.loc[base['_tulane_flag'] & base['Week'].notna()]
        .groupby('Week')
        .size()
        .rename('tulane_mentions')
        .reset_index()
    )
    agg = agg.merge(tulane_week, on='Week', how='left').fillna({'tulane_mentions': 0})

    # Exponential decay for old peer activity (half-life ≈ 21 days)
    if not agg.empty:
        week_max = agg['Week'].max()
        agg['days_ago'] = (week_max - agg['Week']).dt.days.clip(lower=0)
        lam = np.log(2.0) / 21.0
        agg['decay_w'] = np.exp(-lam * agg['days_ago'])

        # Weighted peer index: substantial > moderate, decay old events, downweight if Tulane already active
        agg['peer_index'] = agg['decay_w'] * (2 * agg['peers_sub'] + 1 * agg['peers_mod'])
        agg['sector_pressure'] = agg['peer_index'] / (1.0 + agg['tulane_mentions'])

        # Robust scale → 0–5
        lo, hi = np.percentile(agg['sector_pressure'], [5, 95]) if agg['sector_pressure'].notna().any() else (0.0, 1.0)
        rng = max(1e-12, hi - lo)
        agg['Industry_Risk_Peer'] = (((agg['sector_pressure'] - lo) / rng).clip(0, 1) * 5).round().astype(int)
    else:
        agg['Industry_Risk_Peer'] = 0

    # Merge back per row
    base = base.drop(columns=['Industry_Risk_Peer'], errors='ignore')
    base = base.merge(agg[['Week', 'Industry_Risk_Peer']], on='Week', how='left')
    base['Industry_Risk_Peer'] = base['Industry_Risk_Peer'].fillna(0).astype(int)

    # Final Industry_Risk = max(presence, peer)
    base['Industry_Risk'] = np.maximum(base['Industry_Risk_Presence'], base['Industry_Risk_Peer']).astype(int)
    print('Industry risk created', flush = True)



    # ---------- Impact_Score per risk ----------
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

    # ---------- Final blended Risk_Score (0..5) per (article × risk) ----------
    w = {
        'Recency': 0.15,
        'Source_Accuracy': 0.10,
        'Impact_Score': 0.35,
        'Acceleration_value': 0.25,
        'Location': 0.05,
        'Industry_Risk': 0.05,
        'Frequency_Score': 0.05
    }
    weight_sum = sum(w.values())  # 0.90

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
    base['Weights'] = base['Risk_Score']  # back-compat

    # helpful: keep original full risk list too
    base['Predicted_Risk_Single'] = base['Risk_item']

    # stable ID for joining back if needed
    base = base.reset_index().rename(columns={'index':'ArticleID'})
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
        if any(k in t for k in ["shooting", "shots fired", "gunfire", "killed", "wounded", "lockdown", "shelter-in-place", "active shooter", "homecoming"]): 
            return "Violence or Threats" 
        if any(k in t for k in ["hazing", "pledge", "fraternity", "sorority"]) and ("student" in t or "chapter" in t or "greek" in t): 
            return "Student Conduct Incident" # News about “AI at <university>” (strategy/ethics) shouldn't be Vendor Cyber Exposure 
        if "artificial intelligence" in t or "ai " in t or " generative ai" in t: 
            if any(k in t for k in ["teaching", "grading", "policy", "ethics", "bias", "governance", "academic integrity", "faculty", "students"]): 
                return "Artificial Intelligence Ethics & Governance" # generic campus modernization 
            if any(k in t for k in ["digital transformation", "modernization", "workflow", "automation"]): 
                return "Rapid Speed of Disruptive Innovation" # Vendor Cyber Exposure should only trigger on vendor/SaaS/security words 
        if label == "Vendor Cyber Exposure" and not any(k in t for k in ["vendor", "third-party", "saas", "hosting", "soc 2", "breach", "dpi a", "dpa", "pii", "cybersecurity", "supplier"]): # fall back to AI-governance if it's an AI campus piece 
            if "ai" in t or "artificial intelligence" in t: 
                return "Artificial Intelligence Ethics & Governance" # otherwise leave it; caller will keep cosine’s choice 
            return label # Leadership Missteps should only appear if mishandling/contradiction is alleged
        if label == "Leadership Missteps" and not any(k in t for k in ["apolog", "resign", "ethics", "contradict", "downplay", "memo", "statement", "press release", "evasive"]): # if it was a violent incident, route to Violence 
            if any(k in t for k in ["shooting", "shots fired", "gunfire", "killed", "wounded", "lockdown"]): 
                return "Violence or Threats" 
        return label
    def predict_with_fallback(proba_lr, cos_all, prob_cut, margin_cut, tau, trained_labels, all_labels):
        top_idx = proba_lr.argmax(axis=1)
        top_val = proba_lr[np.arange(len(proba_lr)), top_idx]
        tmp = proba_lr.copy()
        tmp[np.arange(len(tmp)), top_idx] = -1
        second = tmp.max(axis=1)
        margin = top_val - second
        use_lr = (top_val >= prob_cut) & (margin >= margin_cut)
    
        cos_all_max = cos_all.max(axis=1)
        cos_all_idx = cos_all.argmax(axis=1)
    
        lr_names  = np.array(trained_labels)[top_idx]
        cos_names = np.array(all_labels)[cos_all_idx]
    
        final_names = np.where(use_lr, lr_names, cos_names)
        # FIX: with raw cosine, compare against tau_raw
        final_names = np.where(~use_lr & (cos_all_max < tau), "No Risk", final_names)
    
        return {
            "final_names": final_names,
            "use_lr": use_lr,
            "lr_top_idx": top_idx,
            "lr_top_prob": top_val,
            "cos_all_idx": cos_all_idx,
            "cos_all_max": cos_all_max,
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
    tau = 0.30
    numeric_factors = list(bundle['numeric_factors'])
    trained_label_txt = list(bundle['trained_label_text'])
    all_labels = json_all_labels
    all_label_txt = list(bundle['all_label_text'])


    df = df.copy()
    df = df.drop_duplicates(subset = 'Title', keep ='last')
    df['Title'] = df['Title'].fillna('').str.strip()

    df['Content'] = df['Content'].fillna('').str.strip()
    df['Text'] = (df['Title'] + '. ' + df['Title'] + '. ' + df['Content']).str.strip()

    df = df.reset_index(drop = True)

    #if 'Predicted_Risks_new' in df.columns:
    #    todo_mask = (df['Predicted_Risks_new'].isna()) | (df['Predicted_Risks_new'].eq('')) | (df['Predicted_Risks_new'].eq('No Risk'))
    #else:
    #    todo_mask = pd.Series(True, index=df.index)
    recent_cut = pd.Timestamp.now(tz='utc') - pd.Timedelta(days=30)
    df['Published_utc'] = pd.to_datetime(df['Published'], errors='coerce', utc = True)
    recent_mask = df['Published_utc'] >= recent_cut
    todo_mask = recent_mask.fillna(False)
    sub = df.loc[todo_mask].copy()
    texts = df.loc[todo_mask, 'Text'].tolist()
    
    print(f"[dbg] total rows: {len(df)}", flush = True)
    print(f"[dbg] parsable Published: {df['Published_utc'].notna().sum()}", flush = True)
    print(f"[dbg] recent (<=30d): {recent_mask.fillna(False).sum()}", flush = True)
    print(f"[dbg] to score (todo_mask): {todo_mask.sum()}", flush = True)
    change = texts
    if not texts:
        return df

    
    with open('Model_training/risks.json', 'r') as f:
        risks_data = json.load(f)

    all_risks = [risk['name'] for group in risks_data['new_risks'] for risks in group.values() for risk in risks]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-mpnet-base-v2', device = device)
    # Encode articles and risks
    #article_embeddings = model.encode(texts, convert_to_numpy = True, normalize_embeddings = True, show_progress_bar=True,  batch_size=256 if device=='cuda' else 32)
    article_embeddings = model.encode(texts, convert_to_numpy = True, normalize_embeddings = True, show_progress_bar=True,  batch_size=256 if device=='cuda' else 32)
    C = article_embeddings
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
    
    # FIX: raw cosine, NOT normalized to sum=1
    cos_all = avg_emb @ lbl_emb_all.T
    lbl_emb_trained = model.encode(trained_label_txt, show_progress_bar = True, normalize_embeddings = True, batch_size = 256)


    out = predict_with_fallback(proba, cos_all, prob_cut, margin_cut, tau, trained_labels, all_labels)
    sub_texts = sub['Text'].tolist()
    routed = [rule_route(txt, lbl) for txt, lbl in zip(sub_texts, np.asarray(out['final_names']).tolist())]
    out['final_names'] = np.array(routed)
    sub['pred_source'] = np.where(out['use_lr'], 'lr', 'cos')
    sub['Predicted_Risks_new'] = out['final_names']
    sub['Pred_LR_label'] = out['lr_top_prob']
    sub['Pred_cos_label_all'] = np.array(all_labels)[out['cos_all_idx']]
    sub['Pred_cos_score_all'] = out['cos_all_max']
    df.loc[sub.index, ['pred_source', 'Predicted_Risks_new', 'Pred_LR_label', 'Pred_cos_label_all', 'Pred_cos_score_all']] = sub[
                        ['pred_source', 'Predicted_Risks_new', 'Pred_LR_label', 'Pred_cos_label_all', 'Pred_cos_score_all']].values



    # Calculate cosine similarity
    #cosine_scores = util.cos_sim(article_embeddings, risk_embeddings)

    #if 'Predicted_Risks_new' not in df.columns:
    #    df['Predicted_Risks_new'] = ''
    # Assign risks based on threshold
    #threshold = 0.35  # you can tune this
    #out = []
    #for row in cosine_scores:
    #    matched = [all_risks[j] for j, s in enumerate(row) if float(s) >= threshold]
    #    out.append('; '.join(matched) if matched else 'No Risk')
    #df.loc[todo_mask, 'Predicted_Risks_new'] = out
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

    try:
        existing = pd.read_csv('BERTopic_before.csv')
        labeled_titles = set(existing['Title']) if 'Title' in existing else set()

    except FileNotFoundError:
        existing = pd.DataFrame()
        existing = pd.DataFrame(columns = ['Title', 'University Label'])
        labeled_titles = set()

    if not existing.empty and 'University Label' in existing.columns:
        all_articles = all_articles.merge(
            existing[['Title', 'University Label']],
            on='Title', how='left',
            suffixes=('', '_prev')
        )
    labeled_titles = set(existing['Title']) if 'Title' in existing else set()

    # Only run labeling on unlabeled articles
    new_articles = all_articles[~all_articles['Title'].isin(labeled_titles)]
    print(f"🔎 Total articles: {len(all_articles)} | Unlabeled: {len(new_articles)}", flush=True)

    results = asyncio.run(university_label_async(new_articles))

    if results:
        labels_df = pd.DataFrame(results)[['Title', 'University Label']]
        all_articles = all_articles.merge(labels_df, on='Title', how='left', suffixes=('', '_new'))
        if 'University Label_prev' not in all_articles.columns:
            all_articles['University Label_prev'] = pd.NA
        all_articles['University Label'].fillna(all_articles['University Label_prev'], inplace=True)
        all_articles.drop(columns=['University Label_prev'], inplace=True)

        if not existing.empty:
            combined = pd.concat([existing, labels_df], ignore_index=True)
        else:
            combined = labels_df
    else:
        combined = existing

    combined.to_csv('BERTopic_before.csv', columns = ['Title', 'University Label'], index = False)

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


def load_midstep_from_release(local_cache_path = 'Model_training/Step1.csv.gz'):
    rel = get_release_by_tag(Github_owner, Github_repo, Release_tag)
    if rel:
        asset = next((a for a in rel.get('assets', []) if a['name']=='Step1.csv.gz'), None)
        if asset:
            r = requests.get(asset['browser_download_url'], timeout = 60)
            if r.ok:
                return pd.read_csv(io.BytesIO(r.content), compression = 'gzip')
    P = Path(local_cache_path)
    if P.exists():
        return pd.read_csv(local_cache_path, compression='gzip')
    return pd.DataFrame()

#Assign topics and probabilities to new_df
#print("✅ Starting transform_text on new data...", flush=True)
#new_df = transform_text(df)
##Fill missing topic/probability rows in the original df
#mask = (df['Topic'].isna()) | (df['Probability'].isna())
#df.loc[mask, ['Topic', 'Probability']] = new_df[['Topic', 'Probability']]
#df[['Topic', 'Probability']] = new_df[['Topic', 'Probability']]
#Save only new, non-duplicate rows
#print("✅ Saving new topics to CSV...", flush=True)
#df_combined = save_new_topics(df, new_df)
#
#Double-check if there are still unmatched (-1) topics and assign a temporary model to assign topics to them
def coerce_pub_utc(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (int, float)):
        if x > 1e12:  # ms
            return pd.to_datetime(x, unit="ms", errors="coerce", utc=True)
        if x > 1e9:   # s
            return pd.to_datetime(x, unit="s", errors="coerce", utc=True)
    # strip common tz words, then parse to UTC
    sx = str(x)
    sx = re.sub(r'\s(EST|EDT|PDT|CDT|MDT|GMT)\b', '', sx, flags=re.I)
    return pd.to_datetime(sx, errors="coerce", utc=True)
#df_combined = load_midstep_from_release()
#print("✅ Running double-check for unmatched topics (-1)...", flush=True)
#cutoff_utc = pd.Timestamp(datetime.utcnow() - timedelta(days = 30), tz = 'utc')
#df_combined['Published'] = df_combined['Published'].apply(coerce_pub_utc)
#atomic_write_csv('Model_training/Step0.csv.gz', df_combined, compress = True)
#upload_asset_to_release(Github_owner, Github_repo, Release_tag, 'Model_training/Step0.csv.gz', GITHUB_TOKEN)

#recent_df = df_combined[df_combined['Published'].notna() & (df_combined['Published'] >= cutoff_utc)].copy()
#temp_model, topic_ids = double_check_articles(recent_df)
#If there are unmatched topics, name them using Gemini
#print("✅ Checking for unmatched topics to name using Gemini...", flush=True)
#if temp_model and topic_ids:
     #topic_name_pairs = get_topic(temp_model, topic_ids)
     #existing_risks_json(topic_name_pairs, temp_model)
##Assign weights to each article
#results_df = load_midstep_from_release()
#results_df = load_university_label(results_df)
#results_df = predict_risks(df_combined)
#results_df['Predicted_Risks'] = results_df.get('Predicted_Risks_new', '')
#print("✅ Applying risk_weights...", flush=True)
#atomic_write_csv('Model_training/Step1.csv.gz', results_df, compress = True)
#upload_asset_to_release(Github_owner, Github_repo, Release_tag, 'Model_training/Step1.csv.gz', GITHUB_TOKEN)
#
#df = load_midstep_from_release()
df = load_midstep_from_release()
#df = pd.read_csv('Model_training/Step1.csv.gz', compression = 'gzip')

results_df = results_df.drop(columns = ['Acceleration_value_x', 'Acceleration_value_y'], errors = 'ignore')
atomic_write_csv('Model_training/initial_label.csv.gz', results_df, compress = True)
upload_asset_to_release(Github_owner, Github_repo, Release_tag, 'Model_training/initial_label.csv.gz', GITHUB_TOKEN)

#
results_df['Predicted_Risks'] = results_df.get('Predicted_Risks_new', results_df.get('Predicted_Risks', ''))
df = risk_weights(results_df)
print("Finished assigning risk weights", flush = True)
df = df.drop(columns = ['University Label_x', 'University Label_y'], errors = 'ignore')
print("Saving BERTopic_results2.csv.gz", flush = True)
atomic_write_csv("Model_training/BERTopic_results2.csv.gz", df, compress=True)
print('Uploading to releases', flush=True)
upload_asset_to_release(Github_owner, Github_repo, Release_tag, 'Model_training/BERTopic_results2.csv.gz', GITHUB_TOKEN)
#Show the articles over time
#
print("Articles over time", flush = True)
#
track_over_time(df)
