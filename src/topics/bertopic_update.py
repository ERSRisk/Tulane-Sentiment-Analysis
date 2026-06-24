import json
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np
import pandas as pd
import os
import re
import requests
from pathlib import Path
import gzip
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gc


from src.risk.scoring import (
    risk_weights
)

from src.risk.prediction import (
    predict_risks
)

from src.storage.gcs import (
    upload_file,
    download_file,
    blob_exists,
)
from src.storage.github_releases import (
    ensure_release,
    get_release_by_tag,
    upload_asset,

)

from src.topics.topic_assets import (
    load_dir_model,
    upload_dir_model_zip,
    Github_owner,
    Github_repo,
    Release_tag, 
    Asset_name, 
    GITHUB_TOKEN,
)


from src.utils.diagnostics import mem, debug_date

from src.topics.university_labeling import load_university_label

from src.utils.io import atomic_write_csv

from src.topics.topic_naming import get_topic, existing_risks_json

DIR_PATH = Path("Model_training/bertopic_dir")


print("Context library loaded.")


BUCKET_NAME = "tulane-risk-data"
mem("start")

rss_url = "https://github.com/ERSRisk/Tulane-Sentiment-Analysis/releases/download/rss_json/all_RSS.json.gz"
    
def coerce_pub_utc(x):
    if pd.isna(x):
        return pd.NaT
    if isinstance(x, (int, float)):
        if x > 1e12:  
            return pd.to_datetime(x, unit="ms", errors="coerce", utc=True)
        if x > 1e9: 
            return pd.to_datetime(x, unit="s", errors="coerce", utc=True)
    sx = str(x)
    sx = re.sub(r'\s(EST|EDT|PDT|CDT|MDT|GMT)\b', '', sx, flags=re.I)
    return pd.to_datetime(sx, errors="coerce", utc=True)

def load_rss_dataframe():
    print(f"📥 Downloading all_RSS.json from release link...", flush=True)
    response = requests.get(rss_url, stream = True, timeout = 200)
    response.raise_for_status()

    Path('pipeline/resources').mkdir(parents = True, exist_ok = True)
    rss_path = "pipeline/resources/all_RSS.json.gz"

    with open(rss_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size = 1024 * 1024):
            if chunk:
                f.write(chunk)
    
    with gzip.open(rss_path, 'rt', encoding = 'utf-8') as f:
        articles = json.load(f)
    # Now load it
    df = pd.DataFrame(articles)
    
    mem("after RSS dataframe")

    df['Source'] = df.get('Source', '').astype(str).fillna('')


    debug_date(df, "A_raw_rss_df")
    
    Path("Online_Extraction").mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copyfile(rss_path, 'Online_Extraction/all_RSS.json.gz')
    df = df[~(df['Source']=="Economist")]
    df['Text'] = df['Title'] + '. ' + df['Content']
    debug_date(df, "B_after_source_filtering")
    return df
    
def load_or_train_model(df):
    topic_model = load_dir_model()
    
        
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
    return topic_model

def load_existing_topics_checkpoint(df):
    dfs = []
    
    if blob_exists('latest/BERTopic_results2.csv.gz'):
        df2 = download_file('latest/BERTopic_results2.csv.gz', 'pipeline/resources/BERTopic_results2.csv.gz')
        mem("after loading existing_df")
        print("It exists", flush = True)
        dfs.append(df2)
    else:
        print("It did not get extracted", flush = True)

    if blob_exists('latest/BERTopic_results3.csv.gz'):
        df3 = download_file('latest/BERTopic_results3.csv.gz', 'pipeline/resources/BERTopic_results3.csv.gz')
        mem("after loading existing_df")
        dfs.append(df3)

    if dfs:
        existing_df = pd.concat(dfs, ignore_index = True)

        existing_df['Link'] = existing_df['Link'].astype(str).str.strip()
        existing_df = existing_df.drop_duplicates(subset = ['Link'], keep = 'last')

        print(f"Combined checkpoint rows: {len(existing_df)}", flush=True)
    else:
        existing_df = pd.DataFrame()
    
    if existing_df is None or existing_df.empty:
        existing_df = pd.DataFrame()
    if not existing_df.empty and 'Link' in existing_df.columns:
        existing_df['Link'] = existing_df['Link'].astype(str).str.strip()
        df['Link'] = df['Link'].astype(str).str.strip()
        processed_links = set(existing_df['Link'])
        df_to_transform = df[~df['Link'].isin(processed_links)].copy()
        print(f"Dataframe to transform is removing already preprocessed articles.", flush = True)
    else:
        df_to_transform = df.copy()
        print(f"Dataframe to transform has no articles to remove.", flush = True)

    return existing_df, df_to_transform

def transform_text(texts, topic_model):
    texts = texts.copy()
    
    print(f"Transforming {len(texts)} articles in batches...", flush=True)

    batch_size = 100
    texts_list = texts["Text"].tolist()
    n = len(texts_list)

    # Pre-allocate lightweight arrays (NOT lists of big objects)
    topics_out = np.full(n, -1, dtype=np.int32)
    prob_out   = np.full(n, np.nan, dtype=np.float32)

    # ---- 1) Transform in batches, but keep ONLY the assigned prob ----
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = texts_list[start:end]

        topics, probs = topic_model.transform(batch)

        # If probs missing, approximate per-batch
        if probs is None:
            print("[info] transform() returned no probabilities; using approximate_distribution()", flush=True)
            probs = topic_model.approximate_distribution(batch)

        topics = np.asarray(topics, dtype=np.int32)
        topics_out[start:end] = topics

        # probs is usually (batch, n_topics). Extract only prob of assigned topic.
        # For outliers (-1), keep NaN.
        if probs is not None:
            probs = np.asarray(probs)
            row_idx = np.arange(len(batch))
            valid = topics >= 0
            # Guard against weird shapes
            if probs.ndim == 2 and probs.shape[0] == len(batch):
                prob_out[start:end][valid] = probs[row_idx[valid], topics[valid]].astype(np.float32)

        print(f"✅ Transformed batch {start//batch_size + 1}/{(n + batch_size - 1)//batch_size}", flush=True)

        # free batch objects
        del batch, topics, probs
        gc.collect()

    # ---- 2) OPTIONAL: Outlier reduction is expensive. If you must, do it on a CAP. ----
    # If reduce_outliers is causing trouble, either skip it on GitHub or cap it.
    # Example cap:
    # if (topics_out == -1).sum() > 0:
    #     cap = 3000
    #     idx = np.where(topics_out == -1)[0][:cap]
    #     tmp_topics = topic_model.reduce_outliers([texts_list[i] for i in idx],
    #                                             topics_out[idx].tolist(),
    #                                             strategy="embeddings",
    #                                             threshold=0.4)
    #     topics_out[idx] = np.asarray(tmp_topics, dtype=np.int32)
    #     del tmp_topics
    #     gc.collect()

    # ---- 3) Streamlit cosine assignment, batched ----
    def get_embedder(topic_model):
        if hasattr(topic_model, "embedding_model_") and topic_model.embedding_model_ is not None:
            return topic_model.embedding_model_
        if hasattr(topic_model, "embedding_model") and topic_model.embedding_model is not None:
            return topic_model.embedding_model
        raise RuntimeError("Could not locate BERTopic's embedding model.")

    def embedding_dim(embedder):
        if hasattr(embedder, "get_sentence_embedding_dimension"):
            return embedder.get_sentence_embedding_dimension()
        if hasattr(embedder, "embedding_model") and hasattr(embedder.embedding_model, "get_sentence_embedding_dimension"):
            return embedder.embedding_model.get_sentence_embedding_dimension()
        v = np.asarray(embedder.encode(["x"], convert_to_numpy=True))
        return int(v.shape[-1])

    def encode(texts_, embedder):
        if isinstance(texts_, str):
            texts_ = [texts_]
        texts_ = [t.strip() if isinstance(t, str) else "" for t in (texts_ or [])]
        if len(texts_) == 0:
            d = embedding_dim(embedder)
            return np.zeros((0, d), dtype=np.float32)

        if hasattr(embedder, "encode"):
            vecs = embedder.encode(
                texts_, convert_to_numpy=True, normalize_embeddings=False,
                batch_size=64, show_progress_bar=False
            )
        elif hasattr(embedder, "embed"):
            vecs = np.asarray(embedder.embed(texts_))
        else:
            raise RuntimeError("Embedder has neither .encode nor .embed")

        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms

    # Load Streamlit topic list
    try:
        with open("pipeline/resources/topics_BERT.json", "r", encoding="utf-8") as f:
            topics_json = json.load(f).get("topics", [])
    except Exception as e:
        topics_json = []
        print(f"[warn] Could not read pipeline/resources/topics_BERT.json: {e}", flush=True)

    remaining_idx = np.where(topics_out == -1)[0]
    if len(remaining_idx) > 0 and len(topics_json) > 0:
        embedder = get_embedder(topic_model)

        rep = topic_model.get_representative_docs()
        rep_map = {int(k): v for k, v in rep.items() if v} if isinstance(rep, dict) else {}

        centroids = []
        streamlit_topic_ids = []

        for t in topics_json:
            if t.get("source") != "Streamlit":
                continue
            tid = int(t["topic"])
            reps = rep_map.get(tid) or [f"{t.get('name','')} ; {t.get('keywords','')}"]
            E_rep = encode(reps, embedder)
            if E_rep.shape[0] == 0:
                continue
            c = E_rep.mean(axis=0)
            c = c / (np.linalg.norm(c) + 1e-12)
            centroids.append(c.astype(np.float32))
            streamlit_topic_ids.append(tid)

        if centroids:
            C = np.stack(centroids, axis=0).astype(np.float32)

            # Batch the remaining embeddings
            st_cos = np.full(n, np.nan, dtype=np.float32)
            cos_batch = 256  # tune down if needed

            for start in range(0, len(remaining_idx), cos_batch):
                idx_batch = remaining_idx[start:start+cos_batch]
                texts_batch = [texts_list[i] for i in idx_batch]
                E = encode(texts_batch, embedder).astype(np.float32)
                if E.shape[0] == 0:
                    continue

                sims = E @ C.T
                j_best = np.argmax(sims, axis=1)
                s_best = sims[np.arange(sims.shape[0]), j_best]

                for row_pos, doc_i in enumerate(idx_batch):
                    if s_best[row_pos] >= 0.40:
                        topics_out[doc_i] = int(streamlit_topic_ids[j_best[row_pos]])
                        st_cos[doc_i] = float(s_best[row_pos])

                del texts_batch, E, sims, j_best, s_best
                gc.collect()

            texts["StreamlitCosine"] = st_cos
        else:
            print("[info] No Streamlit topics with usable centroids found; skipping cosine assignment.", flush=True)

    # ---- 4) Final assembly ----
    texts["Topic"] = topics_out
    texts["Probability"] = prob_out

    how = []
    stcos = texts.get("StreamlitCosine", pd.Series(np.nan, index=texts.index)).to_numpy()
    for t, p, sim in zip(texts["Topic"].to_numpy(), texts["Probability"].to_numpy(), stcos):
        if t == -1:
            how.append("Unassigned")
        elif not np.isnan(p):
            how.append("bertopic")
        elif not np.isnan(sim):
            how.append("streamlit-cosine")
        else:
            how.append("other")
    texts["Assigned_how"] = how

    return texts

def save_new_topics(existing_df, new_df, path="pipeline/resources/BERTopic_results3.csv.gz"):
    existing_df = existing_df.drop_duplicates(subset = ['Link'], keep = 'last')
    if existing_df is not None and not existing_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df.copy()
    print("It worked", flush = True)
    combined.to_csv(path, index=False, compression="gzip")
    return combined

def assign_topics_to_new_articles(df, topic_model):
    existing_df, df_to_transform = load_existing_topics_checkpoint(df)
    debug_date(df_to_transform, "C_df_to_transform")
    print("✅ Starting transform_text on new data...", flush=True)
    topic_model.calculate_probabilities = False
    new_df = transform_text(df_to_transform, topic_model)
    debug_date(new_df, "D_new_df_after_transform")
    del df_to_transform
    gc.collect()
    mem("after transform text")
    new_links = set(new_df['Link'])
    #Fill missing topic/probability rows in the original df
    for c in ['Topic', 'Probability']:
        if c not in df.columns:
            df[c] = np.nan
      
    update_cols = new_df[['Link', 'Topic', 'Probability']].dropna(subset=['Link'])
    df = df.merge(update_cols, on='Link', how='left', suffixes=('', '_new'))
    df['Topic'] = df['Topic_new'].combine_first(df['Topic'])
    df['Probability'] = df['Probability_new'].combine_first(df['Probability'])
    df.drop(columns=['Topic_new','Probability_new'], inplace=True)
    #Save only new, non-duplicate rows
    print("✅ Saving new topics to CSV...", flush=True)
    df_combined = save_new_topics(existing_df, new_df)
    debug_date(df_combined, "E_df_combined_after_save_new_topics")
    del new_df
    gc.collect()
    return df_combined, new_links, existing_df

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

def double_check_and_name_unmatched_topics(df_combined):
    df_combined['Probability'] = pd.to_numeric(df_combined['Probability'], errors = 'coerce')
    #Double-check if there are still unmatched (-1) topics and assign a temporary model to assign topics to them
    
    print("✅ Running double-check for unmatched topics (-1)...", flush=True)
    cutoff_utc = pd.Timestamp(datetime.utcnow() - timedelta(days = 15), tz = 'utc')
    df_combined['Published_utc'] = df_combined['Published'].apply(coerce_pub_utc)
    debug_date(df_combined, "G_df_combined_after_creating_published_utc")
    print(f"Length of dataset: {len(df_combined)}", flush = True)
    print(f"Length of recalculated topic names: {len(df_combined[df_combined['Probability'] < 0.15])}", flush = True)
    low_conf_mask = df_combined['Probability'] < 0.15
    df_combined.loc[low_conf_mask, 'Topic'] = -1
    atomic_write_csv('pipeline/resources/Step0.csv.gz', df_combined, compress = True)
    upload_file('pipeline/resources/Step0.csv.gz', 'latest/Step0.csv.gz')
    #df_combined = load_midstep_from_release()
    recent_df = df_combined[df_combined['Published_utc'].notna() & (df_combined['Published_utc'] >= cutoff_utc)].copy()
    debug_date(recent_df, "G_recent_df_for_double_check")
    temp_model, topic_ids = double_check_articles(recent_df)
    mem("after double_check_articles")
    #If there are unmatched topics, name them using Gemini
    print("✅ Checking for unmatched topics to name using Gemini...", flush=True)
    if temp_model and topic_ids:
        topic_name_pairs = get_topic(temp_model, topic_ids)
        existing_risks_json(topic_name_pairs, temp_model)
    return df_combined

def label_and_predict_risks(df_combined):
    print("Completed save_new_topics", flush = True)
    print("Merged both dfs", flush = True)
    
    df_combined = load_university_label(df_combined)
    debug_date(df_combined, "H_after_load_university_label")
    mem("after load_university_label")
    atomic_write_csv('pipeline/resources/initial_label.csv.gz', df_combined, compress = True)
    upload_file('pipeline/resources/initial_label.csv.gz', 'latest/initial_label.csv.gz')
    #df_combined = load_midstep_from_release()
    results_df = predict_risks(df_combined)
    mask_he = pd.to_numeric(results_df['University Label'], errors = 'coerce').fillna(0).astype(int) == 1
    results_df.loc[mask_he & results_df['Predicted_Risks_new'].isna(), 'Predicted_Risks_new'] = 'No Risk'
    results_df.loc[mask_he & results_df['Predicted_Risks_new'].astype(str).str.strip().eq(''), 'Predicted_Risks_new'] = 'No Risk'
    debug_date(results_df, "I_after_predict_risks")
    mem("after predict_risks")
    del df_combined
    gc.collect()
    results_df['Predicted_Risks'] = results_df.get('Predicted_Risks_new', '')
    results_df = results_df.drop(columns = ['Acceleration_value_x', 'Acceleration_value_y'], errors='ignore')
    print("✅ Applying risk_weights...", flush=True)
    atomic_write_csv('pipeline/resources/Step1.csv.gz', results_df, compress = True)
    upload_file('pipeline/resources/Step1.csv.gz', 'latest/Step1.csv.gz')
    results_df['Predicted_Risks'] = results_df.get('Predicted_Risks_new', results_df.get('Predicted_Risks', ''))
    return results_df

def score_risks_from_step1():
    risk_usecols = [
    'Title', 'Content', 'Source', 'Published', 'Link', 'Entities',
    'Predicted_Risks_new', 'Predicted_Risks', 'Topic', 'Probability',
    'University Label', 'Location',
    'pred_source',
    'Pred_LR_label',
    'Pred_cos_label_all',
    'Pred_cos_score_all'
    ]
    
    risk_df = pd.read_csv('pipeline/resources/Step1.csv.gz', compression = 'gzip', usecols = lambda c: c in risk_usecols, low_memory = False)
    debug_date(risk_df, "J_risk_df_reloaded")
    mem("before risk_weights")
    df = risk_weights(risk_df)
    debug_date(df, "K_after_risk_weights")
    del risk_df
    gc.collect()
    mem("after risk_weights")
    print("Finished assigning risk weights", flush = True)
    df = df.drop(columns = ['University Label_x', 'University Label_y'], errors = 'ignore')
    
    return df

def save_final_topic_outputs(df, new_links, existing_df):
    recent_cut = pd.Timestamp.now(tz='utc') - pd.Timedelta(days=30)

    df['Published_utc'] = pd.to_datetime(
        df['Published'],
        errors='coerce',
        utc=True
    )
    
    rerun_links = set(
        df.loc[
            df['Published_utc'].notna() &
            (df['Published_utc'] >= recent_cut) &
            (pd.to_numeric(df['University Label'], errors='coerce').fillna(0).astype(int) == 1),
            'Link'
        ].astype(str).str.strip()
    )
    
    new_links = set(str(x).strip() for x in new_links)
    
    links_to_save = new_links | rerun_links
    
    df_new_final = df[df['Link'].astype(str).str.strip().isin(links_to_save)].copy()
    debug_date(df_new_final, "L_df_new_final_only_new_links")
    del df
    gc.collect()
    existing_new_version = pd.DataFrame()
    if blob_exists('latest/BERTopic_results3.csv.gz'):
        existing_new_version = download_file(
            'latest/BERTopic_results3.csv.gz',
            'pipeline/resources/BERTopic_results3.csv.gz'
        )
        
    debug_date(existing_new_version, "M_existing_new_version")
    df_new_version = pd.concat([existing_new_version, df_new_final], ignore_index=True)
    df_new_version['Published_utc'] = pd.to_datetime(df_new_version['Published'], errors = 'coerce', utc=True)
    debug_date(df_new_version, "N_df_new_version")
    
    # Make sure newest version wins on duplicate links
    if 'Link' in df_new_version.columns:
        df_new_version = df_new_version.drop_duplicates(subset=['Link'], keep='last')
    
    del existing_new_version
    gc.collect()
    mem("after final concat")
    
    print("Saving BERTopic_results3.csv.gz", flush=True)
    atomic_write_csv("pipeline/resources/BERTopic_results3.csv.gz", df_new_version, compress=True)
    upload_file("pipeline/resources/BERTopic_results3.csv.gz", 'latest/BERTopic_results3.csv.gz')
    
    # NEW: save the canonical fresh full dataset for downstream script 2
    print("Saving latest article base for downstream enrichment", flush=True)
    df_latest_full = pd.concat([existing_df, df_new_final], ignore_index = True)
    df_latest_full['Link'] = df_latest_full['Link'].astype(str).str.strip()
    df_latest_full = df_latest_full.drop_duplicates(subset=['Link'], keep='last')
    del df_new_final
    gc.collect()
    df_latest_full['Published_utc'] = pd.to_datetime(
        df_latest_full['Published'],
        errors='coerce',
        utc=True
    )
    
    debug_date(df_latest_full, "P_df_latest_full")
    
    atomic_write_csv("pipeline/resources/BERTopic_latest_full.csv.gz", df_latest_full, compress=True)
    upload_file(
        "pipeline/resources/BERTopic_latest_full.csv.gz",
        "latest/BERTopic_latest_full.csv.gz"
    )
    
    print("Saving dataset for Streamlit", flush=True)
    debug_date(df_latest_full, "O_before_streamlit")
    df_streamlit = df_latest_full[df_latest_full['University Label'] == 1].copy()
    
    if 'Link' in df_streamlit.columns:
        df_streamlit = df_streamlit.drop_duplicates(subset=['Link'], keep='last')

    debug_date(df_streamlit, "Q_df_streamlit_after_university_label")
    atomic_write_csv("pipeline/resources/BERTopic_Streamlit.csv.gz", df_streamlit, compress=True)
    upload_file('pipeline/resources/BERTopic_Streamlit.csv.gz', 'latest/BERTopic_Streamlit.csv.gz')
    
    del df_streamlit, df_new_version
    gc.collect()


def run_bertopic_update():
    
    df = load_rss_dataframe()

    topic_model = load_or_train_model(df)
    df_combined, new_links, existing_df = assign_topics_to_new_articles(df, topic_model)
    df_combined = double_check_and_name_unmatched_topics(df_combined)
    results_df = label_and_predict_risks(df_combined)

    df = score_risks_from_step1()

    del results_df
    gc.collect()

    save_final_topic_outputs(df, new_links, existing_df)

    
    


def main():
    run_bertopic_update()

if __name__ == "__main__":
    main()

