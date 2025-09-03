import json
import bertopic as bt
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
from datetime import datetime
import ast
from urllib.parse import urlparse

rss_url = "https://github.com/ERSRisk/Tulane-Sentiment-Analysis/releases/download/rss_json/all_RSS.json.gz"

model_url = "https://github.com/ERSRisk/Tulane-Sentiment-Analysis/releases/download/rss_json/BERTopic_model"
model_path = Path("Model_training/BERTopic_model")

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

def download_model_if_exists():
    try:
        print("📦 Checking for model in GitHub release...", flush=True)
        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("✅ Model downloaded successfully.")
            return True
        else:
            print(f"⚠️ Model not found at {model_url}. Status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error while downloading model: {e}")
        return False

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

#topic_blocks = []
#
if model_path.exists() or download_model_if_exists():
    print("Loading existing BERTopic model from disk...")
    model_loaded = True
    GEMINI_API_KEY = os.getenv("PAID_API_KEY")
    client = genai.Client(api_key=GEMINI_API_KEY)
    topic_model = joblib.load(model_path)
else:
    print("Training new BERTopic model from scratch...", flush=True)
    topic_model = bt.BERTopic(language='english', verbose=True)
    topics, probs = topic_model.fit_transform(df['Text'].tolist())

    print(f"✅ BERTopic fit_transform completed. {len(set(topics))} topics found.", flush=True)
    df['Topic'] = topics
    df['Probability'] = probs

    topic_blocks = []
    rep_docs = topic_model.get_representative_docs()
    topics = topic_model.get_topic_info()['Topic'].tolist()
    valid_topics = [t for t in topics if t in rep_docs]

    print(f"🔹 Preparing topic blocks for {len(valid_topics)} valid topics...", flush=True)
    for topic in valid_topics:
        words = topic_model.get_topic(topic)
        docs = topic_model.get_representative_docs()[topic]
        random.shuffle(docs)
        docs = docs[:4]
        keywords = ', '.join([word for word, _ in words])

        def first_n_words(text, n=300):
            words = text.split()
            return text if len(words) <= n else ' '.join(words[:n]) + '...'
        docs_clean = [first_n_words(doc, 300) for doc in docs]
        blocks = f"Topic {topic}: Keywords: {keywords}. Representative Documents: {docs_clean[0]} | {docs_clean[1]}"
        topic_blocks.append((topic, blocks))

    print(f"✅ Prepared {len(topic_blocks)} topic blocks for Gemini.", flush=True)

    # Save model and results
    model_path.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump(topic_model, model_path)
    df.to_csv('BERTopic_results.csv', index=False)
    print("✅ Model saved as .joblib and CSV written.", flush=True)






#GEMINI_API_KEY = os.getenv("PAID_API_KEY")
#client = genai.Client(api_key=GEMINI_API_KEY)
#df['Topic'] = pd.NA
#df['Probability'] = pd.NA

#bert_art = pd.read_csv('BERTopic_results.csv', encoding='utf-8')

#df = pd.concat([df, bert_art], ignore_index=True)
#df = df.drop_duplicates(subset=['Title', 'Content'], keep='last')

#if 'Source' not in df.columns:
#    df['Source'] = ''

# Convert NaN/None to empty string, keep as string dtype
#df['Source'] = df['Source'].astype('string').fillna('')

#def transform_text(texts):
#    texts = texts.copy()
#    print(f"Transforming {len(texts)} articles in batches...")
#    all_topics, all_probs = [], []
#    batch_size = 100  # or smaller
#    texts_list = texts['Text'].tolist()

#    for i in range(0, len(texts_list), batch_size):
#        batch = texts_list[i:i+batch_size]
#        topics, probs = topic_model.transform(batch)
#        all_topics.extend(topics)
#        all_probs.extend(probs)
#        print(f"✅ Transformed batch {i//batch_size + 1}/{(len(texts_list) // batch_size) + 1}")
#    texts['Topic'] = all_topics
#    texts['Probability'] = all_probs
#    return texts

def save_new_topics(existing_df, new_df, path = 'Model_training/BERTopic_results2.csv.gz'):
    if 'Link' in existing_df and 'Link' in new_df:
        unique_new = new_df[~new_df['Link'].isin(existing_df['Link'])]
    else:
        unique_new = new_df

    try:
        on_disk = pd.read_csv(path, compression = 'gzip')
    except FileNotFoundError:
        on_disk = pd.DataFrame()

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
    temp_model = bt.BERTopic(language = 'english', verbose = True)
    temp_model.fit_transform(double_check)
    topic_ids = temp_model.get_topic_info()
    topic_ids = topic_ids[topic_ids['Topic'] != -1]['Topic'].tolist()
    return temp_model, topic_ids

def get_topic(temp_model, topic_ids):
    print("✅ Preparing topic blocks for Gemini naming...", flush=True)
    topic_blocks = []
    for topic in topic_ids:
        words = temp_model.get_topic(topic)
        docs = temp_model.get_representative_docs()[topic]
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
                response = client.models.generate_content(model="gemini-1.5-pro", contents=[prompt])
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
def existing_risks_json(topic_name_pairs, topic_model):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load existing named topics (for matching to *known* topics)
    with open('Model_training/topics_BERT.json', 'r', encoding='utf-8') as f:
        topics = json.load(f)

    existing_topic_names = [t['name'] for t in topics if 'name' in t]
    existing_embeddings = model.encode(existing_topic_names, convert_to_tensor=True)

    matched_topics, unmatched = {}, []

    # Compare each new name against known topics
    for topic_id, new_name in topic_name_pairs:
        new_emb = model.encode([new_name], convert_to_tensor=True)   # (1, d)
        sims = util.cos_sim(new_emb, existing_embeddings)[0]         # (N,)
        best_score = float(sims.max())
        best_index = int(sims.argmax())

        if best_score > 0.85:
            matched_name = existing_topic_names[best_index]
            matched_topics[new_name] = (matched_name, topic_id)
        else:
            unmatched.append((topic_id, new_name))

    # Merge docs/keywords into matched existing topics
    for new_name, (matched_name, topic_id) in matched_topics.items():
        new_docs = topic_model.get_representative_docs().get(topic_id, [])
        new_keywords_pairs = topic_model.get_topic(topic_id) or []
        new_keywords = [w for (w, _) in new_keywords_pairs]

        for topic in topics:
            if topic.get('name') == matched_name:
                # extend docs (dedupe, preserve order)
                seen = set(topic.get('documents', []))
                for d in new_docs:
                    if d not in seen:
                        topic.setdefault('documents', []).append(d)
                        seen.add(d)

                # merge keywords
                existing_keywords = topic.get('keywords', [])
                if isinstance(existing_keywords, str):
                    existing_keywords = [kw.strip() for kw in existing_keywords.split(',') if kw.strip()]
                kw_seen = set(map(str.lower, existing_keywords))
                for kw in new_keywords:
                    if kw.lower() not in kw_seen:
                        existing_keywords.append(kw)
                        kw_seen.add(kw.lower())
                topic['keywords'] = existing_keywords

    with open('Model_training/topics_BERT.json', 'w', encoding='utf-8') as f:
        json.dump(topics, f, indent=4, ensure_ascii=False)

    # ---- Unmatched handling ----
    try:
        with open('Model_training/unmatched_topics.json', 'r', encoding='utf-8') as f:
            existing_unmatched = json.load(f)
            if not isinstance(existing_unmatched, list):
                existing_unmatched = []
    except FileNotFoundError:
        existing_unmatched = []

    # Build a **list** of names aligned with existing_unmatched indices
    unmatched_names = []
    index_map = []  # map from names-list index -> existing_unmatched index
    for i, item in enumerate(existing_unmatched):
        if isinstance(item, dict) and 'name' in item:
            unmatched_names.append(item['name'])
            index_map.append(i)

    if unmatched_names:
        unmatched_embeddings = model.encode(unmatched_names, convert_to_tensor=True)
    else:
        unmatched_embeddings = None

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

        if best_score > 0.85 and best_existing_idx is not None:
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
        else:
            existing_unmatched.append({
                'topic': topic_id,
                'name': name,
                'keywords': new_keywords,
                'documents': new_docs
            })

    with open('Model_training/unmatched_topics.json', 'w', encoding='utf-8') as f:
        json.dump(existing_unmatched, f, indent=4, ensure_ascii=False)

def risk_weights(df):


    # ---------- Load config ----------
    with open('Model_training/risks.json', 'r', encoding='utf-8') as f:
        risks_cfg = json.load(f)

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
            if x > 1e12:  # epoch ms
                return pd.to_datetime(x, unit='ms', errors='coerce', utc=True)
            if x > 1e9:   # epoch s
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
    def _recency_bucket(d):
        if d <= 2: return 5
        if d <= 30: return 4
        if d <= 90: return 3
        if d <= 180: return 2
        if d <= 365: return 1
        return 0
    base['Recency'] = base['Days_Ago'].apply(_recency_bucket)

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

    pr_col = 'Predicted_Risks'
    if pr_col not in base.columns:
        base[pr_col] = ''
    def _parse_tokens(s):
        if s is None or pd.isna(s) or str(s).strip() == '':
            return ""
        return str(s).split(';', 1)[0].strip()
    base['_RiskList'] = base[pr_col].fillna('').astype(str).apply(_parse_tokens)

    #base = base.explode('_RiskList', ignore_index=False).rename(columns={'_RiskList':'Risk_item'})
    #mask = exploded['Risk_item'].notna() & (exploded['Risk_item'].astype(str).str.strip()!='')
    #exploded = exploded[mask].copy()
    #exploded['Risk_norm'] = exploded['Risk_item'].astype(str).str.strip().str.lower()
    base['Risk_item'] = (
    base['Predicted_Risks'].astype(str).str.split(';', 1).str[0].str.strip().str.lower()
    .replace({'': 'no risk'})
    )

    # Published -> datetime (robust coercion)




    if 'Topic' not in base.columns:
        base['Topic'] = -1
    base['Topic'] = base['Topic'].fillna(-1)

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
        import numpy as np
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

        base = base.merge(
            ts[['Risk_item', 'Week', 'Acceleration_value']],
            on=['Risk_item', 'Week'], how='left'
        )
    if 'Acceleration_value' not in base.columns:
        base['Acceleration_value'] = 0


    base['Acceleration_value'] =base['Acceleration_value'].fillna(0).astype(int)




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
    peer_pat = re.compile(r'\b(' + '|'.join([re.escape(p) for p in peers_list]) + r')\b', flags = re.I) if peers_list else None

    moderate_impact = re.compile(r'\b(outage|closure|lawsuit|probation|sanction|breach|evacuation|investigation)\b', re.I)
    substantial_impact = re.compile(r'\b(widespread|catastrophic|shutdown|bankrupt|insolvenc\w*|fatalit\w*|revocation|accreditation\s+revoked)\b', re.I)
    _text_all = (base['Title'].fillna('') + ' ' + base['Content'].fillna('')).astype(str)
    base['_tulane_flag'] = (base.get('Location', 0).astype(int).eq(5)) | _text_all.str.contains(r'\btulane\b', case=False, regex=True)

    def _sev_code(t: str) -> int:
        t = str(t)
        if substantial_impact.search(t):  # you already defined these regexes above
            return 2                      # 2 = substantial
        if moderate_impact.search(t):
            return 1                      # 1 = moderate
        return 0                          # 0 = none/low

    base['_sev_code'] = _text_all.apply(_sev_code).astype(int)

    # 2) Use PUBLISHED TIME so we only consider *previous* Tulane events
    #    If Published is missing, fall back to global history (no time ordering).
    if 'Published' in base.columns and pd.api.types.is_datetime64_any_dtype(base['Published']):
        # Fill NaT with very old date so they don't count as "previous"
        _pub = base['Published'].fillna(pd.Timestamp('1900-01-01'))
        base['_sev_tul'] = np.where(base['_tulane_flag'], base['_sev_code'], 0)
        # Sort by time, then compute cumulative "max severity so far" per risk
        base = base.sort_values(['Risk_item', _pub.name])
        base['_sev_cummax'] = base.groupby('Risk_item')['_sev_tul'].cummax()
        # Shift by one so current row does NOT count itself
        base['_sev_prior'] = base.groupby('Risk_item')['_sev_cummax'].shift(1).fillna(0).astype(int)
    else:
        # Fallback: overall max severity for Tulane mentions per risk (no time ordering)
        _hist = (base.loc[base['_tulane_flag']]
                    .groupby('Risk_item')['_sev_code'].max()
                    .rename('_sev_prior'))
        base = base.merge(_hist, on='Risk_item', how='left')
        base['_sev_prior'] = base['_sev_prior'].fillna(0).astype(int)

    # 3) Map prior severity -> allowed maximum for Frequency_Score
    #    0 -> max 3 (no Tulane history), 1 -> max 4 (moderate), 2 -> max 5 (substantial)
    _allowed_max_map = {0: 3, 1: 4, 2: 5}
    base['_freq_allowed_max'] = base['_sev_prior'].map(_allowed_max_map).astype(int)

    # Apply the cap: you can only show 4/5 if Tulane history justifies it
    base['Frequency_Score'] = np.minimum(base['Frequency_Score'].astype(int),
                                         base['_freq_allowed_max'])

    # (Optional) If you ALSO want to *promote* to 4/5 when history exists even if quantile gave lower:
    # base['Frequency_Score'] = np.maximum(base['Frequency_Score'], base['_freq_allowed_max'])

    # Clean up helpers if you like
    base.drop(columns=['_tulane_flag','_sev_code','_sev_tul','_sev_cummax','_sev_prior','_freq_allowed_max'],
              errors='ignore', inplace=True)

    tmp_ind = base.loc[base['Week'].notna(), ['Week', 'Title', 'Content']].copy()
    tmp_ind['text_all'] = (tmp_ind['Title'] + ' ' + tmp_ind['Content']).fillna('')

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

    tmp_ind['peer'] = tmp_ind['text_all'].apply(find_peer)
    tmp_ind['sev'] = tmp_ind['text_all'].apply(severity)

    agg = (
        tmp_ind.groupby(['Week', 'sev'])['peer'].nunique().unstack(fill_value=0).rename(columns={'moderate': 'peers_mod', 'substantial':'peers_sub'})
    )

    for c in ['peers_mod', 'peers_sub']:
        if c not in agg.columns: agg[c] = 0

    agg = agg.reset_index()

    def peer_industry_score(row):
        if row['peers_sub'] >= 2:
            return 5
        if row['peers_mod'] >=1:
            return 4
        return 0

    agg['Industry_Risk_Peer'] = agg.apply(peer_industry_score, axis =1).astype(int)
    # Map per base row
    base = base.drop(columns = ['Industry_Risk_Peer'], errors = 'ignore')
    base = base.merge(agg[['Week', 'Industry_Risk_Peer']],
                              on=['Week'], how='left')
    base['Industry_Risk_Peer'] = base['Industry_Risk_Peer'].fillna(0).astype(int)
    base['Industry_Risk'] = np.maximum(base['Industry_Risk_Presence'], base['Industry_Risk_Peer']).astype(int)



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

    return base
def predict_risks(df):

    df['Title'] = df['Title'].fillna('').str.strip()

    df['Content'] = df['Content'].fillna('').str.strip()
    df['Text'] = (df['Title'] + '. ' + df['Content']).str.strip()
    df = df.reset_index(drop = True)

    with open('Model_training/risks.json', 'r') as f:
        risks_data = json.load(f)

    all_risks = [risk['name'] for group in risks_data['new_risks'] for risks in group.values() for risk in risks]

    model = SentenceTransformer('all-mpnet-base-v2')
    # Encode articles and risks
    article_embeddings = model.encode(df['Text'].tolist(), show_progress_bar=True, convert_to_tensor=True)
    risk_embeddings = model.encode(all_risks, convert_to_tensor=True)

    # Calculate cosine similarity
    cosine_scores = util.cos_sim(article_embeddings, risk_embeddings)

    if 'Predicted_Risks_new' not in df.columns:
        df['Predicted_Risks_new'] = ''
    # Assign risks based on threshold
    threshold = 0.35  # you can tune this
    out = [''] * len(df)
    for pos in range(len(df)):
        existing = str(df.at[pos, 'Predicted_Risks_new']).strip()
        if existing:
            out[pos] = existing
            continue
        scores = cosine_scores[pos]
        matched = [all_risks[j] for j, s in enumerate(scores) if float(s) >= threshold]
        out[pos] = '; '.join(matched) if matched else 'No Risk'

    df['Predicted_Risks_new'] = out
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
    return client.models.generate_content(model="gemini-1.5-flash", contents=[prompt])

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
            Check each article Title and Content for news regarding *specifically* about higher education, university news, or
            university funding. If the article refers to higher education or university news, 
            return a **compact and valid JSON object**, properly escaped, without explanations:
            {{
                "Title":"same title",
                "Content":"same content",
                "University Label": 1
            }}
            Else, set "University Label" to 0
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
            on = 'Title', how = 'left',
            suffixes = ('', '_prev')
        ).rename(columns = {'University Label': 'University Label_prev'})
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


#Assign topics and probabilities to new_df
#print("✅ Starting transform_text on new data...", flush=True)
#new_df = transform_text(df)
#Fill missing topic/probability rows in the original df
#mask = (df['Topic'].isna()) | (df['Probability'].isna())
#df.loc[mask, ['Topic', 'Probability']] = new_df[['Topic', 'Probability']]
#Save only new, non-duplicate rows
#print("✅ Saving new topics to CSV...", flush=True)
#df_combined = save_new_topics(df, new_df)

#Double-check if there are still unmatched (-1) topics and assign a temporary model to assign topics to them
#print("✅ Running double-check for unmatched topics (-1)...", flush=True)
#temp_model, topic_ids = double_check_articles(df_combined)

#If there are unmatched topics, name them using Gemini
#print("✅ Checking for unmatched topics to name using Gemini...", flush=True)
#if temp_model and topic_ids:
#    topic_name_pairs = get_topic(temp_model, topic_ids)
#    existing_risks_json(topic_name_pairs, temp_model)

#Assign weights to each article
#df = predict_risks(df_combined)
#df['Predicted_Risks'] = df.get('Predicted_Risks_new', '')
#print("✅ Applying risk_weights...", flush=True)
#atomic_write_csv('Model_training/Step1.csv.gz', df, compress = True)
df = pd.read_csv('Model_training/label_initial.csv.gz', compression = 'gzip')
df = predict_risks(df)
df['Predicted_Risks'] = df.get('Predicted_Risks_new', '')
#results_df = load_university_label(df)
df = df.drop(columns = ['Acceleration_value_x', 'Acceleration_value_y'], errors = 'ignore')
df = risk_weights(df)
df = df.drop(columns = ['University Label_x', 'University Label_y'], errors = 'ignore')
atomic_write_csv("Model_training/BERTopic_results2.csv.gz", df, compress=True)
#Show the articles over time
track_over_time(df)
