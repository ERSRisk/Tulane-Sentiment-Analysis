import json
import re
from google.genai.errors import APIError
import time
import random
import requests
import tempfile
import os
import gzip
from sentence_transformers import SentenceTransformer, util
import hashlib


from src.storage.gcs import (
    upload_file
)

from src.utils.gemini import get_gemini_client



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
    with open('pipeline/resources/topics_BERT.json', 'w') as f:
        json.dump(topic_dict, f, indent=4)

def get_topic(temp_model, topic_ids):
    client = get_gemini_client()
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
                response = client.models.generate_content(model="gemini-2.5-flash", contents=[prompt])
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

def label_model_topics(topic_model, path = 'pipeline/resources/topics_BERT.json'):
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
        return upload_file(path, f'latest/{path}')

def existing_risks_json(topic_name_pairs, topic_model):
    unmatched = list(topic_name_pairs)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Load existing named topics (for matching to *known* topics)
    with open('pipeline/resources/topics_BERT.json', 'r', encoding='utf-8') as f:
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

