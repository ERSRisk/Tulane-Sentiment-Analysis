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
    with open('Model_trianing/topics_BERT.json', 'w') as f:
        json.dump(topic_dict, f, indent=4)

topic_blocks = []

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
    df.to_csv('Model_training/BERTopic_results.csv', index=False)
    print("✅ Model saved as .joblib and CSV written.", flush=True)

    


    

GEMINI_API_KEY = os.getenv("PAID_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
df['Topic'] = pd.NA
df['Probability'] = pd.NA

bert_art = pd.read_csv('Model_training/BERTopic_results.csv', encoding='utf-8')

df = pd.concat([df, bert_art], ignore_index=True)
df = df.drop_duplicates(subset=['Title', 'Content'], keep='last')

def transform_text(texts):
    print(f"Transforming {len(df)} articles in batches...")
    all_topics, all_probs = [], []
    batch_size = 100  # or smaller
    texts_list = df['Text'].tolist()
    
    for i in range(0, len(texts_list), batch_size):
        batch = texts_list[i:i+batch_size]
        topics, probs = topic_model.transform(batch)
        all_topics.extend(topics)
        all_probs.extend(probs)
        print(f"✅ Transformed batch {i//batch_size + 1}/{(len(texts_list) // batch_size) + 1}")
    texts['Topic'] = all_topics
    texts['Probability'] = all_probs
    return texts

def save_new_topics(existing_df, new_df):
    existing_topics = set(existing_df['Link']) if 'Link' in existing_df else set()
    unique_new_topics = new_df[~new_df['Link'].isin(existing_topics)]
    
    if not unique_new_topics.empty:
        combined = pd.concat([existing_df, unique_new_topics])
        combined.to_csv('Model_training/BERTopic_results.csv', index = False)
        return unique_new_topics
    return pd.DataFrame()


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
    with open('Model_training/topics_BERT.json') as f:
        topics = json.load(f)
    existing_topics = [topic['name'] for topic in topics]
    existing_embeddings = model.encode(existing_topics, convert_to_tensor = True)


    matched_topics, unmatched = {}, []

    for topic_id, new_name in topic_name_pairs:
        new_embedding = model.encode([new_name], convert_to_tensor=True)[0]
        similarities = util.cos_sim(new_embedding, existing_embeddings)
        best_score = float(similarities.max())
        best_index = int(similarities.argmax())

        if best_score > 0.85:
            matched_name = existing_topics[best_index]
            matched_topics[new_name] = (matched_name, topic_id)
        else:
            unmatched.append((topic_id, new_name))
    
    for new_name, (matched_name, topic_id) in matched_topics.items():
        new_docs = topic_model.get_representative_docs()[topic_id]
        new_keywords = [word for word, _ in topic_model.get_topic(topic_id)]
        for topic in topics:
            if topic['name'] == matched_name:
                topic['documents'].extend(new_docs)
                topic['documents'] = list(set(topic['documents']))
                existing_keywords = topic.get('keywords', [])
                if isinstance(existing_keywords, str):
                    existing_keywords = [kw.strip() for kw in existing_keywords.split(',')]
                
                topic['keywords'] = list(set(existing_keywords + new_keywords))

    with open('Model_training/topics_BERT.json', 'w', encoding='utf-8') as f:
        json.dump(topics, f, indent=4)

    try:
        with open('Model_training/unmatched_topics.json', 'r') as f:
            existing_unmatched = json.load(f)
    except FileNotFoundError:
        existing_unmatched = []

    existing_unmatched_names = {item['name'] for item in existing_unmatched}
    existing_unmatched_embeddings = model.encode(existing_unmatched_names, convert_to_tensor = True) if existing_unmatched else []
    for topic_id, name in unmatched:
        new_embedding = model.encode([name], convert_to_tensor=True)[0]
        new_docs = topic_model.get_representative_docs()[topic_id]
        new_keywords = [word for word, _ in topic_model.get_topic(topic_id)]

        if existing_unmatched_embeddings:
            similarities = util.cos_sim(new_embedding, existing_unmatched_embeddings)
            best_score = float(similarities.max())
            best_index = int(similarities.argmax())
        else:
            best_score = 0

        if best_score > 0.85:
            matched = existing_unmatched[best_index]
            matched['documents'].extend(new_docs)
            matched['keywords'] = list(set(matched.get('keywords', []) + new_keywords))
        else:
            unmatched_entry={
                'topic': topic_id,
                'name': name,
                'keywords': new_keywords,
                'documents': new_docs
            }
            existing_unmatched.append(unmatched_entry)

    with open('Model_training/unmatched_topics.json', 'w') as f:
        json.dump(existing_unmatched, f, indent=4)

def risk_weights(df):
    with open('Model_training/risks.json', 'r') as f:
        data = json.load(f)
    df['Weights'] = ""
    df['Predicted_Risks'] = df['Predicted_Risks'].fillna('').str.strip().str.lower()
    label_correctio = {
    "declining student enrollment": "declining student enrollment",

    "geopolitical shifts, regional conflicts, and governmental instability": "geopolitical instability",

    "geopolitical shifts, regional conflicts, and governmental instability": "geopolitical instability",

    "failure to comply with or change in legal or regulatory requirements": "regulatory compliance failure"}

    for k, v in label_correctio.items():
        df['Predicted_Risks'] = df['Predicted_Risks'].str.replace(v, k, regex=False)
    df['Predicted_Risks'] = df['Predicted_Risks'].astype(str).str.strip().str.split(';')
    df = df.explode('Predicted_Risks')
    df['Predicted_Risks'] = df['Predicted_Risks'].str.strip()
    df = df[df['Predicted_Risks'] != '']
    risk_list = data['risks']
    sources = data['sources']
    weights = []
    for i, row in df.iterrows():
        weight = 0
        for risk in risk_list:
            risk_name = risk['name'].lower()
            level = risk['level']
            likelihood = risk['likelihood']
            impact = risk['impact']
            velocity = risk['velocity']
            if risk_name in row['Predicted_Risks']:
                weight += ((likelihood/5) + (impact/5) + (velocity/5))/3
        for source in sources:
            source_name = source.get('name', '')
            source_accuracy = source.get('accuracy', '')
            source_bias = source.get('bias', '')
            if source_name in row['Source']:
                weight *= 0.85 + 0.15*(source_accuracy/5)
        weights.append(weight *5  if weight >0 else 0)
    df['Weights'] = weights
    return df

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
    threshold = 0.55  # you can tune this
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
def track_over_time(df):
    df['Published'] = pd.to_datetime(df['Published'], errors = 'coerce')
    df = df.dropna(subset=['Published'])

    df['week'] = df['Published'].dt.to_period('W').apply(lambda r: r.start_time)
    topic_name_map = {topic['topic']: topic['name'] for topic in json.load(open('Model_training/topics_BERT.json'))}
    df['Topic_Name'] = df['Topic'].map(topic_name_map)
    topic_trend = df.groupby(['week', 'Topic_Name']).size().reset_index(name='article_count')
    topic_trend.to_csv('Model_training/topic_trend.csv', index = False)


def call_gemini(prompt):
    GEMINI_API_KEY = os.getenv("PAID_API_KEY")
    client = genai.Client(api_key=GEMINI_API_KEY)
    return client.models.generate_content(model="gemini-1.5-flash", contents=[prompt])

# 🧠 Async article processor
@backoff.on_exception(backoff.expo,
                      (genai.errors.ServerError, requests.exceptions.ConnectionError),
                      max_tries=6,
                      jitter=None,
                      on_backoff=lambda details: print(
                          f"Retrying after error: {details['exception']} (try {details['tries']} after {details['wait']}s)", flush=True)
)
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
            Content: {" ".join(str(content).split()[:200])}
            Check each article Title and Content for news regarding higher education, university news, or
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
                json_str = re.search(r"```json\s*(\{.*\})\s*```", response_text, re.DOTALL)
                raw = json_str.group(1) if json_str else response_text

                try:
                    return json.loads(raw)
                except json.JSONDecodeError as e1:
                    try:
                        return ast.literal_eval(raw)
                    except Exception as e2:
                        print(f"⚠️ JSON decode fallback error: {e1} | Eval error: {e2}", flush=True)
                        return None
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
    all_articles = pd.read_csv('Model_training/BERTopic_results.csv')
    try:
        existing = pd.read_csv('BERTopic_before.csv')
        labeled_titles = set(existing['Topic']) if 'Topic' in existing else set()
    except FileNotFoundError:
        existing = pd.DataFrame()
        labeled_titles = set()

    new_articles = all_articles[~all_articles['Title'].isin(labeled_titles)]
    print(f"🔎 Total articles: {len(all_articles)} | Unlabeled: {len(new_articles)}", flush=True)
    
    results = asyncio.run(university_label_async(new_articles))
    new_df = pd.DataFrame(results)
    if not existing.empty:
        combined = pd.concat([existing, new_df], ignore_index = True)
    else:
        combined = new_df
    return combined

    
#Assign topics and probabilities to new_df
print("✅ Starting transform_text on new data...", flush=True)
new_df = transform_text(df)
#Fill missing topic/probability rows in the original df
mask = (df['Topic'].isna()) | (df['Probability'].isna())
df.loc[mask, ['Topic', 'Probability']] = new_df[['Topic', 'Probability']]
#Save only new, non-duplicate rows
print("✅ Saving new topics to CSV...", flush=True)
save_new_topics(df, new_df)

#Double-check if there are still unmatched (-1) topics and assign a temporary model to assign topics to them
print("✅ Running double-check for unmatched topics (-1)...", flush=True)
new_articles, topic_ids = double_check_articles(new_df)

#If there are unmatched topics, name them using Gemini
print("✅ Checking for unmatched topics to name using Gemini...", flush=True)
if new_articles:
    topic_name_pairs = get_topic(new_articles, topic_ids)
    existing_risks_json(topic_name_pairs, new_articles)

#Assign weights to each article
df = predict_risks(df)
print("✅ Applying risk_weights...", flush=True)
df = risk_weights(df)
results_df = load_university_label(df)
results_df.to_csv('BERTopic_before.csv', index=False)
#Show the articles over time
track_over_time(df)
