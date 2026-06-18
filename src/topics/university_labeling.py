import backoff
from google import genai
import requests
import pandas as pd
from src.topics.university_contexts import context_library
import asyncio
from src.utils.gemini import call_gemini
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity


@backoff.on_exception(backoff.expo,
                          (genai.errors.ServerError, requests.exceptions.ConnectionError),
                          max_tries=6,
                          jitter=None,
                          on_backoff=lambda details: print(
                              f"Retrying after error: {details['exception']} (try {details['tries']} after {details['wait']}s)", flush=True))
async def process_article(article, sem, batch_number=None, total_batches=None, article_index=None):
    async with sem:
        try:
            relevant_cats = []
            for rank in [1, 2, 3]:
                cat = article.get(f"top_context_{rank}")
                score = article.get(f"top_score_{rank}", 0)
                if cat and score >= 0.3:
                    relevant_cats.append(cat)
    
            if not relevant_cats:
                print(f"Row {article_index:>3} | 0")
                return {"Title": str(article.get('Title', '')), "University Label": 0}
            if batch_number is not None and total_batches is not None and article_index is not None:
                print(f"📦 Processing Batch {batch_number} of {total_batches} | Article {article_index}", flush=True)
            truncated = article['combined_text'][:3000]
            content = article['Content']
            article_title = article['Title']
            if pd.isna(content) or pd.isna(article_title):
                return None
            categories_block = ''
            for name in relevant_cats:
                description = context_library[name]
                categories_block += f"\n### {name}\n{description.strip()}\n"
            prompt = f"""You are an Enterprise Risk Management analyst for Tulane University, \\
            a private research university in New Orleans, Louisiana.
            
            Read the article below and decide whether it is relevant to Tulane University \\
            based on the risk categories provided. Be strict: only mark relevant=1 if the article \\
            clearly describes a risk or event that could affect Tulane or a close peer institution.
            
            ARTICLE TITLE: {article_title}
            
            ARTICLE TEXT:
            {truncated}
            
            ---
            RISK CATEGORIES TO CONSIDER:
            {categories_block}
            ---
            
            Respond with ONLY a valid JSON object in this exact format — no markdown, no extra text:
            {{
                "University Label": 0,
                "reasoning": "one sentence explaining your decision"
            }}
            
            Labeling rules:
            - Return 1 ONLY if the article reports higher-ed institution news in the United States.
            - Return 1 if the article reports a Louisiana legislative bill REGARDING colleges/universities or higher education.
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
            
            """

            response = await asyncio.to_thread(call_gemini, prompt)
            default = {"Title": str(article_title), "University Label": 0, "reasoning": "parse error"}
            try:
                response_text = getattr(response, "text", "") or ""
                cleaned = response_text.strip()
                if cleaned.startswith("```"):
                    cleaned = cleaned.split("```")[1]
                    if cleaned.lower().startswith("json"):
                        cleaned = cleaned[4:]
                cleaned = cleaned.strip()
                parsed = json.loads(cleaned)
                return {
                    "Title": str(article_title),
                    "University Label": 1 if int(parsed.get("University Label", 0)) == 1 else 0,
                    "reasoning": parsed.get("reasoning", "")
                }
            

                ulabel = parsed.get('University Label')

                if ulabel is None:
                    ulabel = parsed.get('university_label') or rec.get('University_label') or rec.get('university label') or 0

                try:
                    ulabel = int(ulabel)
                    ulabel = 1 if ulabel ==1 else 0
                except Exception:
                    ulabel = 0
                return {'Title': str(title), "Content": str(content), 'University Label': ulabel}
            except Exception as e:
                print(f"  [parse error] {e} | raw: {response_text[:200]}")
                return default
        except Exception as e:
            print(f"🔥 Uncaught error in article {article_index} of batch {batch_number}: {e}", flush=True)
            return None

    # 🚀 Async batch runner
async def university_label_async(articles, batch_size=15, concurrency=10):
    
    sem = asyncio.Semaphore(concurrency)
    tasks = []

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded.")

    total_articles = len(articles)
    total_batches = (total_articles + batch_size - 1) // batch_size
    context_names = list(context_library.keys())
    context_texts = list(context_library.values())
    context_embeddings = model.encode(context_texts, show_progress_bar = True)
    for start in range(0, total_articles, batch_size):
        batch_number = (start // batch_size) + 1
        print(f"🚚 Starting Batch {batch_number} of {total_batches}", flush=True)
        batch = articles.iloc[start:start+batch_size]
        batch['combined_text'] = (
            batch["Title"].fillna("").astype(str) + " " +
            batch["Summary"].fillna("").astype(str) + " " +
            batch["Content"].fillna("").astype(str)
        )
        article_embeddings = model.encode(batch['combined_text'].tolist(), show_progress_bar = True)

        similarity_matrix = cosine_similarity(article_embeddings, context_embeddings)
        top_3_indices = np.argsort(similarity_matrix, axis=1)[:,::-1][:,:3]

        batch = batch.copy()
        batch["top_context_1"] = [context_names[row[0]] for row in top_3_indices]
        batch["top_context_2"] = [context_names[row[1]] for row in top_3_indices]
        batch["top_context_3"] = [context_names[row[2]] for row in top_3_indices]
        
        batch["top_score_1"] = [similarity_matrix[i, row[0]] for i, row in enumerate(top_3_indices)]
        batch["top_score_2"] = [similarity_matrix[i, row[1]] for i, row in enumerate(top_3_indices)]
        batch["top_score_3"] = [similarity_matrix[i, row[2]] for i, row in enumerate(top_3_indices)]
            
        for i, (_, row) in enumerate(batch.iterrows()):
            tasks.append(process_article(row, sem,
                                            batch_number=batch_number,
                                            total_batches=total_batches,
                                            article_index=i+1))

    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

def load_university_label(new_label):
    
    all_articles = new_label.copy()
    all_articles['Source'] = all_articles.get('Source', '').astype(str).fillna('')
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=10)
    if 'Published_utc' in all_articles.columns:
        all_articles['Published_utc'] = pd.to_datetime(all_articles['Published_utc'], errors = 'coerce', utc =  True)
    else:
        all_articles['Published_utc'] = pd.to_datetime(
        all_articles['Published'], errors='coerce', utc=True
        )

    recent = all_articles[all_articles['Published_utc'] >= cutoff]

    try:
        existing = pd.read_csv('pipeline/resources/BERTopic_before.csv')
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
    sorted = new_articles.sort_values(by = 'Published_utc', ascending = False)
    print(sorted[['Title', 'Published_utc']].head())
    new_articles = new_articles[~(new_articles['University Label'].isin([0,1]))]
    print(f"🔎 Total articles: {len(recent)} | Unlabeled: {len(new_articles)}", flush=True)

    results = asyncio.run(university_label_async(new_articles))

    if results:
        labels_df = pd.DataFrame(results)[['Title', 'University Label']]
        labels_df['Title'] = labels_df['Title'].astype(str).str.strip()
        labels_df['University Label'] = pd.to_numeric(
            labels_df['University Label'], errors = 'coerce'
        ).fillna(0).astype(int)

        new_articles['Title'] = new_articles['Title'].astype(str).str.strip()
        missing_titles = set(new_articles['Title']) - set(labels_df['Title'])
        if missing_titles:
            missing_df = pd.DataFrame({
                'Title': list(missing_titles),
                'University Label': [0] * len(missing_titles)
            })
            labels_df = pd.concat([labels_df, missing_df], ignore_index = True)
        all_articles = all_articles.merge(
            labels_df,
            on = 'Title',
            how = 'left',
            suffixes = ('', '_new')
        )

        if 'University Label_new' in all_articles.columns:
            all_articles['University Label'] = all_articles['University Label_new'].combine_first(
                all_articles.get('University Label')
            )
            all_articles = all_articles.drop(columns = ['University Label_new'])

    all_articles['University Label'] = pd.to_numeric(
        all_articles.get('University Label'), errors = 'coerce'
    ).fillna(0).astype(int)

    final_labels = (
        all_articles[['Title','University Label']]
        .dropna(subset = ['Title'])
        .drop_duplicates(subset = ['Title'], keep = 'last')
    )
    combined = pd.concat([existing, final_labels], ignore_index = True)
    combined = combined.drop_duplicates(subset = ['Title'], keep ='last')

    combined.to_csv('pipeline/resources/BERTopic_before.csv',
                    columns=['Title', 'University Label'],
                    index=False)
    all_articles['Source'] = all_articles.get('Source', '').astype(str).fillna('')
    all_articles['University Label'] = pd.to_numeric(
        all_articles['University Label'], errors='coerce'
    ).fillna(0).astype(int)
    return all_articles