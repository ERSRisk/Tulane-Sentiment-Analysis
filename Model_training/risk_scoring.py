import pandas as pd
from datetime import datetime
from datetime import timezone
import json
import spacy
from spacy.matcher import PhraseMatcher
from collections import Counter
import asyncio

articles = pd.read_csv('BERTopic_results_test.csv')
with open('Model_training/risks.json', 'r') as f:
    risks = json.load(f)

nlp = spacy.load('en_core_web_sm')
matcher = PhraseMatcher(nlp.vocab, attr='LOWER')

for risk, phrases in risks['HigherEdRisks'].items():
    patterns = [nlp(text) for text in phrases]
    matcher.add(risk, patterns)

all_risks = articles['Predicted_Risks'].explode()
risk_counts = Counter(all_risks.dropna())
risk_freq_df = pd.DataFrame(risk_counts.items(), columns = ['Risk', 'Count'])
risk_freq_df['Frequency_Score'] = pd.qcut(risk_freq_df['Count'], 5, labels=[1, 2, 3, 4, 5])
risk_freq_df['Frequency_Score'] = risk_freq_df['Frequency_Score'].astype(int)
risk_score_map = dict(zip(risk_freq_df['Risk'], risk_freq_df['Frequency_Score']))

accuracy_map = {entry['name']: entry['accuracy'] for entry in risks['sources']}
risk_level_map = {
    "Low":1,
    "Medium Low":2,
    "Medium":3,
    "Medium High":4,
    "High":5
}
risks_map = {entry.get('name','').lower(): risk_level_map.get(str(entry.get('level', '')).lower(), 0)
    for entry in risks['risks']
    if entry.get('name')}

articles = articles.dropna(subset = ['Title'])
articles['Published_raw'] = articles['Published']
articles['Published'] = articles['Published'].str.replace(r'\s(EST|EDT|PDT|CDT|MDT|GMT)', '', regex=True)
articles['Published'] = pd.to_datetime(articles['Published'], format = 'mixed', utc = 'True')
articles['Days_Ago'] = (datetime.now(timezone.utc) - articles['Published']).dt.days

def tag_period(days):
    if days <= 30:
        return 'recent'
    elif days <= 60:
        return 'previous'
    else:
        return 'older'
    
articles['Time_Window'] = articles['Days_Ago'].apply(tag_period)

topic_time_counts = (
    articles.groupby(['Topic', 'Time_Window'])
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

articles['Acceleration'] = (
    topic_time_counts['recent'] - topic_time_counts['previous']
)

async def process_articles(i, article, sem):
    async with sem:
        return await asyncio.to_thread(process_article_sync, i, article)
    
def process_article_sync(i, article):
    result = {
        'Recency': 0,
        'Source_Accuracy': 0,
        'Impact_Score': 0,
        'Acceleration_value': 0,
        'Location': 0,
        'Detected_Risks': '',
        'Industry_Risk': 0,
        'Frequency_Score': 0,
        'Risk_Score': 0
    }

    print(f"Processing article: [{i+1}]")
    
    published = article['Published']
    recency = 0
    try:
        days_ago = (datetime.now(timezone.utc) - published).days
        
        if days_ago <= 30:
            recency = 5
        elif days_ago <= 60:
            recency = 4
        elif days_ago <= 90:
            recency = 3
        elif days_ago <= 180:
            recency = 2
        elif days_ago <= 365:
            recency = 1
        elif days_ago > 365:
            recency = 0
    except Exception as e:
        print(f"Error processing date for article {i}: {e}")
        recency = 0
    result['Recency'] = recency

    source_name = article['Source']
    accuracy = accuracy_map.get(source_name, 0)
    result['Source_Accuracy'] = accuracy

    raw_risks = article['Predicted_Risks']
    if pd.isna(raw_risks):
        predicted_risks  = []
    else:
        predicted_risks = str(raw_risks).lower().split(', ')
    impact_score = max((risks_map.get(risk, 0) for risk in predicted_risks), default=0)
    result['Impact_Score'] = impact_score

    if article['Acceleration'] <= 0:
        acceleration = 0
    elif article['Acceleration'] <= 2:
        acceleration = 1
    elif article['Acceleration'] <= 3:
        acceleration = 2
    elif article['Acceleration'] <= 5:
        acceleration = 3
    elif article['Acceleration'] <= 10:
        acceleration = 4
    else:
        acceleration = 5

    result['Acceleration_value'] = acceleration

    entities = article['Entities']
    if isinstance(entities, list) and any(entity in ['New Orleans', 'Louisiana'] for entity in entities):
        location = 5
    if isinstance(entities, list) and any(entity in ['Baton Rouge', 'Alabama', 'Texas', 'Mississippi'] for entity in entities):
        location = 1
    else:
        location = 0
    result['Location'] = location

    
    doc = nlp(article['Title']+' '+ article['Content'])
    matches = matcher(doc)
    detected = list(set([nlp.vocab.strings[match_id] for match_id, start, end in matches]))
    if detected:
        result['Detected_Risks'] = ', '.join(detected)
        result['Industry_Risk'] = 5
    else:
        result['Detected_Risks'] = ''
        result['Industry_Risk'] = 0

    if not isinstance(article['Predicted_Risks'], list):
        freq= 0
    if not article['Predicted_Risks']:
        freq = 0
    else:
        freq = max(risk_score_map.get(risk, 0) for risk in article['Predicted_Risks'].split(', '))
    result['Frequency_Score'] = freq

    result['Risk_Score'] = (
        ((result['Recency'] * 0.1) + 
        (result['Source_Accuracy'] * 0.15) + 
        (result['Impact_Score'] * 0.25) + 
        (result['Acceleration_value'] *0.08) + 
        (result['Location'] *0.05) + 
        (result['Industry_Risk'] *0.2) + 
        (result['Frequency_Score']*0.07))/5
    )

    return i, result

async def process_articles_async(articles, concurrency =20):
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        process_articles(i, row, sem)
        for i, (_, row) in enumerate(articles.iterrows())
    ]
    results = await asyncio.gather(*tasks)
    return results

results = asyncio.run(process_articles_async(articles))

for i, result in results:
    for key, value in result.items():
        articles.at[i, key] = value
    

articles.to_csv('BERTopic_results_test.csv', index=False)
    
