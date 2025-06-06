from sentence_transformers import SentenceTransformer, util
import json
import os

model = SentenceTransformer('all-mpnet-base-v2')
risks = json.loads(os.getenv('RISKS_LIST'))

with open('extracted_RSS.json', 'r') as f:
    articles = json.load(f)

risk_embeddings = model.encode(risks, convert_to_tensor=True)

results = []
for article in articles:
    article_text = article['Title'] + '. ' + article['Content']
    article_embedding = model.encode(article_text, convert_to_tensor=True)

    cosine_scores = util.cos_sim(article_embedding, risk_embeddings)[0]
    top_indices = cosine_scores.argsort(descending=True)[:3]

    top_risks = [
        {'risk': risks[i], 'score': round(float(cosine_scores[i]), 3)}
        for i in top_indices if float(cosine_scores[i]) > 0.31
    ]

    article['Top Risks'] = top_risks
    results.append(article)

with open('articles_with_risk.json', 'w') as f:
    json.dump(results, f, indent=4)

