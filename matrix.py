from sentence_transformers import CrossEncoder
import json
import os

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
risks = json.loads(os.getenv('RISKS_LIST'))


with open('extracted_RSS.json', 'r') as f:
    articles = json.load(f)

article_embeddings = []
for article in articles:
    article_text = article['Title'] + '. ' + article['Content']
    pairs = [(article_text, risk) for risk in risks]
    scores = model.predict(pairs)

    top_risks = [
        (risk, float(score))
        for risk, score in zip(risks, scores)
        if score >= 0.2
    ]
    top_risks = sorted(top_risks, key = lambda x: x[1], reverse=True)[:3]

    article['Top Risks'] = [{'risk': r, 'score': round(s, 3)} for r, s in top_risks]
    article_embeddings.append(article)

with open('articles_with_risk.json', 'w') as f:
    json.dump(article_embeddings, f, indent=4)
