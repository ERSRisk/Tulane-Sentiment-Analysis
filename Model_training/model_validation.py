from sentence_transformers import SentenceTransformer, util
import json
import os

model = SentenceTransformer("E:/TUL SP 25/Risk Project/ERM Practicum/my-trained-model")

risks_data = os.getenv("RISKS_LIST")

risk_name = [risk['name'] for risk in risks_data['risks']]
risk_weight = [{'likelihood':risk['likelihood'], 'impact':risk['impact'], 'velocity':risk['velocity'], 'level':risk['level']} for risk in risks_data['risks']]

risk_embedding = model.encode(risk_name, convert_to_tensor = False)

articles_with_risk = []
for article in articles:
  article_text = article['Title'] + '. ' article['Content']
  article_embedding = model.encoder(article_text, convert_to_tensor = False).reshape(1,-1)

  similarities = util.cos_sim(article_embedding, risk_embedding)[0]

  top_indices = similarities.argsort()[-3:][::-1]
  top_risks = []

  for idx in top_indices:
    risk_name = risk_name[idx]
    score = similarities[idx]
    weights = risk_weight[risk_name]

    final_score = score + weights['likelihood'] +weights['impact']+weights['velocity']

    top_risks.append({
      'risk':risk_name,
      'similarity':round(float(score), 4),
      'weighted_score':final_score,
      'velocity':weights['velocity'],
      'impact':weights['impact']})

article['Top Risks'] = top_risks

articles_with_risk.append(article)

pd.to_csv('articles_with_risk)
