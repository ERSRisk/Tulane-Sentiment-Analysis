from sentence_transformers import SentenceTransformer
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("E:/TUL SP 25/Risk Project/ERM Practicum/my-trained-model")

risks_data = os.getenv("RISKS_LIST")

risk_name = [risk['name'] for risk in risks_data['risks']]
risk_weight = [{'likelihood':risk['likelihood'], 'impact':risk['impact'], 'velocity':risk['velocity'], 'level':risk['level']} for risk in risks_data['risks']]

risk_embedding = model.encode(risk_name, convert_to_tensor = False)

for article in articles:
  article_text = article['Title'] + '. ' article['Content']
  article_embedding = model.encoder(article_text, convert_to_tensor = False).reshape(1,-1)

  similarities = cosine_similarity(
