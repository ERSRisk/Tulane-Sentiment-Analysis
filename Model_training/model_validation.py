from sentence_transformers import SentenceTransformer
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("E:/TUL SP 25/Risk Project/ERM Practicum/my-trained-model")

risks_data = os.getenv("RISKS_LIST")

for risk in risks_data['risks']:
  risk_name = risk['name']
  risk_weight = {'likelihood':risk['likelihood'], 'impact':risk['impact'], 'velocity':risk['velocity'], 'level':risk['level']}

