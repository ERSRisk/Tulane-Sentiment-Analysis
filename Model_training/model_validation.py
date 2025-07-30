from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import json
import joblib

df = pd.read_csv('BERTopic_results.csv')


df['Title'] = df['Title'].fillna('').str.strip()

df['Content'] = df['Content'].fillna('').str.strip()

article_text = (df['Title'].fillna('') + '. ' + df['Content'].fillna('')).str.strip()
df['Text'] = article_text
with open('Model_training/risks.json', 'r') as f:
    risks_data = json.load(f)

all_risks = [risk['name'] for group in risks_data['new_risks'] for risks in group.values() for risk in risks]

model = SentenceTransformer('all-mpnet-base-v2')
# Encode articles and risks
article_embeddings = model.encode(df['Text'].tolist(), show_progress_bar=True, convert_to_tensor=True)
risk_embeddings = model.encode(all_risks, convert_to_tensor=True)

# Calculate cosine similarity
cosine_scores = util.cos_sim(article_embeddings, risk_embeddings)

# Assign risks based on threshold
threshold = 0.55  # you can tune this
predicted_risks = []
for i in range(len(df)):
    scores = cosine_scores[i]
    matched_risks = [all_risks[j] for j, score in enumerate(scores) if score >= threshold]
    predicted_risks.append(matched_risks if matched_risks else ["No Risk"])

df['Predicted_Risks_new'] = ['; '.join(risks) for risks in predicted_risks]

df.to_csv('risk_predictions.csv', index=False)
print("✅ Risk predictions saved to risk_predictions.csv")
