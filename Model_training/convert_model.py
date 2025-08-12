import json
from sentence_transformers import SentenceTransformer, util
import pandas as pd

df = pd.read_csv("Model_training/BERTopic_results.csv")
df = df[df['University Label'] == 1]
model = SentenceTransformer('all-mpnet-base-v2')

def predict_risks(df):

    df['Title'] = df['Title'].fillna('').str.strip()
    
    df['Content'] = df['Content'].fillna('').str.strip()
    df['Text'] = (df['Title'] + '. ' + df['Content']).str.strip()
    df = df.reset_index(drop = True)

    with open('Model_training/risks.json', 'r') as f:
        risks_data = json.load(f)
    
    all_risks = [risk['name'] for group in risks_data['new_risks'] for risks in group.values() for risk in risks]
    
    
    # Encode articles and risks
    article_embeddings = model.encode(df['Text'].tolist(), show_progress_bar=True, convert_to_tensor=True)
    risk_embeddings = model.encode(all_risks, convert_to_tensor=True)
    
    # Calculate cosine similarity
    cosine_scores = util.cos_sim(article_embeddings, risk_embeddings)

    if 'Predicted_Risks_new' not in df.columns:
        df['Predicted_Risks_new'] = ''
    # Assign risks based on threshold
    threshold = 0.35  # you can tune this
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

new_df = predict_risks(df)
new_df.to_csv('Model_training/BERTopic_results1.csv')
