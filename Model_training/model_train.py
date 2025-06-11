import pandas as pd
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader
import random 
import joblib

df = pd.read_csv('train_articles.csv')

df['Top_Risks'] = df['Top_Risks'].apply(lambda x: [r.strip() for r in x.split(';')] if isinstance(x, str) else [])
labeled = df[df['Top_Risks'].notna()]
blanks = df[df['Top_Risks'].isna()].sample(frac = 0.2, random_state = 42)

training_data = pd.concat([labeled, blanks])

training_data['Top_Risks'] = training_data['Top_Risks'].apply(lambda x: x if x else ['None'])
print(training_data.head())
train_examples = []
for _, row in df.iterrows():
    article = row['Content']
    risks = row['Top_Risks']
    for risk in risks:
        train_examples.append(InputExample(texts = [article, risk]))

random.shuffle(train_examples)

train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

model = SentenceTransformer('all-mpnet-base-v2')
train_loss = losses.MultipleNegativesRankingLoss(model)

model.fit(
    train_objectives = [(train_dataloader, train_loss)],
    epochs = 4,
    warmup_steps = 100
)

model.save("my_model_directory")
