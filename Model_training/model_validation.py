from sentence_transformers import SentenceTransformer, util
import json
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import os
import pandas as pd

df = pd.read_csv('Model_training/train_articles.csv')



risk_list = os.getenv('RISKS_LIST')

df['Top_Risks'] = df['Top_Risks'].apply(lambda x: [r.strip() for r in x.split(';') if r.strip()] if pd.notna(x) else "No Risk")

labeled = df[df['Top_Risks'].apply(lambda x: x != ["No Risk"])]
unlabeled = df[df['Top_Risks'].apply(lambda x: x == ["No Risk"])]

sampled_unlabeled = unlabeled.sample(frac = 0.5, random_state = 42)

df = pd.concat([labeled, sampled_unlabeled]).sample(frac = 1, random_state = 42).reset_index(drop = True)    

article_text = df['Title'] + '. ' + df['Content']

mlb = MultiLabelBinarizer(classes = risk_list)
y = mlb.fit_transform(df['Top_Risks'])

model = SentenceTransformer('all-mpnet-base-v2')
X = model.encode(article_text.tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=mlb.classes_))
