import requests
import json
import os
import tempfile
import gzip

retrain_threshold = 300
model_tag = 'regression'
model_asset_name = 'model_bundle.pkl'
meta_asset_name = 'training_meta.json'
Github_owner = 'ERSRisk'
Github_repo = 'Tulane-Sentiment-Analysis'
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')


def gh_headers():
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        raise RuntimeError('Missing token')
    return {
        'Accept':"application/vnd.github+json",
        "Authorization": f"token {token if token else None}",
    }

def ensure_release(owner, repo, tag:str, token):
    headers = gh_headers()
    r = requests.get(f'https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}', headers=headers, timeout=60)
    if r.status_code == 404:
        r = requests.post(f'https://api.github.com/repos/{owner}/{repo}/releases', headers = headers, timeout = 60, json={
        "tag_name": tag, "name": tag, "draft": False, "prerelease": False
    })
    r.raise_for_status()
    rel = r.json()
    return rel

def upload_asset(owner, repo, release, asset_name, data_bytes, token, content_type = 'application/gzip'):
    assets_api = release['assets_url']
    r = requests.get(assets_api, headers = gh_headers())
    r.raise_for_status()
    for a in r.json():
        if a.get("name") == asset_name:
            del_r = requests.delete(a['url'], headers = gh_headers())
            del_r.raise_for_status()
            break
    upload_url = release['upload_url'].split('{')[0]
    params = {"name": asset_name}
    headers = gh_headers()
    headers['Content-Type'] = content_type
    up = requests.post(upload_url, headers = headers, params = params, data = data_bytes)
    if not up.ok:
        raise RuntimeError(f"Upload failed{up.status_code}: {up.text[:500]}")
    return up.json()


def upload_asset_to_release(owner, repo, tag:str, asset_path:str, token:str):
        headers = {'Authorization': f'token {token}',
                  'Accept': 'application/vnd.github+json'}
        r = requests.get(f'https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}', headers=headers, timeout=60)
        r.raise_for_status()
        rel = r.json()
        upload_url = rel["upload_url"].split("{", 1)[0]
        assets = requests.get(f"https://api.github.com/repos/{owner}/{repo}/releases/{rel['id']}/assets", headers=headers, timeout=60).json()
        name = os.path.basename(asset_path)
        for a in assets:
            if a.get("name") == name:
                requests.delete(f"https://api.github.com/repos/{owner}/{repo}/releases/assets/{a['id']}", headers=headers, timeout=60)
        with open(asset_path, "rb") as f:
            up = requests.post(
                f"{upload_url}?name={name}",
                headers={"Authorization": f"token {token}", "Content-Type": "application/octet-stream"},
                data=f.read(), timeout=300
            )
        up.raise_for_status()
        return up.json()





def fetch_release(owner, repo, tag:str, asset_name:str, token:str):
    headers = {'Authorization': f'token {token}',
              'Accept': 'application/vnd.github+json'}
    r = requests.get(f'https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}', headers=headers, timeout=60)
    if r.status_code == 404:
        r = requests.post(f'https://api.github.com/repos/{owner}/{repo}/releases', headers = headers, timeout = 60, json={
        "tag_name": tag, "name": tag, "draft": False, "prerelease": False
    })
    r.raise_for_status()
    rel = r.json()
    upload_url = rel['upload_url'].split('{', 1)[0]
    assets = requests.get(f"https://api.github.com/repos/{owner}/{repo}/releases/{rel['id']}/assets", headers=headers, timeout=60).json()
    a = next((x for x in assets if x.get("name") == asset_name), None)
    if not a:
        return []

    url = a.get('browser_download_url')
    b = requests.get(url, headers = headers, timeout = 120)
    b.raise_for_status()
    content = b.content
    return json.loads(content.decode('utf-8'))


def get_model_and_meta(owner, repo, tag, token):
    try:
        with open('training_meta.json', 'r', encoding = 'utf-8') as f:
            meta = json.load(f)
        if isinstance(meta, dict):
            print(meta['row_count'])
            return meta
        if isinstance(meta, list) and meta:
            return meta[0]
        return {}
    except Exception:
        return {}

def write_training_meta(owner, repo, tag, token, meta: dict):
    payload = [{'__singleton__':1, **meta}]
    with open('training_meta.json', 'w', encoding = 'utf-8') as f:
        json.dump(payload, f)
    return True

def retrain_and_publish(df):
    
   

    
    df = df.dropna(subset = ['Recency_Upd', 'Acceleration_value_Upd'])
    row_count = len(df)
    print(row_count)
    last_retrain = get_model_and_meta(Github_owner, Github_repo, model_tag, GITHUB_TOKEN)
    print(last_retrain)
    last_count = last_retrain.get('row_count', 0)
    print(last_count)
    if row_count - last_count < retrain_threshold:
        print("Not enough new data to retrain the model.")
        return
    print("Retraining the model...")
    from sentence_transformers import SentenceTransformer, util
    import pandas as pd
    import torch
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    import json
    import joblib
    import numpy as np
    from sentence_transformers import SentenceTransformer, util
    import ast

    risk_defs = {
  "Research Funding Disruption": "University research funding halted or withdrawn. Grant cuts, pauses, or canceled awards stop lab projects and furlough staff.",
  "Enrollment Pressure": "Fewer applications or lower student retention reduce tuition revenue. Admissions decline or FAFSA delays cause enrollment stress.",
  "Policy or Political Interference": "State or federal officials intervene in campus DEI or curriculum policies through mandates or funding threats.",
  "Institutional Alignment Risk": "University units pursue conflicting goals. Misaligned strategies or budgets stall progress on institutional plans.",
  "Mission Drift": "University focuses on revenue or prestige projects over teaching and research, weakening academic mission.",
  "Revenue Loss": "University faces financial shortfall due to budget cuts, declining tuition, or reduced auxiliary income.",
  "Insurance Market Volatility": "University insurance premiums rise or coverage is reduced after market hardening or claims disputes.",
  "Unexpected Expenditures": "Campus hit by sudden unplanned costs from facility failures, legal settlements, or emergency repairs.",
  "Endowment Risk": "University endowment loses value or liquidity, forcing payout cuts that affect scholarships or operations.",
  "Infrastructure Failure": "Campus systems fail. Power, HVAC, or network outages disrupt classes and research activity.",
  "Vendor Cyber Exposure": "University data exposed through a third-party SaaS or vendor security breach or SOC 2 gap.",
  "Unauthorized Access/Data Breach": "Hackers access internal university systems or personal records, requiring breach response.",
  "Artificial Intelligence Ethics & Governance": "Campus adoption of AI raises fairness, bias, or transparency issues needing governance policy.",
  "Rapid Speed of Disruptive Innovation": "University processes lag behind rapid digital transformation or automation efforts.",
  "Controversial Public Incident": "Campus statement, protest, or viral video sparks public backlash and reputational scrutiny.",
  "DEI Program Backlash": "Political or donor pressure challenges diversity and inclusion programs on campus.",
  "Leadership Missteps": "University leaders issue misleading statements or mishandle a crisis, prompting criticism or resignations.",
  "High-Profile Litigation": "University faces a lawsuit drawing major public or media attention, often around discrimination or research conduct.",
  "Violence or Threats": "Campus shooting, assault, or credible threat causes lockdowns or safety alerts for students and staff.",
  "Emergency Preparedness Gaps": "Campus emergency plans prove outdated or untested, delaying response during a crisis.",
  "Infectious Disease Outbreak": "Cluster of student or staff illness disrupts classes or triggers campus health measures.",
  "Lab Incident": "Chemical spill or fire in a university lab injures staff or halts research pending investigation.",
  "Environmental Exposure": "Hazardous materials like asbestos, lead, or mold found in campus buildings trigger closures.",
  "Hurricane/Flood/Wildfire": "Natural disaster damages campus property and displaces students or staff.",
  "Student Conduct Incident": "Fraternity hazing, fights, or misconduct lead to discipline, injuries, or suspension.",
  "Academic Disruption": "Classes canceled or delayed due to strikes, outages, or emergencies on campus.",
  "Mental Health Crises": "Campus counseling overwhelmed by student mental health emergencies or suicide risk.",
  "HR Complaint": "Employee alleges harassment, discrimination, or retaliation, leading to internal investigation.",
  "Labor Dispute": "Faculty or staff strike or protest disrupts classes and campus operations.",
  "Whistleblower Claims": "Employee reports internal fraud or safety cover-up, prompting investigation or retaliation concerns.",
  "Accreditation Risk": "Accreditor flags weaknesses in governance, finances, or academic outcomes, threatening status."
}

    df['Title'] = df['Title'].fillna('').str.strip()
    df['Content'] = df['Content'].fillna('').str.strip()
    df['Change reason'] = df['Change reason'].fillna('').str.strip()

    df['Top_Risks'] = df['Predicted_Risks_Upd'].fillna('No Risk')
    def parse_risk_list(x):
        if isinstance(x, list):
            items = [s.strip() for s in x if isinstance(s, str) and s.strip()]
        else:
            s = '' if pd.isna(x) else str(x)
            if s.strip().lower() == 'no risk' or not s.strip():
                items = []
            else:
                items = [t.strip() for t in s.split(';') if t.strip()]
        return items
    df['Top_Risks'] = df['Top_Risks'].apply(parse_risk_list)
    df['Top_Risks'] = df['Top_Risks'].apply(lambda lst: (lst[0] if lst else 'No Risk'))
    print(df['Top_Risks'].map(type).value_counts())
    def clean_risk_label(x):
        if isinstance(x, str):
            s = x.strip()
            if s.startswith('['):
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list,tuple)) and parsed:
                        return parsed[0]
                except Exception:
                    pass
            return s

    df['Top_Risks'] = df['Top_Risks'].apply(clean_risk_label).fillna('No Risk')

    gold = df[df['Top_Risks'] != 'No Risk'].copy()
    vc = gold['Top_Risks'].value_counts()
    keep = vc[vc >= 10].index
    gold = gold[gold['Top_Risks'].isin(keep)].reset_index(drop=True)

    article_text = gold['Title'] + '. ' + gold['Content']
    change_reason = gold['Change reason']
    topic_col_name = next((c for c in ["Topic_Upd", "Topic"] if c in df.columns), None)
    prob_col_name  = next((c for c in ["Probability_Upd", "Probability"] if c in df.columns), None)
    if topic_col_name is not None:
        df[topic_col_name] = pd.to_numeric(df[topic_col_name], errors="coerce").fillna(-1).astype(int)
    if prob_col_name is not None:
        df[prob_col_name] = pd.to_numeric(df[prob_col_name], errors="coerce").fillna(0.0)
    if topic_col_name is not None:
        # work on GOLD so we don’t leak info
        gold["_topic_raw"] = pd.to_numeric(gold[topic_col_name], errors="coerce").fillna(-1).astype(int)

        top_k = 50  # tune 30–100
        top_topics = gold["_topic_raw"].value_counts().head(top_k).index.tolist()

        # rare topics and -1 -> other
        gold["_topic_binned"] = np.where(gold["_topic_raw"].isin(top_topics), gold["_topic_raw"], -1)

        # one-hot → DataFrame aligned to gold.index
        topic_ohe = pd.get_dummies(gold["_topic_binned"].astype("int64"), prefix="topic", dtype=int)

    else:
        topic_ohe = pd.DataFrame(index=gold.index)

    topic_top_ids = top_topics
    topic_ohe_cols = topic_ohe.columns.tolist()


    numeric_factors = [
        'Recency_Upd',
        'Acceleration_value_Upd',
        'Impact_Score_Upd',
        'Source_Accuracy_Upd',
        'Location_Upd',
        'Industry_Risk_Upd',
        'Frequency_Score_Upd',
        'Risk_Score_Upd'
    ]
    if prob_col_name is not None:
        numeric_factors.append(prob_col_name)
    numeric_cols = [c for c in numeric_factors if c in gold.columns]  # <- use gold
    num_mat = gold[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()

    # X_text_red already built from A, C (optionally PCA-reduced)
    # Add topic one-hot to the tail


    sentence_model = SentenceTransformer('all-mpnet-base-v2')
    A = sentence_model.encode(article_text.tolist(), show_progress_bar=True, normalize_embeddings=True)
    C = sentence_model.encode(change_reason.tolist(), show_progress_bar=True, normalize_embeddings=True)

    from sklearn.decomposition import PCA
    X_text = np.hstack([A, C])

    n_samples, n_features = X_text.shape
    n_components =min(384, n_features, max(2, n_samples - 1))

    if n_components >= 2:
        pca = PCA(n_components=n_components, random_state=42, svd_solver='auto')
        X_text_red = pca.fit_transform(X_text)
    else:
        # too few samples to do meaningful PCA; skip it
        X_text_red = X_text
        print(f"[PCA] Skipped PCA (n_samples={n_samples}, n_features={n_features}).")

    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(num_mat)

    topic_ohe = topic_ohe.reindex(gold.index, fill_value=0)

    X_all = np.hstack([
        X_text_red,
        num_scaled,
        topic_ohe.values  # (may be empty if topic_col missing)
    ])

    emb_avg = 0.5 * (A + C)  # <- build from encoded matrices

    le = LabelEncoder()
    y_all = le.fit_transform(gold['Top_Risks'])

    Xe_tr, Xe_te, y_tr, y_te, avg_tr, avg_te = train_test_split(
        X_all, y_all, emb_avg, test_size=0.20, stratify=y_all, random_state=42
    )
    Xe_tr, Xe_va, y_tr, y_va, avg_tr, avg_va = train_test_split(
        Xe_tr, y_tr, avg_tr, test_size=0.20, stratify=y_tr, random_state=42
    )

    ros = RandomOverSampler(random_state=42)
    Xe_tr, y_tr = ros.fit_resample(Xe_tr, y_tr)

    clf = LogisticRegression(
        solver='lbfgs',
        multi_class='auto',
        C=0.3,
        max_iter=2000,
        n_jobs=1
    )
    clf.fit(Xe_tr, y_tr)

    label_emb = sentence_model.encode(le.classes_.tolist(), show_progress_bar=False, normalize_embeddings=True)

    def soft_cos(avg_emb, label_emb):
        cos = avg_emb @ label_emb.T
        cos = (cos + 1.0)/2.0
        return cos / (cos.sum(axis=1, keepdims=True) + 1e-12)

    # --- VALIDATION: compute probs and cosine, normalize avg rows before cosine
    proba_va = clf.predict_proba(Xe_va)
    avg_va = avg_va / (np.linalg.norm(avg_va, axis=1, keepdims=True) + 1e-12)
    cos_va = soft_cos(avg_va, label_emb)

    def predict_with_fallback(proba, cos, prob_cut=0.55, margin_cut=0.10):
        top_idx = proba.argmax(axis=1)
        top_val = proba[np.arange(len(proba)), top_idx]
        proba2 = proba.copy()
        proba2[np.arange(len(proba2)), top_idx] = -1
        second = proba2.max(axis=1)
        margin = top_val - second
        use_lr = (top_val >= prob_cut) & (margin >= margin_cut)
        y_lr = top_idx
        y_cos = cos.argmax(axis=1)
        y_hat = np.where(use_lr, y_lr, y_cos)
        return y_hat, use_lr

    # --- GRID SEARCH (inline; no new helpers)
    trained_labels = le.classes_.tolist()
    label_text_tr = [f"{lbl}: {risk_defs.get(lbl, lbl)}" for lbl in trained_labels]
    label_emb_tr = sentence_model.encode(label_text_tr, show_progress_bar=False, normalize_embeddings=True)

    # 2) Embeddings for ALL label names (for real-world fallback)
    all_labels = list(risk_defs.keys())
    label_text_all = [f"{lbl}: {risk_defs.get(lbl, '')}" for lbl in all_labels]
    label_emb_all = sentence_model.encode(label_text_all, show_progress_bar=False, normalize_embeddings=True)

    # 3) Validation: LR probs + cosine (trained-only for tuning)
    proba_va = clf.predict_proba(Xe_va)
    avg_va = avg_va / (np.linalg.norm(avg_va, axis=1, keepdims=True) + 1e-12)
    cos_va_tr = soft_cos(avg_va, label_emb_tr)      # for tuning
    cos_va_all = soft_cos(avg_va, label_emb_all)    # for final val predictions

    # 4) Grid search thresholds on validation using trained-label cosine
    best_pc, best_mc, best_f1, best_acc, best_share = None, None, -1.0, -1.0, None
    for pc in np.arange(0.30, 0.76, 0.05):
        for mc in np.arange(0.00, 0.21, 0.05):
            y_tmp, used_tmp = predict_with_fallback(proba_va, cos_va_tr, prob_cut=pc, margin_cut=mc)
            f1_tmp = f1_score(y_va, y_tmp, average='weighted', zero_division=0)
            acc_tmp = accuracy_score(y_va, y_tmp)
            if (f1_tmp > best_f1) or (np.isclose(f1_tmp, best_f1) and acc_tmp > best_acc):
                best_pc, best_mc, best_f1, best_acc, best_share = pc, mc, f1_tmp, acc_tmp, used_tmp.mean()

    print(f"[val-tune] prob_cut={best_pc:.2f} margin_cut={best_mc:.2f}  F1={best_f1:.3f} Acc={best_acc:.3f} LR_used={best_share:.2f}")

    # 5) Final VALIDATION predictions:
    #    - Use LR where confident
    #    - Else use cosine over ALL labels (so rare/unseen labels can be chosen)
    y_va_lr_idx = proba_va.argmax(axis=1)
    y_va_lr_names = np.array(trained_labels)[y_va_lr_idx]

    # Recompute used_lr mask with the tuned thresholds (on trained-label cosine)
    y_va_hat_trained, used_lr_va = predict_with_fallback(proba_va, cos_va_tr, prob_cut=0.45, margin_cut=0.2)

    # Cosine over all labels for the fallback rows

    y_va_cos_all_idx = cos_va_all.argmax(axis=1)
    y_va_cos_all_names = np.array(all_labels)[y_va_cos_all_idx]

    tau = 0.35
    cos_va_all_max = cos_va_all.max(axis=1)


    no_risk_mask_va = (~used_lr_va) & (cos_va_all_max < tau)



    # Combine to final VAL names
    y_va_names = le.inverse_transform(y_va)                    # ground-truth names (trained labels)
    y_va_hat_names = np.where(
        used_lr_va,
        y_va_lr_names,                                   # LR when confident
        np.where(cos_va_all_max < tau, 'No Risk', y_va_cos_all_names)  # else cosine over ALL labels with No Risk fallback
    )

    print("Validation share using LR:", used_lr_va.mean())
    print("Validation accuracy:", accuracy_score(y_va_names, y_va_hat_names))
    print("Validation weighted F1:", f1_score(y_va_names, y_va_hat_names, average='weighted', zero_division=0))
    present_va = np.unique(np.concatenate([y_va_names, y_va_hat_names]))
    print(classification_report(y_va_names, y_va_hat_names, labels=present_va, zero_division=0))

    # 6) TEST predictions with the SAME thresholds; cosine over ALL labels for fallback
    proba_te = clf.predict_proba(Xe_te)
    avg_te = avg_te / (np.linalg.norm(avg_te, axis=1, keepdims=True) + 1e-12)
    cos_te_tr = soft_cos(avg_te, label_emb_tr)
    cos_te_all = soft_cos(avg_te, label_emb_all)



    # LR top on test (trained labels)
    y_te_lr_idx = proba_te.argmax(axis=1)
    y_te_lr_names = np.array(trained_labels)[y_te_lr_idx]

    # Confidence mask from trained-label cosine
    y_te_hat_trained, used_lr_te = predict_with_fallback(proba_te, cos_te_tr, prob_cut=0.45, margin_cut=0.2)

    # Fallback via ALL labels
    y_te_cos_all_idx = cos_te_all.argmax(axis=1)
    y_te_cos_all_names = np.array(all_labels)[y_te_cos_all_idx]

    cos_te_all_max = cos_te_all.max(axis=1)


    no_risk_mask_te = (~used_lr_te) & (cos_te_all_max < tau)

    # Combine to final TEST names
    y_te_names = le.inverse_transform(y_te)                    # ground-truth names (trained labels)
    y_te_hat_names = np.where(
        used_lr_te,
        y_te_lr_names,
        np.where(cos_te_all_max < tau, 'No Risk', y_te_cos_all_names)
    )

    print("Test share using LR:", used_lr_te.mean())
    print("Test accuracy:", accuracy_score(y_te_names, y_te_hat_names))
    print("Test weighted F1:", f1_score(y_te_names, y_te_hat_names, average='weighted', zero_division=0))
    present_te = np.unique(np.concatenate([y_te_names, y_te_hat_names]))
    print(classification_report(y_te_names, y_te_hat_names, labels=present_te, zero_division=0))

    import joblib
    from pathlib import Path
    ART_DIR = Path("Model_training/")
    ART_DIR.mkdir(parents=True, exist_ok=True)

    # If you used PCA above, it exists as 'pca'. If you skipped it, set to None.
    pca_to_save = pca if 'pca' in locals() else None

    bundle = {
        "clf": clf,                                     # trained LogisticRegression
        "scaler": scaler,                               # StandardScaler for numeric features
        "pca": pca_to_save,                             # PCA for text features (or None)
        "label_encoder": le,                            # maps ids <-> names for TRAINED labels
        "trained_label_names": trained_labels,          # list[str]
        "risk_defs": risk_defs,                         # dict for creating label text
        "sentence_model_name": "all-mpnet-base-v2",     # <-- keep in sync with training
        # Fallback tuning
        "best_prob_cut": 0.42,                # e.g. 0.50
        "best_margin_cut": 0.2,              # e.g. 0.00
        "openset_tau": 0.35,                            # cosine threshold for 'No Risk'
        # Text used for cosine (trained + open-set)
        "trained_label_text": [f"{lbl}: {risk_defs.get(lbl, lbl)}" for lbl in trained_labels],
        "all_labels": list(risk_defs.keys()),
        "all_label_text": [f"{lbl}: {risk_defs.get(lbl, '')}" for lbl in risk_defs.keys()],
        # Feature config
        "numeric_factors": [
            "Recency_Upd","Acceleration_value_Upd","Impact_Score_Upd",
            "Source_Accuracy_Upd","Location_Upd","Industry_Risk_Upd",
            "Frequency_Score_Upd","Risk_Score_Upd"
        ],
        "topic_top_ids": list(map(int, topic_top_ids)),  # list[int]
        "topic_ohe_cols": topic_ohe_cols,
    }

    joblib.dump(bundle, ART_DIR / "model_bundle.pkl")
    print(f"Saved model bundle to {ART_DIR / 'model_bundle.pkl'}")

    rel = ensure_release(Github_owner, Github_repo, model_tag, GITHUB_TOKEN)
    bundle_path = Path('Model_training/model_bundle.pkl')
    upload_asset(Github_owner, Github_repo, rel, model_asset_name,
                 data_bytes=bundle_path.read_bytes(),
                 token=GITHUB_TOKEN,
                 content_type="application/octet-stream"
    )
    print('Uploaded new model to releases', flush = True)
    meta = {
        'row_count': int(row_count),
        'threshold': retrain_threshold,
    }
    with open('Model_training/training_meta.json', 'w', encoding = 'utf-8') as f:
        json.dump([{'__singleton__':1, **meta}], f)
    print('Uploaded new training meta', flush = True)
    return True

import pandas as pd
if __name__ == '__main__':
    df = pd.read_csv('Model_training/BERTopic_changes.csv')
    retrain_and_publish(df)
