import re
import pandas as pd
import numpy as np
import json
from src.storage.github_releases import load_model_bundle
import torch
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai.errors import ClientError, ServerError
import time
import random
import requests

from config.settings import (
    GITHUB_OWNER,
    GITHUB_REPO
)

from src.utils.gemini import get_gemini_client

client = get_gemini_client()


def predict_risks(df):
    def soft_cosine_probs(vecs, label_emb):
        cos = vecs @ label_emb.T
        cos = (cos + 1.0) / 2.0
        denom = cos.sum(axis=1, keepdims=True) + 1e-12
        return cos / denom

    def has_any(t, terms):
        t = str(t).lower()
        for term in terms:
            term = str(term).lower().strip()
            pattern = r'(?<!\w)' + re.escape(term).replace(r'\ ', r'\s+') + r'(?!\w)'
            if re.search(pattern, t):
                return True
        return False

    def dedupe_keep_order(items):
        seen = set()
        out = []
        for x in items:
            if x is None:
                continue
            x = str(x).strip()
            if not x:
                continue
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    def rule_route(text, label="No Risk"):
        t = str(text).lower()
        label = "No Risk" if pd.isna(label) or str(label).strip() == "" else str(label).strip()

        student_conduct_terms = [
            "hazing", "fraternity", "sorority", "greek life",
            "student misconduct", "disciplinary violation", "student discipline",
            "student suspension", "student fight", "alcohol violation",
            "drug violation", "title ix complaint", "sexual misconduct"
        ]

        ai_terms = [
            "artificial intelligence", " ai ", "ai policy", "ai literacy",
            "generative ai", "chatgpt", "machine learning", "algorithmic",
            "automated decision", "ai tools"
        ]

        loan_terms = [
            "student loan", "loan limits", "graduate loan", "professional degree",
            "fafsa", "financial aid", "borrowers", "repayment plan",
            "loan forgiveness", "tuition affordability"
        ]

        funding_disruption_terms = [
            "funding cut", "funding cuts", "grant cut", "grant cuts",
            "freeze", "frozen", "terminated grants", "cancelled grants",
            "canceled grants", "rescinded", "clawback",
            "cap on indirect costs", "funding pause", "withheld funding"
        ]

        positive_funding_terms = [
            "awarded", "grant awarded", "received funding", "new grant",
            "funding to support", "launches center", "new research center",
            "cooperative agreement"
        ]

        labor_terms = [
            "strike", "union", "collective bargaining", "contract negotiations",
            "labor dispute", "workers", "faculty union", "graduate student workers"
        ]

        policy_terms = [
            "department of education", "education department",
            "department of justice", "doj",
            "civil rights investigation", "office for civil rights",
            "title vi", "title ix",
            "federal investigation", "executive order",
            "department of homeland security", "dhs",
            "visa restriction", "deportation", "sevp",
            "in-state tuition", "undocumented students",
            "federal rule on higher education",
            "state law on higher education",
            "anti-dei", "dei ban"
        ]

        physical_threat_terms = [
            "shooting",
            "gunfire",
            "firearm",
            "firearms",
            "armed suspect",
            "bomb threat",
            "explosion",
            "stabbing",
            "stabbed",
            "lockdown",
            "active shooter",
            "weapon",
            "weapons",
            "homicide",
            "mass shooting",
            "violent threat"
        ]

        immigration_terms = [
            "visa", "international student", "foreign student", "dhs",
            "sevp", "deport", "deportation", "immigration", "ice"
        ]

        protest_terms = [
            "protest", "protesters", "demonstration", "activist",
            "detained", "arrested", "campus protest"
        ]

        # Deterministic obvious cases / cleanup guards.
        if label == "Student Conduct Incident" and not has_any(t, student_conduct_terms):
            if has_any(t, protest_terms + immigration_terms):
                return "Policy or Political Interference"
            return "No Risk"

        if label == "Violence or Threats" and not has_any(t, physical_threat_terms):
            return "No Risk"

        if label == "Research Funding Disruption":
            if has_any(t, positive_funding_terms) and not has_any(t, funding_disruption_terms):
                return "No Risk"

        if label == "Revenue Loss":
            if not has_any(t, ['tulane', 'tulane university']):
                return "No Risk"

        if label == "Policy or Political Interference":
            if not has_any(t, policy_terms):
                return "No Risk"

        if has_any(t, ai_terms):
            if label in ["No Risk", "Student Conduct Incident", "DEI Program Backlash"]:
                return "Artificial Intelligence Ethics & Governance"

        if has_any(t, [
            "infectious disease",
            "disease outbreak",
            "viral outbreak",
            "bacterial outbreak",
            "pandemic",
            "epidemic",
            "communicable disease",
            "virus spread",
            "cdc outbreak",
            "measles outbreak",
            "flu outbreak",
            "covid outbreak",
            "norovirus",
            "avian flu",
            "h5n1"
        ]):
            if label in ["No Risk", "Emergency Preparedness Gaps"]:
                return "Infectious Disease Outbreak"

        if has_any(t, loan_terms):
            if label in ["No Risk", "Student Conduct Incident"]:
                return "Enrollment Pressure"

        if has_any(t, labor_terms):
            if label in ["No Risk", "DEI Program Backlash"]:
                return "Labor Dispute"

        if has_any(t, ["cyberattack", "data breach", "hackers", "vendor", "third-party vendor"]):
            if label in ["No Risk", "Student Conduct Incident"]:
                return "Vendor Cyber Exposure"

        return label

    def predict_with_fallback(proba_lr, cos_all, prob_cut, margin_cut, tau, tau_gray, trained_labels, all_labels):
        top_idx = proba_lr.argmax(axis=1)
        top_val = proba_lr[np.arange(len(proba_lr)), top_idx]

        tmp = proba_lr.copy()
        tmp[np.arange(len(tmp)), top_idx] = -1
        second = tmp.max(axis=1)
        margin = top_val - second

        lr_mask = (top_val >= prob_cut) & (margin >= margin_cut)

        cos_all_max = cos_all.max(axis=1)
        cos_all_idx = cos_all.argmax(axis=1)

        lr_names = np.array(trained_labels)[top_idx]
        cos_names = np.array(all_labels)[cos_all_idx]

        route = np.full(len(top_val), 'norisk', dtype=object)
        final = np.array(['No Risk'] * len(top_val), dtype=object)

        route[lr_mask] = 'lr'
        final[lr_mask] = lr_names[lr_mask]

        cos_hi = (~lr_mask) & (cos_all_max >= tau)
        route[cos_hi] = "gray"
        final[cos_hi] = cos_names[cos_hi]

        gray = (~lr_mask) & (cos_all_max >= tau_gray) & (cos_all_max < tau)
        route[gray] = "gray"
        final[gray] = cos_names[gray]

        return {
            "final_names": final,
            "route": route,
            "lr_top_prob": top_val,
            "lr_top_idx": top_idx,
            "lr_margin": margin,
            "cos_all_idx": cos_all_idx,
            "cos_all_max": cos_all_max,
            "cos_names": cos_names
        }

    if 'Published_utc' not in df.columns:
        df['Published_utc'] = pd.to_datetime(df['Published'], errors='coerce', utc=True)
    else:
        df['Published_utc'] = pd.to_datetime(df['Published_utc'], errors='coerce', utc=True)

    with open('pipeline/resources/risks.json', 'r', encoding='utf-8') as f:
        risks_cfg = json.load(f)

    json_all_labels = [
        r['name']
        for block in risks_cfg.get('new_risks', [])
        for _, items in block.items()
        for r in items
    ]

    bundle = load_model_bundle(GITHUB_OWNER, GITHUB_REPO, 'regression')

    clf = bundle['clf']
    scaler = bundle['scaler']
    pca = bundle['pca']
    le = bundle['label_encoder']
    trained_labels = bundle['trained_label_names']
    risk_defs = bundle['risk_defs']
    model_name = bundle['sentence_model_name']

    prob_cut = 0.80
    margin_cut = 0.15
    tau = 0.78
    tau_gray = 0.55

    numeric_factors = list(bundle['numeric_factors'])
    trained_label_txt = list(bundle['trained_label_text'])
    all_labels = json_all_labels
    all_label_txt = list(bundle['all_label_text'])

    df = df.copy()
    df = df.sort_values('Published_utc').drop_duplicates('Title', keep='last').reset_index(drop=True)

    df['University Label'] = pd.to_numeric(
        df['University Label'],
        errors='coerce'
    ).fillna(0).astype(int)

    mask_he = df['University Label'] == 1

    df['Title'] = df['Title'].fillna('').str.strip()
    df['Content'] = df['Content'].fillna('').str.strip()
    df['Text'] = (df['Title'] + '. ' + df['Content']).str.strip()

    df = df.reset_index(drop=True)

    stale_mask = df.get('pred_source', '').astype(str).eq('rule_pre_gemini')
    for col in ['Predicted_Risks_new', 'Predicted_Risks', 'pred_source']:
        if col in df.columns:
            df.loc[stale_mask, col] = ''

    recent_cut = pd.Timestamp.now(tz='utc') - pd.Timedelta(days=30)
    recent_mask = df['Published_utc'] >= recent_cut

    if 'Predicted_Risks_new' in df.columns:
        todo_mask = mask_he & recent_mask.fillna(False)
        sub = df.loc[todo_mask].copy()
        texts = df.loc[todo_mask, 'Text'].tolist()
    else:
        todo_mask = pd.Series(True, index=df.index)
        df['Published_utc'] = pd.to_datetime(df['Published'], errors='coerce', utc=True)
        recent_mask = df['Published_utc'] >= recent_cut
        todo_mask &= recent_mask.fillna(False)
        todo_mask &= mask_he
        sub = df.loc[todo_mask].copy()
        texts = df.loc[todo_mask, 'Text'].tolist()

    print(f"[dbg] total rows: {len(df)}", flush=True)
    print(f"[dbg] parsable Published: {df['Published_utc'].notna().sum()}", flush=True)
    print(f"[dbg] recent (<=30d): {recent_mask.fillna(False).sum()}", flush=True)
    print(f"[dbg] to score (todo_mask): {todo_mask.sum()}", flush=True)

    if not texts:
        return df

    all_risks = [
        risk['name']
        for group in risks_cfg['new_risks']
        for risks in group.values()
        for risk in risks
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-mpnet-base-v2', device=device)

    article_embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=256 if device == 'cuda' else 32
    )

    A = article_embeddings
    C = np.zeros_like(A)
    X_text = np.hstack([article_embeddings, C])
    X_text_red = pca.transform(X_text) if pca is not None else X_text

    n = len(texts)
    if len(numeric_factors) == 0:
        num_scaled = np.zeros((n, 0))
    else:
        means = np.asarray(bundle['scaler'].mean_, dtype=float)
        num_mat = np.tile(means, (n, 1))
        num_scaled = bundle['scaler'].transform(num_mat)

    topic_col_name = 'Topic'
    top_ids = bundle.get('topic_top_ids', [])
    ohe_cols_expected = bundle.get('topic_ohe_cols', [])

    topic_raw = pd.to_numeric(
        sub.get(topic_col_name),
        errors='coerce'
    ).fillna(-1).astype(int)

    topic_binned = np.where(np.isin(topic_raw, top_ids), topic_raw, -1)

    topic_ohe = pd.get_dummies(
        pd.Series(topic_binned),
        prefix='topic',
        dtype=int
    )

    for col in ohe_cols_expected:
        if col not in topic_ohe:
            topic_ohe[col] = 0

    topic_ohe = topic_ohe[ohe_cols_expected].to_numpy()

    X_all = np.hstack([X_text_red, num_scaled, topic_ohe])

    proba = clf.predict_proba(X_all)

    avg_emb = article_embeddings
    avg_emb = avg_emb / (np.linalg.norm(avg_emb, axis=1, keepdims=True) + 1e-12)

    lbl_emb_all = model.encode(
        all_label_txt,
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=256 if device == 'cuda' else 32
    )

    cos_all = avg_emb @ lbl_emb_all.T

    out = predict_with_fallback(
        proba,
        cos_all,
        prob_cut,
        margin_cut,
        tau,
        tau_gray,
        trained_labels,
        all_labels
    )

    sub['pred_source'] = out['route']
    sub['Predicted_Risks_new'] = out['final_names']

    risk_to_idx = {label: i for i, label in enumerate(all_labels)}
    predicted_labels = sub['Predicted_Risks_new'].tolist()

    chosen_label_sim = []
    best_alt_sim = []
    semantic_margin = []

    for i, lbl in enumerate(predicted_labels):
        idx = risk_to_idx.get(lbl, None)

        if idx is None or lbl == "No Risk":
            chosen = 0.0
            alt = float(cos_all[i].max())
        else:
            chosen = float(cos_all[i, idx])
            tmp = cos_all[i].copy()
            tmp[idx] = -1
            alt = float(tmp.max())

        chosen_label_sim.append(chosen)
        best_alt_sim.append(alt)
        semantic_margin.append(chosen - alt)

    sub['chosen_label_sim'] = chosen_label_sim
    sub['best_alt_sim'] = best_alt_sim
    sub['semantic_margin'] = semantic_margin

    cos_suggestion = np.array(all_labels)[out['cos_all_idx']]

    review_terms = [
        "student loan", "student debt", "financial aid", "fafsa",
        "enrollment dropped", "enrollment decline", "adult enrollment",
        "department of justice", "doj", "department of education",
        "civil rights investigation", "title ix", "title vi",
        "in-state tuition", "undocumented students",
        "artificial intelligence", "ai policy", "ai literacy",
        "cyberattack", "data breach", "vendor", "third-party",
        "strike", "union", "collective bargaining",
        "mental health", "student wellness",
        "academic freedom", "student speech", "free speech",
        "infectious disease", "disease outbreak", "outbreak",
        "epidemic", "pandemic", "quarantine", "cdc",
        "virus"
    ]

    norisk_review = (
        (out['route'] == 'norisk') &
        sub['Text'].apply(lambda x: has_any(x, review_terms))
    )

    semantic_review = (
        (sub['Predicted_Risks_new'] != 'No Risk') &
        (
            (sub['chosen_label_sim'] < 0.45) |
            (sub['semantic_margin'] < 0.03)
        )
    )

    disagreement_review = (
        (sub['Predicted_Risks_new'] != cos_suggestion) &
        (out['lr_top_prob'] < 0.90)
    )

    gray_mask = (
        (out['route'] == 'gray') |
        norisk_review |
        semantic_review |
        disagreement_review
    )

    if gray_mask.any():
        gray_idx = sub.index[gray_mask]
        sub_pos = {idx: pos for pos, idx in enumerate(sub.index)}

        adjudicated = []

        for row_idx in gray_idx:
            pos = sub_pos[row_idx]
            txt = sub.loc[row_idx, 'Text']
            current_label = sub.loc[row_idx, 'Predicted_Risks_new']

            top3_idx = np.argsort(-cos_all[pos])[:3]
            candidate_labels = dedupe_keep_order(
                [current_label] +
                [all_labels[j] for j in top3_idx] +
                ['No Risk']
            )

            candidate_labels = [
                lbl for lbl in candidate_labels
                if lbl in all_labels or lbl == "No Risk"
            ]

            label_list = json.dumps(candidate_labels)

            prompt = f"""
You are labeling institutional risk articles for a U.S. higher-education risk dashboard.

Return ONLY valid JSON:
{{"label": "...", "confidence": 0.0, "evidence": "...", "reason": "..."}}

Choose exactly one label from this closed list:
{label_list}

Article:
{txt[:3000]}

Rules:
- Choose a risk label only if the article contains explicit evidence of that institutional risk.
- If the article is not clearly about a U.S. higher-education institutional risk, return "No Risk".
- Do NOT infer risk from generic words like "students", "campus", "college", or "university".
- If the article is merely about a university, student, research, hospital, policy, AI, or government topic without a clear institutional risk mechanism, return "No Risk".
- Use "Student Conduct Incident" ONLY for hazing, Greek life misconduct, student disciplinary violations, student fights, alcohol/drug violations, sexual misconduct, or Title IX student conduct complaints.
- Layoffs, budget deficits, program cuts, closures, buyouts, state funding reductions, debt, or enrollment losses are NEVER "Student Conduct Incident".
- AI tools, AI literacy, AI policy, AI governance, ChatGPT, algorithmic systems, or AI in teaching/support services -> "Artificial Intelligence Ethics & Governance".
- Student loans, FAFSA, financial aid, graduate loan caps, repayment rules, or borrowing limits -> "Enrollment Pressure".
- Federal research funding cuts, grant freezes, NSF/NIH disruption, indirect cost caps, terminated grants -> "Research Funding Disruption".
- Positive grant awards, research collaboration launches, or new research centers -> "No Risk" unless there is a clear compliance, funding, or governance threat.
- Strikes, unions, bargaining, worker actions, contract negotiations -> "Labor Dispute".
- Academic freedom, tenure, shared governance, faculty senate restrictions, censorship, or speech suppression -> "Faculty conflict" or "Policy or Political Interference".
- If the article describes a federal lawsuit, Department of Justice action, Department of Education action, OCR investigation, Title VI/Title IX enforcement, immigration/student visa rule, or state/federal higher-ed law, prefer "Policy or Political Interference" over "Enrollment Pressure" unless the article is mainly about student demand, affordability, admissions volume, or institutional enrollment declines.
- Protests, activist arrests, political speech disputes, immigration enforcement, or federal/state education policy actions are NOT "Student Conduct Incident".
- "Violence or Threats" requires physical danger: weapons, shooting, bomb, assault, active shooter, credible violent threat, or lockdown for safety.
- If confidence is below 0.70, return "No Risk".
- The "evidence" field must quote or paraphrase the specific article evidence supporting the label.

Mandatory Tulane-only labels:
Only use these labels if Tulane University or Tulane leadership is explicitly mentioned:
"High-Profile Litigation", "Emergency Preparedness Gaps", "Unexpected Expenditures",
"Leadership Missteps", "Revenue Loss", "Institutional Alignment Risk",
"Controversial Public Incident".
"""

            max_tries = 6
            last_err = None
            resp = None

            for attempt in range(1, max_tries + 1):
                try:
                    resp = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[prompt]
                    )
                    break

                except ClientError as e:
                    msg = str(e).lower()
                    if ("resource exhausted" in msg) or ("quota" in msg) or ("429" in msg):
                        s = str(e)
                        m = (
                            re.search(r"retryDelay'\s*:\s*'(\d+)s'", s) or
                            re.search(r"retryDelay\s*[:=]\s*'?(\d+)s'?", s, flags=re.I)
                        )
                        retry_delay = int(m.group(1)) if m else None

                        if retry_delay is None:
                            retry_delay = min(120, (2 ** (attempt - 1))) + random.uniform(0, 1.5)

                        print(
                            f"Gemini quota/rate limit (attempt {attempt}/{max_tries}). "
                            f"Sleeping {retry_delay:.1f}s...",
                            flush=True
                        )
                        time.sleep(retry_delay)
                        last_err = e
                        continue

                    raise

                except (ServerError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    wait = min(120, (2 ** (attempt - 1))) + random.uniform(0, 1.5)
                    print(
                        f"⚠️ Gemini transient error (attempt {attempt}/{max_tries}): {e} | "
                        f"Sleeping {wait:.1f}s...",
                        flush=True
                    )
                    time.sleep(wait)
                    last_err = e
                    continue

                except Exception as e:
                    wait = min(30, (2 ** (attempt - 1))) + random.uniform(0, 1.0)
                    print(
                        f"⚠️ Gemini unexpected error (attempt {attempt}/{max_tries}): {e} | "
                        f"Sleeping {wait:.1f}s...",
                        flush=True
                    )
                    time.sleep(wait)
                    last_err = e
                    continue

            else:
                raise RuntimeError(
                    f"Gemini failed after {max_tries} attempts. Last error: {last_err}"
                )

            raw = getattr(resp, "text", "").strip()
            m = re.search(r"\{.*\}", raw, flags=re.S)

            try:
                obj = json.loads(m.group(0) if m else raw)
                label = obj.get("label", "No Risk")
                conf = float(obj.get("confidence", 0.0) or 0.0)
                evidence = str(obj.get("evidence", "") or "").strip()

                if conf < 0.70:
                    label = "No Risk"

                if len(evidence) < 10 and label != "No Risk":
                    label = "No Risk"

                if label not in (all_labels + ["No Risk"]):
                    label = "No Risk"

            except Exception:
                label = "No Risk"

            adjudicated.append(label)

        sub.loc[gray_idx, 'Predicted_Risks_new'] = adjudicated
        sub.loc[gray_idx, 'pred_source'] = 'gemini'

    final_labels = []
    final_sources = []

    for txt, lbl, src in zip(
        sub['Text'].tolist(),
        sub['Predicted_Risks_new'].tolist(),
        sub['pred_source'].tolist()
    ):
        fixed = rule_route(txt, lbl)

        if fixed != lbl:
            final_labels.append(fixed)
            final_sources.append("rule_final_cleanup")
        else:
            final_labels.append(lbl)
            final_sources.append(src)

    sub['Predicted_Risks_new'] = final_labels
    sub['pred_source'] = final_sources

    sub['Pred_LR_label'] = out['lr_top_prob']
    sub['Pred_LR_margin'] = out['lr_margin']
    sub['Pred_cos_label_all'] = np.array(all_labels)[out['cos_all_idx']]
    sub['Pred_cos_score_all'] = out['cos_all_max']

    for col in [
        'pred_source',
        'Predicted_Risks_new',
        'Pred_LR_label',
        'Pred_LR_margin',
        'Pred_cos_label_all',
        'Pred_cos_score_all',
        'chosen_label_sim',
        'best_alt_sim',
        'semantic_margin'
    ]:
        df.loc[sub.index, col] = sub[col]

    return df