import json
import time
import pandas as pd
import re
from datetime import datetime
import numpy as np
from urllib.parse import urlparse


def risk_weights(df):
    t0 = time.perf_counter()
    print(f"[risk_weights] start: df = {df.shape}", flush = True)

    # ---------- Load config ----------
    with open('pipeline/resources/risks.json', 'r', encoding='utf-8') as f:
        risks_cfg = json.load(f)

    json_all_labels = [r['name'] for block in risks_cfg.get('new_risks', []) for _, items in block.items() for r in items]

    print("risk labels loaded", flush = True)
    accuracy_map = {}
    for s in risks_cfg.get('sources', []):
        name = str(s.get('name', '') or '')
        acc = s.get('accuracy', 0) or 0
        accuracy_map[name] = acc
    level_name_map = {"low":1, "medium low":2, "medium":3, "medium high":4, "high":5}
    risks_map = {}
    for r in risks_cfg.get('risks', []):
        nm = (r.get('name') or '').strip().lower()
        lvl = r.get('level', 0)
        if isinstance(lvl, str):
            lvl = level_name_map.get(lvl.strip().lower(), 0)
        try:
            lvl = float(lvl)
        except Exception:
            lvl = 0.0
        if nm:
            risks_map[nm] = lvl

    higher_ed_dict = risks_cfg.get('HigherEdRisks', None)

    base = df
    for col in ['Title','Content','Source']:
        if col not in base.columns:
            base[col] = ''
    base['Title'] = base['Title'].fillna('').astype(str)
    base['Content'] = base['Content'].fillna('').astype(str)
    base['Source'] = base['Source'].fillna('').astype(str)
    def _coerce_pub(x): 
        if pd.isna(x): 
            return pd.NaT 
        if isinstance(x, (int, float)): 
            if x > 1e12: # epoch ms 
                return pd.to_datetime(x, unit='ms', errors='coerce', utc=True) 
            if x > 1e9: # epoch s 
                return pd.to_datetime(x, unit='s', errors='coerce', utc=True) 
        sx = str(x) 
        sx = re.sub(r'\s(EST|EDT|PDT|CDT|MDT|GMT)\b', '', sx, flags=re.I) 
        return pd.to_datetime(sx, errors='coerce', utc=True) 
    if 'Published' not in base.columns: 
        base['Published'] = pd.NaT 
    base['Published'] = base['Published'].apply(_coerce_pub) 
    if pd.api.types.is_datetime64tz_dtype(base['Published']): 
        base['Published'] = base['Published'].dt.tz_convert('UTC').dt.tz_localize(None) 

    now_naive = datetime.utcnow() 
    base['Days_Ago'] = (now_naive - base['Published']).dt.days 
    base['Days_Ago'] = base['Days_Ago'].fillna(10_000).astype(int) 
    risk_half_life = { 
        "Research Funding Disruption": 60, 
        "Enrollment Pressure": 60, 
        "Policy or Political Interference": 90, 
        "Institutional Alignment Risk": 60, 
        "Mission Drift": 90, 
        "Revenue Loss": 90, 
        "Insurance Market Volatility": 90, 
        "Unexpected Expenditures": 15, 
        "Endowment Risk": 30, 
        "Constant Inflation": 15, 
        "Infrastructure Failure": 15, 
        "Transportation/Access Disruption": 7, 
        "Supply Chain Delay": 15, 
        "Emergency Preparedness Gaps": 15, 
        "Title IX/ADA Noncompliance": 30, 
        "Accreditation Risk": 120, 
        "FERPA/HIPAA Violations": 7, 
        "Grant Mismanagement": 7, 
        "Audit Findings": 30, 
        "Unauthorized Access/Data Breach": 7, 
        "Credential Phishing": 7, 
        "Vendor Cyber Exposure": 7, 
        "Cloud Misconfiguration": 7, 
        "Artificial Intelligence Ethics & Governance": 7, 
        "Rapid Speed of Disruptive Innovation": 90, 
        # --- Reputational and Social --- 
        "Controversial Public Incident": 30, 
        "DEI Program Backlash": 30, 
        "High-Profile Litigation": 90, 
        "Leadership Missteps": 30, 
        "Media Campaigns": 15, 
        # --- Health, Safety and Security --- 
        "Violence or Threats": 10, 
        "Infectious Disease Outbreak": 30, 
        "Lab Incident": 7, 
        "Workplace Safety Violation": 7, 
        "Environmental Exposure": 30, 
        # --- Environmental & Climate --- 
        "Hurricane/Flood/Wildfire": 30, 
        "Extreme Weather Events": 30, 
        "Climate Infrastructure Risks": 15, 
        "Environmental Noncompliance": 90, 
        "Insurance Withdrawal": 120, 
        # --- Student Experience & Welfare --- 
        "Mental Health Crises": 15, 
        "Housing/Food Insecurity": 15, 
        "Academic Disruption": 15, 
        "Student Conduct Incident": 7, 
        "Accessibility Barriers": 15, 
        # --- Internal Organization --- 
        "HR Complaint": 15, 
        "Labor Dispute": 30, 
        "Morale challenges": 30, 
        "Faculty conflict": 15, 
        "Executive Board conflicts": 30, 
        "Nepotism/Conflict of Interest": 15, 
        "Policy Misapplication": 15, 
        "Whistleblower Claims": 30 } 
    cand = base.get('Predicted_Risks_new', pd.Series('', index=base.index)).fillna('')
    cand = np.where(cand=='', base.get('Predicted_Risks', ''), cand)
    cand = pd.Series(cand, index=base.index).fillna('').astype(str)
    
    def _first_label(s):
        s = s.strip()
        if not s:
            return ''
        s = re.sub(r'^\[|\]$', '', s)
        s = re.split(r'[;,]', s)[0].strip().strip("'\"")
        return s
    
    base['_RiskList'] = cand.apply(_first_label)
    base['Risk_item'] = np.where(base['_RiskList'].eq(''), 'No Risk', base['_RiskList'])

    if 'Topic' not in base.columns:
        base['Topic'] = -1
    base['Topic'] = base['Topic'].fillna(-1)
    def recency_features_topic_risk(df, now=None):
        fx = df[['Topic', '_RiskList', 'Published', 'Days_Ago']].copy()

        required = {'Topic', '_RiskList', 'Published', 'Days_Ago'}
        if not required.issubset(fx.columns) or fx.empty:
            return pd.DataFrame(columns=['Topic','_RiskList','last_seen_days','decayed_volume','recency_score_tr'])

        if now is None:
            now = pd.Timestamp.utcnow()

        art_w = 1.0
        if 'Impact_Score' in base.columns:
            art_w = pd.to_numeric(base['Impact_Score'], errors='coerce').fillna(0.0).clip(0, 1)

        def half_life(risk):
            return risk_half_life.get(risk, 30)

        hl  = fx['_RiskList'].map(lambda r: max(1.0, half_life(r)))
        lam = np.log(2.0) / hl
        w_decay = np.exp(-lam * fx['Days_Ago'])
        fx['_w'] = w_decay * art_w

        grp = fx.groupby(['Topic', '_RiskList'], dropna=False)
        out = grp.agg(
            last_seen=('Days_Ago', 'min'),
            decayed_volume=('_w', 'sum'),
            mentions=('Published', 'count')
        ).reset_index()

        out['hl'] = out['_RiskList'].map(lambda r: max(1.0, half_life(r)))
        out['freshness'] = np.exp(-np.log(2.0) * (out['last_seen'] / out['hl']))

        def _safe_minmax(s):
            rng = s.max() - s.min()
            return (s - s.min()) / (rng + 1e-12)

        out['decayed_z'] = out.groupby('_RiskList')['decayed_volume'].transform(_safe_minmax)

        w_fresh, w_vol = 0.6, 0.4
        out['recency_score_tr'] = (w_fresh * out['freshness'] + w_vol * out['decayed_z']).clip(0, 1)
        out = out.rename(columns={'last_seen': 'last_seen_days'})
        return out[['Topic','_RiskList','last_seen_days','decayed_volume','recency_score_tr']]


    def attach_topic_risk_recency(df):
        tr = recency_features_topic_risk(df)
        for c in ['last_seen_days','decayed_volume','recency_score_tr']:
            if c not in tr.columns:
                tr[c] = np.nan
        cols_to_drop = ["last_seen_days","decayed_volume",
                    "recency_score_tr","recency_score_tr_x","recency_score_tr_y"]
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

        tr_small = tr[["Topic", "_RiskList", "last_seen_days", "decayed_volume", "recency_score_tr"]].rename(
            columns={"recency_score_tr": "recency_score_tr_tr"}
        )
        overlap = [c for c in tr_small.columns if c in df.columns and c not in ['Topic', '_RiskList']]
        if overlap:
            df = df.drop(columns = overlap)
        enriched = df.merge(tr_small, on=["Topic","_RiskList"], how="left", validate = "m:1")

        days = pd.to_numeric(enriched.get('Days_Ago', np.nan), errors='coerce').astype(float)
        enriched['article_freshness'] = np.exp(-np.log(2.0) * (days / 14.0)).fillna(0.0)

        if 'recency_score_tr_tr' not in enriched.columns:
            enriched['recency_score_tr_tr'] = 0.0

        alpha = 0.7
        enriched['Recency_TR_Blended'] = (
            alpha * enriched['recency_score_tr_tr'].fillna(0.0)
            + (1 - alpha) * enriched['article_freshness']
        ).clip(0, 1)

        return enriched
    

    def _src_acc(row):
        src = str(row.get('Source','') or '')
        link = str(row.get('Link', '') or '')
        src_l = src.lower()
        dom = ''
        if link:
            try:
                dom = urlparse(link).netloc.lower()
            except Exception:
                dom = ''
        if dom.endswith('.gov') or dom.endswith('.mil') or dom.endswith('tulane.edu'):
            return 5
        best = 0.0
        for name, acc in accuracy_map.items():
            if name and name.lower() in src_l:
                try:
                    v = float(acc)
                except Exception:
                    v = 0.0
                best = max(best, v)

        return min(best, 4.0)
    base['Source_Accuracy'] = base.apply(_src_acc, axis=1)
    print('Source accuracy created', flush = True)

    def _loc_score(row):
        us_sources = ['foxnews','NIH', 'NOAA', 'FEMA', 'NASA', 'CISA', 'NIST', 'NCES', 'CMS', 'CDC', 'BEA', 'The Advocate', 'LA Illuminator', 'The Hill', 'NBC News', 'PBS', 'StatNews', 'NY Times', 'Washington Post', 'TruthOut', 'Politico', 'Inside Higher Ed', 'CNN', 'Yahoo News', 'FOX News', 'ABC News', 'Huffington Post', 'Business Insider', 'Bloomberg', 'AP News']
        raw = row.get('Entities', None)

        if isinstance(raw, list):
            entities = [str(e).lower() for e in raw if e is not None]
        elif isinstance(raw, str) and raw.strip():
            entities = [raw.strip().lower()]
        else:
            entities = []
        text = (row.get('Title','') + ' ' + row.get('Content','')).lower()
        def has_any(keys): 
            lk = [k.lower() for k in keys]
            return any(k in text for k in lk)
        if has_any(['tulane','tulane university']): return 5
        if has_any(['new orleans','louisiana','nola']): return 4
        if has_any(['baton rouge','governor landry','lafayette','lsu','university of louisiana']): return 3
        if has_any(['gulf coast','mississippi','texas','alabama']): return 2
        if has_any(['u.s.','united states','america','federal','washington dc','trump']) or str(row.get('Source','')) in us_sources: return 1
        return 0

    base['Location'] = base.apply(_loc_score, axis=1)
    base['Location'] = pd.to_numeric(base['Location'], errors = 'coerce').fillna(0).astype(int)
    print('Location created', flush = True)



    def _window_tag(d):
        if d <= 30: return 'recent'
        if d <= 60: return 'previous'
        return 'older'
    tmp = base[['Topic','Days_Ago']].copy()
    tmp['Time_Window'] = tmp['Days_Ago'].apply(_window_tag)
    topic_counts = tmp.groupby(['Topic','Time_Window']).size().unstack(fill_value=0)
    for c in ['recent','previous']:
        if c not in topic_counts.columns:
            topic_counts[c] = 0

    periods = base['Published'].dt.to_period('W-MON')
    base['Week'] = periods.dt.to_timestamp(how = 'start')
    ts = (
        base.loc[base['Week'].notna()]
        .groupby(['Risk_item','Week'])
        .size().rename('n').reset_index()
        .sort_values(['Risk_item','Week'])
    )
    if not ts.empty:
        ts['EMWA'] = ts.groupby('Risk_item')['n'].transform(
            lambda s: s.ewm(span = 4, adjust =False).mean()
        )

        ts['EMWA_Delta'] = ts.groupby('Risk_item')['EMWA'].diff().fillna(0.0)
        def slope(counts, k=6):
            x = np.arange(len(counts), dtype=float)
            out = np.zeros(len(counts), dtype=float)
            for i in range(len(counts)):
                lo = max(0, i - min(k, i) + 1) 
                xi = x[lo:i+1]; yi = counts[lo:i+1].astype(float)
                if len(xi) >= 2: 
                    m, _ = np.polyfit(xi, yi, 1)
                    out[i] = m
            return out

        ts['Slope'] = ts.groupby('Risk_item', group_keys = False)['n'].apply(lambda g: pd.Series(slope(g.values, k=6), index = g.index)).astype(float)

        def normalize_groupwise(s, by):
            return s.groupby(by, group_keys=False).rank(pct = True).fillna(0.0)

        ts['emwa_norm']  = normalize_groupwise(ts['EMWA'].clip(lower=0),  ts['Risk_item'])
        ts['slope_norm'] = normalize_groupwise(ts['Slope'].clip(lower = 0), ts['Risk_item'])

        w_emwa, w_slope = 0.6, 0.4
        ts['accel_score'] = (w_emwa*ts['emwa_norm'] + w_slope * ts['slope_norm']).clip(0,1)
        weeks_seen = ts.groupby('Risk_item')['Week'].transform('nunique')
        ts.loc[weeks_seen < 4, 'accel_score'] *= 0.6


        #if 'Sentiment Score' not in base.columns:
        #    base['Sentiment Score'] = 0.0
        #ts_sent = (
        #    base.loc[base['Week'].notna()]
        #    .groupby(['Risk_item','Week'])
        #    .agg(sent_mean=('Sentiment Score','mean'))
        #    .reset_index()
        #    .sort_values(['Risk_item','Week'])
        #)
        #ts_sent['sent_flipped'] = -ts_sent['sent_mean']
        #ts_sent['sent_ewma'] = ts_sent.groupby('Risk_item')['sent_flipped'].transform(
        #    lambda s: s.ewm(span=4, adjust=False).mean()
        #)
        #ts_sent['sent_delta'] = ts_sent.groupby('Risk_item')['sent_ewma'].diff().fillna(0.0)
        #ts_sent['sent_slope'] = ts_sent.groupby('Risk_item', group_keys=False)['sent_flipped'] \
        #    .apply(lambda g: pd.Series(slope(g.values, k=6), index=g.index)) \
        #    .astype(float)
        #ts_sent['sent_delta_norm'] = normalize_groupwise(ts_sent['sent_delta'], ts_sent['Risk_item'])
        #ts_sent['sent_slope_norm'] = normalize_groupwise(ts_sent['sent_slope'], ts_sent['Risk_item'])
        #w_sent_delta, w_sent_slope = 0.6, 0.4
        #ts_sent['accel_score_sent'] = (w_sent_delta*ts_sent['sent_delta_norm'] + w_sent_slope*ts_sent['sent_slope_norm']).clip(0,1)

        ts['accel_score'] = ts['accel_score'].fillna(0.0)



        def _acc_value(d):
            if d < 0.1: return 0
            if d < 0.25: return 1
            if d < 0.40: return 2
            if d < 0.60: return 3
            if d < 0.80: return 4
            return 5
        ts['Acceleration_value'] = ts['accel_score'].apply(_acc_value).astype(int)

        def cue_eta_from_text(t):
            t = str(t).lower()
            if re.search(r'\b(today|tonight|tomorrow|immediately|right now)\b', t):
                return 3
            if re.search(r'\b(this week|in\s*the\s*coming\s*days)\b', t):
                return 7
            # near-term windows
            if re.search(r'\b(next week|within\s*2\s*weeks|in\s*2\s*weeks)\b', t):
                return 14
            if re.search(r'\b(within\s*30\s*days|this month|in\s*\d+\s*days)\b', t):
                return 30
            return 45

        temp = base.loc[base['Week'].notna(), ['Risk_item', 'Week', 'Title','Content', 'Location','University Label']].copy()
        temp['cue_eta'] = (temp['Title'] + ' ' + temp['Content'].fillna('')).apply(cue_eta_from_text)

        def location_eta(l):
            try:
                loc = int(l)
            except Exception:
                loc = 0
            if loc == 5: return 7
            if loc == 1: return 21
            return 45
        temp['loc_eta'] = temp['Location'].apply(location_eta)
        def tulane_pull(u_label, eta):
            try:
                label = int(u_label)
            except Exception:
                label = 0
            return max(1, eta - (7 if label ==1 else 0))

        ulab = pd.to_numeric(temp['University Label'], errors='coerce').fillna(0).astype('int8')
        eta_min = np.minimum(temp['cue_eta'].to_numpy(), temp['loc_eta'].to_numpy())
        temp['eta_article'] = np.maximum(1, eta_min - np.where(ulab.to_numpy() == 1, 7, 0))

        eta_by_bucket = (
            temp.groupby(['Risk_item', 'Week'])['eta_article'].min()
            .rename('eta_days_proxy').reset_index()
        )

        ts = ts.merge(eta_by_bucket, on = ['Risk_item', 'Week'], how='left')
        ts['eta_days_proxy'] = ts['eta_days_proxy'].fillna(45).astype(float)

        ts['eta_days_proxy'] = (ts['eta_days_proxy']- (ts['accel_score'] * 5.0)).clip(lower =1)

        def cap_by_eta(val, eta_days):
            if eta_days <= 15:
                return(min(int(val), 5))
            elif eta_days <= 30:
                return (min(int(val),4))
            else: 
                return min(int(val), 3)

        ts['Acceleration_value'] = ts.apply(
            lambda r: cap_by_eta(r['Acceleration_value'], r['eta_days_proxy']),
            axis=1
        ).astype(int)

    base = base.drop(columns = ['Acceleration_value_x', 'Acceleration_value_y'], errors = 'ignore')
    right = ts[['Risk_item', 'Week', 'Acceleration_value']].rename(columns = {'Acceleration_value': 'Acceleration_value_new'})
    base = base.merge(right, on = ['Risk_item', 'Week'], how = 'left', validate = 'm:m')
    if 'Acceleration_value' in base.columns:
        base['Acceleration_value'] = base['Acceleration_value_new'].fillna(base['Acceleration_value'])
    else:
        base['Acceleration_value'] = base['Acceleration_value_new']


    base = base.drop(columns = ['Acceleration_value_new'])
    base['Acceleration_value'] =base['Acceleration_value'].fillna(0).astype(int)
    print('Acceleration value created', flush = True)


    if base.empty:
        base['Frequency_Score'] = 0
    else:
        counts = base['Risk_item'].value_counts().rename_axis('Risk_item').reset_index(name='Count')
        try:
            bins = pd.qcut(counts['Count'].rank(method='first'), 5, labels=[1,2,3,4,5])
            counts['Frequency_Score'] = bins.astype(int)
        except Exception:
            mn, mx = counts['Count'].min(), counts['Count'].max()
            if mx == mn:
                counts['Frequency_Score'] = 3
            else:
                scaled = 1 + 4 * (counts['Count'] - mn) / float(mx - mn)
                counts['Frequency_Score'] = scaled.round().clip(1,5).astype(int)
        freq_map = dict(zip(counts['Risk_item'], counts['Frequency_Score']))
        base['Frequency_Score'] = base['Risk_item'].map(freq_map).fillna(0).astype(int)

    hed = risks_cfg.get('HigherEdRisks') or {}
    hed_norm = {
        str(cat).strip().lower(): [str(p).strip().lower() for p in (phr_list or []) if str(p).strip()]
        for cat, phr_list in hed.items()
    }

    def phrase_to_pattern(phrase: str) -> str:
        tokens = re.split(r"\s+", phrase.strip())
        # allow any non-word chars between tokens; keep word boundaries at ends
        return r"(?<!\w)" + r"[\W_]+".join(map(re.escape, tokens)) + r"(?!\w)"

    cat_regex = {}
    for cat, phrases in hed_norm.items():
        if not phrases:
            continue
        pats = [phrase_to_pattern(p) for p in phrases]
        pats.append(phrase_to_pattern(cat))
        cat_regex[cat] = re.compile("(" + "|".join(pats) + ")", flags=re.I)
    
    text_all = (base['Title'].fillna('') + ' ' + base['Content'].fillna('')).astype(str)

    has_highered_category = pd.Series(False, index=base.index)
    for cat, rx in cat_regex.items():
        has_highered_category |= text_all.str.contains(rx, na=False)
    
    ul = pd.to_numeric(base.get('University Label', 0), errors='coerce').fillna(0).astype('int8')
    base['Industry_Risk_Presence'] = np.where(has_highered_category | (ul == 1), 3, 0).astype('int8')


    peers_list = risks_cfg.get('Peer_Institutions') or []
    peer_pat = re.compile(r'\b(' + '|'.join([re.escape(p) for p in peers_list]) + r')\b', flags=re.I) if peers_list else None

    moderate_impact = re.compile(r'\b(outage|closure|lawsuit|probation|sanction|breach|evacuation|investigation)\b', re.I)
    substantial_impact = re.compile(r'\b(widespread|catastrophic|shutdown|bankrupt|insolvenc\w*|fatalit\w*|revocation|accreditation\s+revoked)\b', re.I)
    _text_all = (base['Title'].fillna('') + ' ' + base['Content'].fillna('')).astype(str)
    base['_tulane_flag'] = (base.get('Location', 0).astype(int).eq(5)) | _text_all.str.contains(r'\btulane\b', case=False, regex=True)

    def find_peer(t):
        if not peer_pat:
            return ''
        m = peer_pat.search(t or '')
        return m.group(0) if m else ''

    def severity(text):
        if substantial_impact.search(text or ''):
            return 'substantial'
        if moderate_impact.search(text or ''):
            return 'moderate'
        return ''

    tmp_ind = base.loc[base['Week'].notna(), ['Week', 'Title', 'Content']].copy()
    tmp_ind['text_all'] = (tmp_ind['Title'] + ' ' + tmp_ind['Content']).fillna('')
    tmp_ind['peer'] = tmp_ind['text_all'].apply(find_peer)
    tmp_ind['sev'] = tmp_ind['text_all'].apply(severity)

    # Count unique peers by severity per week
    agg = (
        tmp_ind.groupby(['Week', 'sev'])['peer']
        .nunique()
        .unstack(fill_value=0)
        .rename(columns={'moderate': 'peers_mod', 'substantial': 'peers_sub'})
    )
    for c in ['peers_mod', 'peers_sub']:
        if c not in agg.columns:
            agg[c] = 0
    agg = agg.reset_index()

    
    tulane_week = (
        base.loc[base['_tulane_flag'] & base['Week'].notna()]
        .groupby('Week')
        .size()
        .rename('tulane_mentions')
        .reset_index()
    )
    agg = agg.merge(tulane_week, on='Week', how='left').fillna({'tulane_mentions': 0})


    if not agg.empty:
        week_max = agg['Week'].max()
        agg['days_ago'] = (week_max - agg['Week']).dt.days.clip(lower=0)
        lam = np.log(2.0) / 21.0
        agg['decay_w'] = np.exp(-lam * agg['days_ago'])

        
        agg['peer_index'] = agg['decay_w'] * (2 * agg['peers_sub'] + 1 * agg['peers_mod'])
        agg['sector_pressure'] = agg['peer_index'] / (1.0 + agg['tulane_mentions'])

        
        lo, hi = np.percentile(agg['sector_pressure'], [5, 95]) if agg['sector_pressure'].notna().any() else (0.0, 1.0)
        rng = max(1e-12, hi - lo)
        agg['Industry_Risk_Peer'] = (((agg['sector_pressure'] - lo) / rng).clip(0, 1) * 5).round().astype(int)
    else:
        agg['Industry_Risk_Peer'] = 0


    base = base.drop(columns=['Industry_Risk_Peer'], errors='ignore')
    base = base.merge(agg[['Week', 'Industry_Risk_Peer']], on='Week', how='left')
    base['Industry_Risk_Peer'] = base['Industry_Risk_Peer'].fillna(0).astype(int)


    base['Industry_Risk'] = np.maximum(base['Industry_Risk_Presence'], base['Industry_Risk_Peer']).astype(int)
    print('Industry risk created', flush = True)



    impact_weights = {"financial": 0.35, "reputational": 0.15, "academic": 0.25, "operational": 0.25}
    new_risks = risks_cfg.get('new_risks')
    risk_dims_map = {}
    for block in new_risks:
        for category, items in block.items():
            for r in items:
                name = re.sub(r'\s+', ' ', r.get('name', '')).strip().lower()
                dims = r.get('impact dims')
                risk_dims_map[name] = {
                    'financial': float(dims.get('financial', 0.0)),
                    'reputational': float(dims.get('reputational', 0.0)),
                    'academic': float(dims.get('academic', 0.0)),
                    'operational': float(dims.get('operational', 0.0))
                }

    existential_threat_patterns = [
    r'\b(permanent\s+closure|cease\s+operations|bankrupt|insolven|shut\s*down\s*permanent|revocation\s+of\s+accreditation)\b',
    r'\b(catastrophic\s+(damage|failure)|total\s+loss|existential\s+threat)\b'
]
    severe_patterns = [
    r'\b(university[-\s]*wide|campus[-\s]*wide|enterprise[-\s]*wide|entire\s+university|all\s+systems\s+down)\b',
    r'\b(ransomware|mass\s+evacuation|classes\s+canceled\s+across\s+campus|network\s+outage)\b',
    r'\b(federal\s+investigation|systemic\s+title\s*ix|major\s+scandal|fatalit(y|ies))\b',
    r'\b(executive\s+action\b(?:\s\w+){0,100}?\s(college|university|universities))\b',
    r'\b(regulatory\s+action|regulation\s\b(?:\s\w+){0,100}\s(higher\seducation|university|universities|college|colleges))\b'
]

    def find_patterns(text, patterns):
        for p in patterns:
            if re.search(p, text, flags=re.I):
                return True
        return False

    def impact_row(row):
        risk = re.sub(r'\s+', ' ', str(row['Risk_item'])).strip().lower()
        dims = risk_dims_map.get(risk)

        if dims:
            fin, rep, acad, oper = dims['financial'], dims['reputational'], dims['academic'], dims['operational']
        else:
            fin, rep, acad, oper = 1.0, 1.0, 1.0, 1.0

        base = (fin * impact_weights['financial'] +
                rep * impact_weights['reputational'] +
                acad * impact_weights['academic'] +
                oper * impact_weights['operational'])

        text = (str(row.get('Title','')) + ' ' + str(row.get('Content',''))).lower()
        existential = find_patterns(text, existential_threat_patterns)
        severe = find_patterns(text, severe_patterns) or (int(row.get('Location',0))==5 and int(row.get('University Label',0))==1)

        if existential:
            return min(5.0, max(5.0, base))
        if severe:
            return min(4.0, max(4.0, base))
        else:
            return min(base, 3.9)
    for col in ['Location', 'University Label']:
        if col not in base.columns:
            base[col] = 0
        base[col] = pd.to_numeric(base[col], errors = 'coerce').fillna(0).astype(int)
    base['Impact_Score'] = base.apply(impact_row, axis=1).astype(float)
    print('Impact score computed', flush = True)

    t_rec = time.perf_counter()
    print("attach_topic_risk_recency() start", flush = True)
    base = attach_topic_risk_recency(base) 
    base['Recency'] = (base['Recency_TR_Blended'] * 5).round(2)
    print("[recency] attached in {time.perf_counter()-t_rec:.1f}s", flush = True)


    w = {
        'Recency': 0.15,
        'Source_Accuracy': 0.10,
        'Impact_Score': 0.35,
        'Acceleration_value': 0.25,
        'Location': 0.05,
        'Industry_Risk': 0.05,
        'Frequency_Score': 0.05
    }
    weight_sum = sum(w.values()) 

    num = (
        base['Recency'] * w['Recency'] +
        base['Source_Accuracy'] * w['Source_Accuracy'] +
        base['Impact_Score'] * w['Impact_Score'] +
        base['Acceleration_value'] * w['Acceleration_value'] +
        base['Location'] * w['Location'] +
        base['Industry_Risk'] * w['Industry_Risk'] +
        base['Frequency_Score'] * w['Frequency_Score']
    )
    base['Risk_Score'] = (num / weight_sum).clip(0,5).round(3)
    base['Weights'] = base['Risk_Score']

    print(f"[risk_weights] done: base = {base.shape} elapsed = {time.perf_counter()- t0:.1f}s", flush = True)

    return base