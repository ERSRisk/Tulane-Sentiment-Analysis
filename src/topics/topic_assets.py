import requests
import pandas as pd
from pathlib import Path
import io
import zipfile
from bertopic import BERTopic
import os

from src.storage.github_releases import (
    ensure_release,
    get_release_by_tag,
    upload_asset,
)

from src.storage.gcs import (
    download_file,
    blob_exists,
)

DIR_URL  = "https://github.com/ERSRisk/Tulane-Sentiment-Analysis/releases/download/rss_json/bertopic_dir.zip"
DIR_PATH = Path("pipeline/resources/bertopic_dir")
Github_owner = 'ERSRisk'
Github_repo = 'Tulane-Sentiment-Analysis'
Release_tag = 'BERTopic_results'
Asset_name = 'BERTopic_results2.csv_part1.csv.gz'
GITHUB_TOKEN = os.getenv('TOKEN')


def load_articles_from_release(local_cache_path='pipeline/resources/BERTopic_results2.csv.gz',
                               usecols=None, dtype=None, Asset_name = 'BERTopic_results2.csv.gz'):
    def create_github_release(owner, repo, tag, token):
        url = f"https://api.github.com/repos/{owner}/{repo}/releases"
        headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json"
        }
    
        payload = {
            "tag_name": tag,
            "name": tag,
            "draft": False,
            "prerelease": False
        }
    
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        print("Release created successfully.", flush=True)
    rel = get_release_by_tag(Github_owner, Github_repo, Release_tag)
    if not rel:
        print("Release does not exist. Creating new release...", flush=True)
        create_github_release(Github_owner, Github_repo, Release_tag, GITHUB_TOKEN)
        return pd.DataFrame()
    # 1) Try remote release asset (streamed)
    if rel:
        asset = next((a for a in rel.get('assets', []) if a['name'] == Asset_name), None)
        if asset:
            url = asset['browser_download_url']
            # Stream to disk instead of holding in RAM
            with requests.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                Path(local_cache_path).parent.mkdir(parents=True, exist_ok=True)
                with open(local_cache_path, 'wb') as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                        if chunk:
                            f.write(chunk)
            # Read the gz file directly from disk
            return pd.read_csv(local_cache_path, compression='gzip',
                               low_memory=False, usecols=usecols, dtype=dtype)

    # 2) Fallback to local cache if present
    P = Path(local_cache_path)
    if P.exists():
        return pd.read_csv(local_cache_path, compression='gzip',
                           low_memory=False, usecols=usecols, dtype=dtype)

    # 3) Nothing available
    print(f"{Asset_name} does not exist yet. Returning empty DataFrame.", flush=True)
    return pd.DataFrame()


def load_full_topics(existing_df):
    dfs = []

    try:
        if blob_exists('latest/BERTopic_results3.csv.gz'):
            r3 = download_file('latest/BERTopic_results3.csv.gz', 'pipeline/resources/BERTopic_results3.csv.gz')
        if r3 is not None and not r3.empty:
            dfs.append(r3)
    except Exception as e:
        print(f"Could not load BERTopic_results3.csv.gz: {e}", flush=True)
    if existing_df is not None and not existing_df.empty:
        dfs.append(existing_df)

    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index = True)
    out = out.drop_duplicates(subset = ['Link'], keep = 'last')
    return out

def upload_dir_model_zip(owner, repo, tag, token, dir_path=DIR_PATH, asset_name="bertopic_dir.zip"):
        # zip the directory model into memory
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            base = dir_path.name
            for root, _, files in os.walk(dir_path):
                for fn in files:
                    full = Path(root) / fn
                    rel  = Path(base) / full.relative_to(dir_path)
                    zf.write(full, arcname=str(rel))
        buf.seek(0)
        # upload via your existing GitHub helper
        rel = ensure_release(owner, repo, tag, token)
        upload_asset(owner, repo, rel, asset_name, buf.getvalue(), token, content_type="application/zip")
        print(f"✅ Uploaded {asset_name} to release {tag}.")
    
    
        
def load_dir_model():
    # load from disk if present
    if DIR_PATH.exists() and any(DIR_PATH.iterdir()):
        print("📦 Loading BERTopic from local directory model...")
        return BERTopic.load(str(DIR_PATH))
    # try Releases (zip)
    try:
        print("🌐 Fetching bertopic_dir.zip from Releases...")
        r = requests.get(DIR_URL, timeout=120)
        if r.ok and r.content[:2] == b"PK":  # zip magic
            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                zf.extractall(DIR_PATH.parent)
            print("✅ Extracted bertopic_dir.zip.")
            return BERTopic.load(str(DIR_PATH))
        else:
            print(f"⚠️ No directory model at {DIR_URL} (status {r.status_code}).")
    except Exception as e:
        print("⚠️ Could not download dir model:", e)
    return None