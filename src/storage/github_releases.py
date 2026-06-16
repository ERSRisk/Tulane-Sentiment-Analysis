import os
import requests
from pathlib import Path
import joblib

def gh_headers():
    token = os.getenv('TOKEN')
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
def get_release_by_tag(owner, repo, tag):
    url = f'https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}'
    r = requests.get(url, headers = gh_headers())
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json()

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
def load_model_bundle(owner, repo, tag, asset_name = 'model_bundle.pkl', local_cache_path = 'pipeline/resources/artifacts/model_bundle.pkl'):
    P = Path(local_cache_path)
    P.parent.mkdir(parents = True, exist_ok=True)

    if P.exists() and P.stat().st_size > 0:
        return joblib.load(P)
    rel = get_release_by_tag(owner, repo, tag)
    if not rel:
        raise RuntimeError(f"Release tag {tag} not found")
    assets = rel.get('assets', []) or []
    asset = next((a for a in assets if a.get('name')==asset_name), None)
    if not asset:
        raise RuntimeError(f"Asset {asset_name} not found")

    asset_url =asset['url']
    headers = gh_headers()
    headers['Accept'] = 'application/octet-stream'
    r = requests.get(asset_url, headers = headers, timeout = 120)
    r.raise_for_status()
    P.write_bytes(r.content)
    return joblib.load(P)