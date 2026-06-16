from pathlib import Path
import pandas as pd
from google.cloud import storage

from config.settings import GCS_BUCKET_NAME

def get_bucket(bucket_name: str = GCS_BUCKET_NAME):
    client = storage.Client()
    return client.bucket(bucket_name)

def upload_file(local_path: str, blob_path: str) -> None:
    bucket = get_bucket()
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)

def download_file(blob_path: str, local_path: str) -> None:
    bucket = get_bucket()
    blob = bucket.blob(blob_path)

    Path(local_path).parent.mkdir(parents = True, exist_ok=True)
    blob.download_to_filename(local_path)
    compression = "gzip" if str(local_path).endswith('.gz') else None
    return pd.read_csv(local_path, compression = compression, low_memory = False)

def download_blob_to_file(blob_path: str, local_path: str, bucket_name: str = GCS_BUCKET_NAME) -> None:
    bucket = get_bucket()
    blob = bucket.blob(blob_path)

    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(local_path)

def blob_exists(blob_path: str) -> bool:
    bucket = get_bucket()
    blob = bucket.blob(blob_path)
    return blob.exists()

def upload_bytes(data: bytes, blob_path:str, content_type:str = "application/octet-stream", bucket_name:str = GCS_BUCKET_NAME) -> None:
    bucket = get_bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_string(data, content_type=content_type)
