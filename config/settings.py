import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_CACHE_DIR = Path(os.getenv('LOCAL_CACHE_DIR', 'pipeline/resources'))

GCS_BUCKET_NAME = os.getenv('GCS_BUCKET_NAME', 'tulane-risk-data')

GITHUB_OWNER = os.getenv("GITHUB_OWNER", "ERSRisk")
GITHUB_REPO = os.getenv("GITHUB_REPO", "Tulane-Sentiment-Analysis")
GITHUB_TOKEN = os.getenv("TOKEN")

NEWS_API_KEY = os.getenv('NEWS_API_KEY')
GEMINI_API_KEY = os.getenv('PAID_API_KEY')

RISKS = os.getenv('RISKS')

GCS_LATEST_PREFIX = "latest"
