import socket
import aiohttp
import feedparser
import traceback
import spacy
import time
import subprocess
import asyncio
import json
import time
import re
import sys
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pickle
import trafilatura
import os
from playwright.sync_api import sync_playwright
import requests
import io
import gzip
import base64
import ast
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime
import fitz
from urllib.parse import urljoin
import pandas as pd
from requests.utils import requote_uri
from datetime import datetime, date, timedelta
from dateutil import parser
from newspaper import Article

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
search = 'Tulane'
start_date = (date.today() - timedelta(days = 7))
end_date = date.today()
timezone_option = 'CDT'
def fetch_news(search, start_date, end_date):
    news_url = (
            f"https://newsapi.org/v2/everything?q={search} NOT sports NOT Football NOT basketball&"
            f"from={start_date}&to={end_date}&sortBy=popularity&apiKey={NEWS_API_KEY}"
        )
    response = requests.get(news_url)
    if response.status_code == 200:
        news_data = response.json()
        return news_data.get("articles", [])  # Return the articles
    else:
        return []
    
def fetch_content_news(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return None
    


rss_feed =   {"RSS_Feeds":[{
              "WHO": ["https://www.nytimes.com/svc/collections/v1/publish/https://www.nytimes.com/topic/organization/world-health-organization/rss.xml"],
              "NIH": ["https://grants.nih.gov/news-events/nih-extramural-nexus-news/rss-feed"],
              "NOAA": ["noaa.gov/rss.xml"],
              "FEMA": ["https://www.fema.gov/feeds/news.rss", "https://www.fema.gov/feeds/disasters-major.rss", "https://www.fema.gov/feeds/disasters-fire.rss"],
              "NASA": ["https://www.nasa.gov/news-release/feed/"],
  "White House": ['https://www.whitehouse.gov/presidential-actions/feed/'],
  "CISA": ["https://www.cisa.gov/news.xml"],
  "OSHA": ["https://www.osha.gov/news/newsreleases.xml"],
  "National Institute of Standards and Technology": ["https://www.nist.gov/news-events/news/rss.xml"],
  "National Center for Education Statistics": ["https://nces.ed.gov/whatsnew/whatsnew_rss.asp"],
  "Centers for Medicare & Medicaid Services":["https://www.cms.gov/newsroom/rss-feeds"],
  "IMF": ["https://www.imf.org/en/Publications/RSS?language=eng"],
              "Bureau of Economic Analysis":["https://apps.bea.gov/rss/rss.xml?_gl=1*f107ux*_ga*OTI3ODA4ODM3LjE3NTE1NTI2MTY.*_ga_J4698JNNFT*czE3NTE1NTI2MTUkbzEkZzEkdDE3NTE1NTI2NDMkajMyJGwwJGgw"],
              "CDC":["http://wwwnc.cdc.gov/eid/rss/ahead-of-print.xml"],
              "The Advocate": ["https://www.theadvocate.com/search/?q=&t=article&l=35&d=&d1=&d2=&s=start_time&sd=desc&c%5b%20%5d=new_orleans/news*,baton_rouge/news/politics/legislature,baton_rouge/news/politics,new_orleans/opinion*,baton_rouge/opinion/stephanie_grace,baton_rouge/opinion/jeff_sadow,ba%20ton_rouge/opinion/mark_ballard,new_orleans/sports*,baton_rouge/sports/lsu&nk=%23tncen&f=rss",
                        "https://www.theadvocate.com/search/?q=&t=article&l=35&d=&d1=&d2=&s=start_time&sd=desc&c%5b%5d=new_orleans/news/business&nk=%20%23tncen&f=rss",
                        "https://www.theadvocate.com/search/?q=&t=article&l=35&d=&d1=&d2=&s=start_time&sd=desc&c%5b%5d=new_orleans/news/communities*&nk=%20%23tncen&f=rss"],
        "LA Illuminator":"https://lailluminator.com/feed/",
            "The Hill": ["https://thehill.com/homenews/senate/feed/", 
                    "https://thehill.com/homenews/house/feed/",
                    "https://thehill.com/homenews/administration/feed/",
                    "https://thehill.com/homenews/campaign/feed/",
                    "https://thehill.com/regulation/feed/",
                    "https://thehill.com/lobbying/feed/",
                    "https://thehill.com/policy/defense/feed/",
                    "https://thehill.com/policy/energy-environment/feed/",
                    "https://thehill.com/finance/feed/",
                    "https://thehill.com/policy/healthcare/feed/",
                    "https://thehill.com/policy/technology/feed/",
                    "https://thehill.com/policy/international/feed/",
                    "https://thehill.com/business/feed/",
                    "https://thehill.com/business/banking-financial-institutions/feed/",
                    "https://thehill.com/business/budget/feed/",
                    "https://thehill.com/business/taxes/feed/",
                    "https://thehill.com/business/economy/feed/",
                    "https://thehill.com/business/trade/feed/"
                    ],
            "NBC News": ["https://feeds.nbcnews.com/nbcnews/public/world",
                    "https://feeds.nbcnews.com/nbcnews/public/politics",
                    "https://feeds.nbcnews.com/nbcnews/public/science",
                    "https://feeds.nbcnews.com/nbcnews/public/health"],
            "PBS": ["https://www.pbs.org/newshour/feeds/rss/politics",
                "https://www.pbs.org/newshour/feeds/rss/nation",
                "https://www.pbs.org/newshour/feeds/rss/world",
                "https://www.pbs.org/newshour/feeds/rss/health",
                "https://www.pbs.org/newshour/feeds/rss/science",
                "https://www.pbs.org/newshour/feeds/rss/education"],
        "StatNews": ["https://www.statnews.com/category/health/feed/",
                "https://www.statnews.com/category/pharma/feed/",
                "https://www.statnews.com/category/biotech/feed/",
                "https://www.statnews.com/category/politics/feed/",
                "https://www.statnews.com/category/health-tech/feed/"],
        "NY Times": ["https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
                    "https://rss.nytimes.com/services/xml/rss/nyt/Education.xml",
                    "https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml",
                    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
                    "https://rss.nytimes.com/services/xml/rss/nyt/EnergyEnvironment.xml",
                    "https://rss.nytimes.com/services/xml/rss/nyt/Economy.xml",
                    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
                    "https://rss.nytimes.com/services/xml/rss/nyt/Science.xml",
                    "https://rss.nytimes.com/services/xml/rss/nyt/Health.xml"
                    ],
        "Washington Post": ["https://feeds.washingtonpost.com/rss/business/technology/",
                    "https://feeds.washingtonpost.com/rss/national/",
                    "https://feeds.washingtonpost.com/rss/world/",
                    "https://feeds.washingtonpost.com/rss/business/"],
        "TruthOut": "https://truthout.org/feed/?withoutcomments=1",
        "Politico": ["https://rss.politico.com/congress.xml",
                    "https://rss.politico.com/healthcare.xml",
                    "https://rss.politico.com/defense.xml",
                    "https://rss.politico.com/economy.xml",
                    "https://rss.politico.com/energy.xml",
                    "https://rss.politico.com/politics-news.xml"],
        "Inside Higher Ed": "https://www.insidehighered.com/rss.xml",
        "CNN": ["http://rss.cnn.com/rss/cnn_world.rss",
                "http://rss.cnn.com/rss/cnn_allpolitics.rss",
                "http://rss.cnn.com/rss/cnn_tech.rss",
                "http://rss.cnn.com/rss/cnn_health.rss",
                "http://rss.cnn.com/rss/money_news_international.rss",
                "http://rss.cnn.com/rss/money_news_economy.rss",
                "http://rss.cnn.com/rss/money_markets.rss"
                ],
        "CBC": "https://www.cbc.ca/webfeed/rss/rss-world",
        "Yahoo News": ["https://finance.yahoo.com/news/rssindex",
                    "https://www.yahoo.com/news/rss"],
        "FOX News": ["https://moxie.foxnews.com/google-publisher/world.xml",
                    "https://moxie.foxnews.com/google-publisher/politics.xml",
                    "https://moxie.foxnews.com/google-publisher/science.xml",
                    "https://moxie.foxnews.com/google-publisher/health.xml",
                    "https://moxie.foxnews.com/google-publisher/tech.xml"],
        "ABC News": ["https://abcnews.go.com/abcnews/usheadlines",
                    "https://abcnews.go.com/abcnews/politicsheadlines",
                    "https://abcnews.go.com/abcnews/internationalheadlines",
                    "https://abcnews.go.com/abcnews/moneyheadlines",
                    "https://abcnews.go.com/abcnews/technologyheadlines",
                    "https://abcnews.go.com/abcnews/healthheadlines"],
        "The Guardian": ["https://www.theguardian.com/world/rss",
                        "https://www.theguardian.com/us-news/rss",
                        "https://www.theguardian.com/inequality/rss",
                        "https://www.theguardian.com/science/rss"],
        "Huffington Post": ["https://chaski.huffpost.com/us/auto/vertical/us-news",
                            "https://chaski.huffpost.com/us/auto/vertical/health"],
        "Business Insider": "https://feeds.businessinsider.com/custom/all",
        "Reuters": "https://ir.thomsonreuters.com/rss/news-releases.xml?items=20",
        "Economist": ["https://www.economist.com/finance-and-economics/rss.xml",
"https://www.economist.com/business/rss.xml",
"https://www.economist.com/united-states/rss.xml",
"https://www.economist.com/science-and-technology/rss.xml"],
        "Bloomberg": ["https://feeds.bloomberg.com/markets/news.rss",
                     "https://feeds.bloomberg.com/politics/news.rss",
                     "https://feeds.bloomberg.com/economics/news.rss",
                     "https://feeds.bloomberg.com/industries/news.rss",
                     "https://feeds.bloomberg.com/business/news.rss"],
        "BBC": ["https://feeds.bbci.co.uk/news/world/rss.xml",
               "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
               "https://feeds.bbci.co.uk/news/technology/rss.xml"],
        "BioRxiv": ["https://connect.biorxiv.org/biorxiv_xml.php?subject=all"],
        "ReliefWeb": ["https://reliefweb.int/updates/rss.xml?view=headlines",
                     "https://reliefweb.int/updates/rss.xml",
                     "https://reliefweb.int/disasters/rss.xml"],
        "GDeltProject": ["http://data.gdeltproject.org/gdeltv3/gal/feed.rss"],

                           
              "AP News":["http://associated-press.s3-website-us-east-1.amazonaws.com/business.xml",
                                     "http://associated-press.s3-website-us-east-1.amazonaws.com/climate-and-environment.xml",
                                     "http://associated-press.s3-website-us-east-1.amazonaws.com/health.xml",
                                     "http://associated-press.s3-website-us-east-1.amazonaws.com/politics.xml",
                                     "http://associated-press.s3-website-us-east-1.amazonaws.com/science.xml",
                                     "http://associated-press.s3-website-us-east-1.amazonaws.com/technology.xml",
                                     "http://associated-press.s3-website-us-east-1.amazonaws.com/us-news.xml",
                                     "http://associated-press.s3-website-us-east-1.amazonaws.com/world-news.xml"]}]

        }
#with sync_playwright() as p:
#    browser = p.chromium.launch(headless = True)
#    page = browser.new_page()
#    page.goto("https://gohsep.la.gov/about/news/", wait_until = 'load', timeout = 60_000)
#    page.wait_for_selector('div.col-lg-9 ul li', timeout = 30_000)
#    news_items = page.locator("div.col-lg-9 ul li")
#    extracted_gohsep_news = []

#    for item in news_items.all():
#        link = item.locator("a")
#        url = link.get_attribute("href")
#        url = url if url.startswith("http") else "https://gohsep.la.gov" + url
#        extracted_gohsep_news.append({
#            "source": "Gohsep",
#            "url": url
#        })

#    for news in extracted_gohsep_news:
#        page.goto(news["url"], wait_until = 'load', timeout = 60_000)
#        page.wait_for_selector('p', timeout = 30_000)
#        try:
#            paragraph = page.locator("p.MsoNormal span").all()
#            content = " ".join([p.inner_text() for p in paragraph])
#        except:
#            content = "Content not found"

#        news['content'] = content
#        print(news['content'])
#        rss_feed['Extracted_News'] = {}

        #if news['source'] not in rss_feed['RSS_Feeds']:
          #if news['source'] not in rss_feed["RSS_Feeds"][0]:
            #rss_feed["RSS_Feeds"][0][news['source']] = []
          #rss_feed["RSS_Feeds"][0][news['source']].append(news["url"])
#    browser.close()

paywalled = ['Economist']
keywords = ['Civil Rights', 'Antisemitism', 'Federal Grants','federal grant',
'Discrimination', 'Education Secretary', 'Executive Order', 'Title IX', 'Transgender Athletes',
'Diversity, Equity, and Inclusion', 'DEI', 'Funding Freeze', 'funding frost', 'University Policies', 'Student Success', 'Compliance','Oversight', 'Political Activity', 'Community Relations', 'failure to protect students',
"mockery of this country's higher education system", 'revoking tax-exempt status', 'investigating the university',
'freezing federal funding', 'demanding compliance' ,  'challenges within unit-level initiatives',
'racial equity', 'discrimination against Jewish students', 'ideological agenda', 'Federal Authorities',
'Cancel Visas', 'International Students', 'Department of Homeland Security', 'Immigration Documents',
'Revoking' ,'Deportation', 'Due-Process', 'Executive Order', 'Lawsuit', 'Funding Freeze' ,'Research Grants',
'Indirect Costs' ,'Capping', 'Cultural Funding', 'Budget Cuts' ,'Agency Action', 'Political Statements',
'Legal Status', 'Protests', 'expanded efforts to push international students to leave', 'temporary restraining order',
'violated due-process rights', 'mass layoffs', 'sweeping freeze', 'ideological standards', 'threatened to deport', 'cutting funding', 'Legislature', 'Federal Mandate',
'Legal Injunction', 'National Institutes of Health', 'Legislative Hearing', 'Public Criticism',
'Support for Researches', 'Rescinded Grants', 'Systemic Disease', 'Healthcare Reimbursement',
'Diverse Institution', 'University Admissions', 'First Generation', 'Pell Admission',
'Economic Value', 'Existential Threats' ,'Indirect Overhead Cost', 'Discretionary Funds', 'Endowment Tax',
'Student Visa', 'American University', 'Arizona State University','Boston University','Brown University',
'California State University, Sacramento',
'Chapman University',
'Columbia University',
'Cornell University',
'Drexel University',
'Eastern Washington University',
'Emerson College',
'George Mason University',
'Harvard University',
'Illinois Wesleyan University',
'Indiana University, Bloomington',
'Johns Hopkins University',
'Lafayette College',
'Lehigh University',
'Middlebury College',
'Muhlenberg College',
'Northwestern University',
'Ohio State University',
'Pacific Lutheran University',
'Pomona College',
'Portland State University',
'Princeton University',
'Rutgers University',
'Rutgers University-Newark',
'Santa Monica College',
'Sarah Lawrence College',
'Stanford University',
'State University of New York Binghamton',
'State University of New York Rockland',
'State University of New York, Purchase',
'Swarthmore College',
'Temple University',
'The New School',
'Tufts University',
'Tulane University',
'Union College',
'University of California Davis',
'University of California San Diego',
'University of California Santa Barbara',
'University of California, Berkeley',
'University of Cincinnati',
'University of Hawaii at Manoa',
'University of Massachusetts Amherst',
'University of Michigan',
'University of Minnesota, Twin Cities',
'University of North Carolina',
'University of South Florida',
'University of Southern California',
'University of Tampa',
'University of Tennessee',
'University of Virginia',
'University of Washington-Seattle',
'University of Wisconsin, Madison',
'Wellesley College',
'Whitman College',
'Yale University',
'Arizona State University, Main Campus',
'Boise State University',
'Cal Poly Humboldt',
'California State University, San Bernadino',
'Carnegie Mellon University',
'Clemson University',
'Cornell University',
'Duke University',
'Emory University',
'George Mason University',
'Georgetown University',
'Massachusetts Institute of Technology',
'Montana State University-Bozeman',
'New York University',
'NYU',
'Rice University',
'Rutgers University',
'The Ohio State University, Main Campus',
'Towson University',
'University of Arkansas, Fayetteville',
'University of California-Berkeley',
'University of Chicago',
'University of Cincinnati, Main Campus',
'University of Colorado, Colorado Springs',
'University of Delaware',
'University of Kansas',
'University of Kentucky',
'University of Michigan-Ann Arbor',
'University of Minnesota-Twin Cities',
'University of Nebraska at Omaha',
'University of New Mexico, Main Campus',
'University of North Dakota, Main Campus',
'University of North Texas',
'University of Notre Dame',
'University of Nevada',
'University of Oregon',
'University of Rhode Island',
'University of Utah',
'University of Washington-Seattle',
'University of Wisconsin-Madison',
'University of Wyoming',
'Vanderbilt University',
'Washington State University',
'Washington University in St. Louis',
'Grand Valley State University',
'Ithaca College',
'New England College of Optometry',
'University of Alabama at Birmingham',
'University of Minnesota, Twin Cities',
'University of South Florida',
'University of Oklahoma, Tulsa School of Community Medicine',
'visa', 'nih'
]
keywords = [k.lower() for k in keywords]

ENTRY_SEM = asyncio.Semaphore(20)
MAX_ENTRIES_PER_FEED = 100

Github_owner = 'ERSRisk'
Github_repo = 'Tulane-Sentiment-Analysis'
Release_tag = 'rss_json'
Asset_name = 'all_RSS.json.gz'
GITHUB_TOKEN = os.getenv('TOKEN')

nlp = spacy.load('en_core_web_sm')

def _gh_headers():
  token = os.getenv('TOKEN')
  if not token:
    raise RuntimeError("Missing token")
  return {
    "Accept": "application/vnd.github+json",
    "Authorization": f"token {token if token else None}"
  }

def _get_release_by_tag(owner, repo, tag):
  url = f'https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}'
  r = requests.get(url, headers = _gh_headers(), timeout = 60)
  if r.status_code == 404:
    return None
  r.raise_for_status()
  return r.json()

def ensure_release(owner, repo, tag):
  rel = _get_release_by_tag(owner, repo, tag)
  if rel:
    return rel
  url = f"https://api.github.com/repos/{owner}/{repo}/releases"
  payload = {"tag_name": tag, "name": tag, "prerelease": False, "draft": False}
  r = requests.post(url, headers = _gh_headers(), json = payload, timeout = 60)
  r.raise_for_status()
  return r.json()

def upload_asset(owner, repo, release, asset_name, data_bytes, content_type = 'application/gzip'):
  assets_api = release['assets_url']
  r = requests.get(assets_api, headers = _gh_headers(), timeout = 60)
  r.raise_for_status()
  for a in r.json():
    if a.get('name') == asset_name:
      del_r = requests.delete(a['url'], headers = _gh_headers(), timeout = 60)
      del_r.raise_for_status()
      break
  upload_url = release['upload_url'].split('{')[0]
  params = {"name":asset_name}
  headers = _gh_headers()
  headers['Content-type'] = content_type
  up = requests.post(upload_url, headers = headers, params = params, data = data_bytes, timeout = 60)
  if not up.ok:
    raise RuntimeError(f"Upload failed{up.status_code}: {up.text[:500]}")
  return up.json()

def save_new_articles_to_release(all_articles:list, local_cache_path='Online_Extraction/all_RSS.json.gz'):
    Path(local_cache_path).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    # stream to local gzip (no giant in-mem string)
    with gzip.open(local_cache_path, 'wt', encoding='utf-8') as gz:
        gz.write('[')
        first = True
        for item in all_articles:
            if not first:
                gz.write(',')  # comma between records
            else:
                first = False
            gz.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')))
        gz.write(']')
    gzip_sec = time.perf_counter() - t0
    size_mb = Path(local_cache_path).stat().st_size / 1_000_000
    print(f"gzip sec: {gzip_sec:.2f} | size: {size_mb:.1f} MB", flush = True)

    # read back bytes for upload (bounded memory footprint)
    with open(local_cache_path, 'rb') as f:
        gz_bytes = f.read()

    # optional: quick progress prints
    print(f"Built gzip file: {len(gz_bytes)/1_000_000:.1f} MB")

    rel = ensure_release(Github_owner, Github_repo, Release_tag)
    t1 = time.perf_counter()
    upload_asset(Github_owner, Github_repo, rel, Asset_name, gz_bytes)
    print(f"upload sec: {time.perf_counter() - t1:.2f}")


def load_articles_from_release(local_cache_path = 'Online_Extraction/all_RSS.json.gz'):
  rel = _get_release_by_tag(Github_owner, Github_repo, Release_tag)
  if rel:
    asset = next((a for a in rel.get("assets", []) if a["name"] == Asset_name), None)
    if asset:
      r = requests.get(asset["browser_download_url"], timeout = 60)
      if r.ok:
        data = gzip.decompress(r.content).decode("utf-8")
        return json.loads(data)

  p = Path(local_cache_path)
  if p.exists():
    with gzip.open(p, "rb") as f:
      return json.loads(f.read().decode("utf-8"))
  return []

def create_feeds(rss_feed):
    feeds = []
    for group_name, group_list in rss_feed.items():
      for group in group_list:
        for name, urls in group.items():
            if isinstance(urls, str):
                feeds.append({"source": name, "url": urls, "group": group_name})
            else:
                for url in urls:
                    feeds.append({"source": name, "url": url, "group": group_name})
    return feeds

def COGR():
  print("COGR started", flush = True)
  url = 'https://www.cogr.edu/categories/cogr-updates'
  r = requests.get(url)
  html = r.text

  soup = BeautifulSoup(html, 'html.parser')

  div = soup.find_all(class_ = 'barone')
  div = [d for d in div if d.find('p') and d.find('p').find('strong') and '2025' in d.find('p').find('strong').text]
  links = [a['href'] for d in div for a in d.find_all('a', href = True)]
  links = [link for link in links if link.endswith('.pdf')]
  links = links[0:3]


  for_rss = []
  for link in links:
      r = requests.get(link, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
      r.raise_for_status()

      with open("cogr_update.pdf", "wb") as f:
          f.write(r.content)
      doc = fitz.open("cogr_update.pdf")

      sizes = []

      for page in doc:
          d = page.get_text('dict')
          for b in d['blocks']:
              for ln in b.get('lines', []):
                  for sp in ln['spans']:
                      sizes.append(sp['size'])

      levels = sorted(set(sizes), reverse = True)

      H2 = levels[1]
      #for page in doc:
          #d = page.get_text('dict')
          #for b in d['blocks']:
              #for ln in b.get('lines', []):
                  #for sp in ln['spans']:
                      #if sp['size'] == H2:
                          #print(sp['text'])
      sections = []
      current = None
      pending_header = []
      skipping = False

      for pno, page in enumerate(doc):
          page_h = page.rect.height
          d = page.get_text('dict')
          for b in d['blocks']:
              for ln in b.get('lines', []):
                  spans = ln.get('spans', [])
                  line_text = "".join(sp['text'] for sp in spans).strip()
                  if not line_text:
                      continue
                  y0 = min(sp['bbox'][1] for sp in spans) if spans else 0
                  y1 = max(sp['bbox'][3] for sp in spans) if spans else 0
                  if line_text.isdigit() and (y1 > page_h - 36 or y0 < 36):
                      continue
                  sizes = [sp['size'] for sp in spans]
                  if H2 in sizes:
                      if line_text.isdigit():
                          continue
                      if 'Reminders' in line_text or "LinkedIn" in line_text:
                          skipping = True
                          continue
                      skipping = False
                      if current:
                          current["page_end"] = pno
                          current["body"] = "\n".join(current["body"])
                          sections.append(current)
                          current = None
                      looks_header = line_text
                      pending_header.append(looks_header)
                      continue
                  if skipping:
                    continue
                  if pending_header:
                      header = " ".join(pending_header)
                      current = {"header": header, "body": [], "page_start": pno, "page_end": pno}
                      pending_header = []

                  if current:
                      current['page_end'] = pno
                      current['body'].append(line_text)


      if current:
          current['body'] = "\n".join(current['body'])
          sections.append(current)


      eo_line = re.compile(r'\b(?:E\.?\s*O\.?|EO)\s*(?:No\.?\s*)?(\d{5})\b', re.I)
      for s in sections:
          title = s['header']
          content = s['body']
          published = datetime.now().strftime("%Y-%m-%d")
          spacy_doc = nlp((title + ' ' + content))
          ents = [ent.text for ent in spacy_doc.ents if ent.label_ in ('ORG','PERSON','GPE','LAW','EVENT','MONEY')]
          kws  = [kw for kw in keywords if kw in (title + ' ' + content).lower()]

          if 'executive order' in title.lower():
              eo_blocks = []
              current = None

              for line in content.splitlines():
                  if eo_line.search(line):
                      if current:
                          eo_blocks.append(current)
                      m = eo_line.search(line)
                      eo_num = m.group(1) if m else None
                      current = {'title': f'Executive Order ({eo_num}) Update' if eo_num else title, 'lines': [line]}
                  else:
                      if current:
                          current['lines'].append(line)
              if current:
                  eo_blocks.append(current)
              if not eo_blocks:
                  for_rss.append({
                      "Title": title,
                      "Link": link,
                      "Published": published,
                      "Summary": content[:200],
                      "Content": content,
                      "Source": "COGR",
                      "Entities": ents,
                      "Keyword": kws
                  })
              else:
                  for block in eo_blocks:
                      body = "\n".join(block['lines'])
                      for_rss.append({
                          "Title": block['title'],
                          "Link": link,
                          "Published": published,
                          "Summary": body[:200],
                          "Content": body,
                          "Source": "COGR",
                          "Entities": ents,
                          "Keyword": kws
                      })
          else:
              for_rss.append({"Title": title, "Link": link, "Published": published, "Summary": content[:200], "Content": content, "Source": "COGR", "Entities": ents, "Keyword": kws})  
  print("COGR ended", flush =  True)
  return for_rss
def get_articles_with_full_content(articles, timezone="CDT"):
    """Replace truncated content with full article text and extract formatted date and time"""
    updated_articles = []
    seen_titles = set()

    #Determine offset based on selected timezone
    if timezone == "UTC":
        offset = timedelta(hours=0)
        tz_label = "UTC"
    elif timezone == "CST":
        offset = timedelta(hours=-6)
        tz_label = "CST"
    elif timezone == "CDT":
        offset = timedelta(hours=-5)
        tz_label = "CDT"
    else:
        offset = timedelta(hours=0)
        tz_label = "UTC"

    for article in articles:
        title = article["title"]
        if title in seen_titles:
            continue  # Skip duplicate
        seen_titles.add(title)

        #Get full text if the content is truncated
        full_text = fetch_content_news(article['url']) or article.get('content')
        #Parse publishedAt and split into date and time
        original_dt_str = article.get("publishedAt", "N/A")
        #try to parse and convert
        try:
            original_dt = parser.parse(original_dt_str)
            adjusted_dt = original_dt + offset  # Convert from UTC to CST
            adjusted_date = adjusted_dt.strftime("%m/%d/%Y")
            adjusted_time = adjusted_dt.strftime("%I:%M %p ") + tz_label
        except Exception:
            adjusted_date = "N/A"
            adjusted_time = "N/A"
        ents = [ent.text for ent in spacy_doc.ents if ent.label_ in ('ORG','PERSON','GPE','LAW','EVENT','MONEY')]
        kws  = [kw for kw in keywords if kw in (title + ' ' + text).lower()]

        updated_articles.append({
            "Title": article["title"],
            "Link": article["url"],
            "Published": adjusted_date + " " + adjusted_time,
            "Summary": article.get("description", "No description available."),
            "Content": full_text if full_text else article["content"],
            "Source": article["source"]["name"] if article.get("source") else "N/A",
            "Entities": ents,
            "Keywords": kws
        })
    return updated_articles
def homeland_sec():
  print("Homeland security started", flush = True)
  url = "https://nola.gov/next/homeland-security/news/"
  response = requests.get(url)
  soup = BeautifulSoup(response.content, 'html.parser')
  articles = soup.find_all('div', class_='media mt-4 media-news')
  blocks = [block for block in articles if block.find('a')]
  links = [a['href'] for block in blocks for a in block.find_all('a', href=True)]
  links = [u for u in links if u.rstrip('/').lower() != 'https://nola.gov/next/news']
  links = [urljoin('https://nola.gov', link) for link in links]
  links = [link.replace('u202f', '\u202f') for link in links]

  nola_rss = []
  for link in links:
      link = requote_uri(link)
      if not link or "nola.gov" not in link.lower():
        continue
      downloaded = trafilatura.fetch_url(link)
      soup = BeautifulSoup(downloaded, 'html.parser')
      title = soup.find('h2', class_='mt-0').get_text(strip=True) if soup.find('h2', class_='mt-0') else 'No Title Found'
      text = trafilatura.extract(downloaded, include_comments=False, include_tables=False, include_formatting=False)
      el = soup.find('p', class_='updatedDate').get_text(strip = True) if soup.find('p', class_='updatedDate') else ''
      date_str = re.search(r'(?<!\d)(\d{1,2}/\d{1,2}/\d{4})(?!d)', el).group(1) if re.search(r'(?<!\d)(\d{1,2}/\d{1,2}/\d{4})(?!d)', el) else 'No Date Found'
      published = pd.to_datetime(date_str, format = '%m/%d/%Y', errors = 'coerce').date() if date_str != 'No Date Found' else 'No Date Found'
      published = published.strftime('%Y-%m-%d') if published != 'No Date Found' else 'No Date Found'
      spacy_doc = nlp(text or '')
      ents = [ent.text for ent in spacy_doc.ents if ent.label_ in ('ORG','PERSON','GPE','LAW','EVENT','MONEY')]
      kws  = [kw for kw in keywords if kw in (title + ' ' + text).lower()]
      nola_rss.append({
          'Title': title,
          'Link': link,
          'Published': published,
          'Summary': text[:200] + '...' if text else 'No Summary Found',
          'Content': text if text else 'No Content Found',
          'Source':'NOLA.gov',
          'Entities': ents, 
          'Keyword': kws
      })
  print("Homeland ended", flush = True)
  return nola_rss
def Ace():
  print("Starting Ace", flush = True)
  url = "https://www.acenet.edu/News-Room/Pages/default.aspx"
  response = requests.get(url)
  soup = BeautifulSoup(response.text, 'html.parser')
  titles = soup.find_all('div', class_='rollup-title')
  titles = [title.get_text(strip=True) for title in titles]
  articles = soup.find_all('div', class_='rollup-result tile-type-News tile-type-5')
  articles_list = []
  for article in articles:
      title = article.find('div', class_='rollup-title').get_text(strip=True)
      link = article.find('a', href=True)['href']
      articles_list.append((title, link))

  acenet_data = []
  for title, link in articles_list:
      downloaded = trafilatura.fetch_url(link)
      text = trafilatura.extract(downloaded)
      soup = BeautifulSoup(downloaded, 'html.parser')
      date = soup.find('div', class_='date').get_text(strip=True) if soup.find('div', class_='date') else 'No date found'
      published = pd.to_datetime(date, errors='coerce')
      published = published.strftime('%Y-%m-%d') if pd.notnull(published) else 'Unknown date'
      summary = soup.find('span', class_='ms-rteStyle-StoryLeadIn').get_text(strip=True) if soup.find('span', class_='ms-rteStyle-StoryLeadIn') else text[:200] + '...'
      spacy_doc = nlp(text or '')
      ents = [ent.text for ent in spacy_doc.ents if ent.label_ in ('ORG','PERSON','GPE','LAW','EVENT','MONEY')]
      kws  = [kw for kw in keywords if kw in (title + ' ' + text).lower()]
      acenet_data.append({
          'Title': title,
          'Link': link,
          'Published': published,
          'Summary': summary,
          'Content': text,
          'Source':'American Council on Education',
          'Entities': ents,
          'Keyword': kws}
          )
  print("Ace done", flush = True)
  return acenet_data
def Deloitte():
  print("Starting Deloittle", flush = True)
  url = "https://www.deloitte.com/us/en/insights/industry/articles-on-higher-education.html"
  with sync_playwright() as p:
      browser = p.chromium.launch(headless=True)
      page = browser.new_page()
      page.goto(url, wait_until = 'networkidle')
      page.wait_for_selector('div.card-rows')
      html = page.content()
      browser.close()
  soup = BeautifulSoup(html, 'html.parser')
  articles = soup.find_all('div', class_='card-rows')
  blocks = [l for l in articles if l.find('a')]
  links = [a['href'] for block in blocks for a in block.find_all('a', href=True)]
  links = [urljoin(url, link) for link in links]

  rss_add = []
  for link in links[0:5]:
      downloaded = trafilatura.fetch_url(link)
      soup = BeautifulSoup(downloaded, 'html.parser')
      for el in soup.select('.cmp-di-article-info__content-divider, .cmp-di-profile-promo__country'):
          el.decompose()
      title = soup.find('h1').get_text(strip=True) if soup.find('h1') else ''
      published = soup.select('.cmp-di-article-info__content__read-time')
      published = published[1].get_text(strip=True) if published else ''
      published = pd.to_datetime(published, format='%d %B %Y')
      published = published.strftime('%Y-%m-%d') if not pd.isna(published) else ''
      summary = soup.find('h2', class_='cmp-subtitle__text').get_text(strip=True) if soup.find('h2') else ''
      cleaned_html = str(soup)
      text = trafilatura.extract(cleaned_html, include_comments = False, include_tables = False, include_links = False, favor_recall = True, include_formatting = True) if downloaded else ''
      spacy_doc = nlp(text or '')
      ents = [ent.text for ent in spacy_doc.ents if ent.label_ in ('ORG','PERSON','GPE','LAW','EVENT','MONEY')]
      kws  = [kw for kw in keywords if kw in (title + ' ' + text).lower()]
      rss_add.append({'Title': title, 'Link': link, 'Published':published, 'Summary':summary, 'Content':text, 'Source':'Deloitte Insights', 'Entities': ents, 'Keyword': kws})
  print("Deloitte ended", flush = True)
  return rss_add
def load_existing_articles():
    return load_articles_from_release()

def save_new_articles(existing_articles, new_articles):
    existing_urls = {article['Link'] for article in existing_articles}
    unique_new_articles = [article for article in new_articles if article['Link'] not in existing_urls]

    print(f"Existing articles: {len(existing_articles)}")
    print(f"New unique articles: {len(unique_new_articles)}")

    if unique_new_articles:
        updated_articles = existing_articles + unique_new_articles
        print(f"Saving {len(updated_articles)} total articles to Releases")
        save_new_articles_to_release(updated_articles)
    else:
        print("No new unique articles found.")
    return []

def fetch_content(article_url):
    try:
        # hard timeouts: 10s to connect, 30s total
        r = requests.get(
            article_url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=(10, 30)
        )
        if not r.ok:
            return None
        html = r.text or ""
        if not html.strip():
            return None
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            include_links=False,
            favor_recall=True
        )
        return extracted if extracted else None
    except Exception as e:
        print(f"Error fetching content from {article_url}: {e}")
        return None

def sanitize_dataframe(df):
    return df.applymap(
        lambda x: str(x).replace('\n', ' ')
                    .replace('\r', ' ')
                    .replace('\t', ' ')
                    .replace('"', '""')  # escape double quotes for CSV
                    .strip()
        if isinstance(x, str) else x
    )

def get_available(entry, keys, default = None):
    for key in keys:
        value = entry.get(key)
        if value:
            return value
    return default


final_articles = []
async def fetch_article_content(url):
    try:
        async with ENTRY_SEM:
            return await asyncio.wait_for(
                asyncio.to_thread(fetch_content, url),
                timeout=60
            )
    except asyncio.TimeoutError:
        print(f"⚠️ Timeout fetching article: {url}")
        return None


async def safe_feed_parse(text):
    try:
        proc = await asyncio.create_subprocess_exec(
        sys.executable, "-c",
        (
            "import sys, feedparser, json\n"
            "raw = sys.stdin.read()\n"
            "try:\n"
            "  p = feedparser.parse(raw)\n"
            "  print(json.dumps({\n"
            "    'bozo': p.bozo,\n"
            "    'bozo_exception': str(p.bozo_exception) if p.bozo else None,\n"
            "    'entries': [\n"
            "      {'title': e.get('title'), 'link': e.get('link'), 'published': e.get('published'), 'summary': e.get('summary')}\n"
            "      for e in p.entries\n"
            "    ]\n"
            "  }))\n"
            "except Exception as e:\n"
            "  print(json.dumps({'error': str(e)}))"
        ),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(input=text.encode()), timeout=60
            )
        except asyncio.TimeoutError:
            proc.kill()
            print(f"Parser subprocess timed out for a feed!")
            return None

        if proc.returncode != 0:
            print(f"Parser subprocess crashed: {stderr.decode()}")
            return None

        result = json.loads(stdout.decode())
        if 'error' in result:
            print(f"Parser error: {result['error']}")
            return None
        return result
    except Exception as e:
        print(f"Subprocess failed: {e}")
        return None
async def process_feeds(feeds, session):
    articles = [] 
    for feed in feeds:
        name = feed["source"]
        url = feed["url"]   
        print(f"✅ Processing feed {name} - {url}", flush=True)
        if '/video/' in url or '/podcast/' in url:
            print(f"Skipping video or podcast feed: {url}")
            continue
        try:
            req_timeout = aiohttp.ClientTimeout(total = 60, connect =20, sock_connect = 20, sock_read = 40)
            resp =await asyncio.wait_for(session.get(url, timeout = req_timeout), timeout = 70)
            async with resp as response:
                text = await response.text()
                ctype = (response.headers.get('Content-Type') or '').lower()
                if not any(t in ctype for t in ('xml', 'rss', 'atom')):
                    print(f"Skipping non-XML content: {url} ({ctype})")
                    continue
                feed_extract = await safe_feed_parse(text)
                if not feed_extract:
                    print(f"Skipping feed {url} due to parse failure")
                    continue
        except Exception as e:
            print(f"⚠️ Hard failure fetching {url}: {e}")
            continue        

        if feed_extract['bozo']:
            print(f"Error parsing feed {url}: {feed_extract['bozo_exception']}")
            continue  # skip this feed, already logged

        async def process_entry(entry, source):
            try:
                content = await fetch_article_content(entry['link'])
                text = content if content else get_available(entry, ["summary"])
                if not text or not text.strip():
                    print(f"Skipping with no valid entry text {entry['link']}")
                    return None

                doc = await asyncio.to_thread(nlp, text)
                relevant_entities = ['ORG', 'PERSON', 'GPE', 'LAW', 'EVENT', 'MONEY']
                entities = [ent.text for ent in doc.ents if ent.label_ in relevant_entities]

                combined_text = " ".join(filter(None, [
                    entry.get('title', ''),
                    entry.get('summary', ''),
                    text
                ])).lower()

                matched_keywords = [keyword for keyword in keywords if keyword in combined_text]
                return {
                    "Title": entry.get('title'),
                    "Link": entry.get('link'),
                    "Published": entry.get('published'),
                    "Summary": entry.get('summary'),
                    "Content": "Paywalled article" if any(p.lower() in name.lower() for p in paywalled) else text,
                    "Source": source,
                    "Keyword": matched_keywords,
                    "Entities": entities if entities else None
                }
            except Exception as e:
                print(f"Error processing entry {entry.get('link')}: {e}")
                return None

        entries = feed_extract['entries'][:MAX_ENTRIES_PER_FEED]
        tasks = [process_entry(entry, name) for entry in entries]
        entry_results = await asyncio.gather(*tasks, return_exceptions = True)
        articles.extend([r for r in entry_results if r and not isinstance(r, Exception)])

    return articles

COOKIE_HEADER = os.getenv("COOKIE_HEADER")
async def batch_process_feeds(feeds, batch_size = 15, concurrent_batches =5, deadline_seconds = None, partial_path = Path('Online_Extraction/partial_all_RSS.json.gz')):
    partial_path.parent.mkdir(parents = True, exist_ok = True)
    all_articles = []
    seen_links = set()
    batches = [feeds[i:i + batch_size] for i in range(0, len(feeds), batch_size)]
    client_timeout = aiohttp.ClientTimeout(
      total = None,
      connect = 20,
      sock_connect = 20,
      sock_read = 40,
    )
    headers = {
    "Cookie": COOKIE_HEADER,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    start = time.perf_counter()
    connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=300, family=socket.AF_INET)
    async with aiohttp.ClientSession(headers=headers, timeout=client_timeout, connector=connector) as session:
        for i in range(0, len(batches), concurrent_batches):
            if deadline_seconds is not None and (time.perf_counter() - start) > deadline_seconds:
              print("Soft deadline reached - stopping  after current progress", flush = True)
              break
            batch_group = batches[i:i + concurrent_batches]
            print(f"Processing batch {i // batch_size + 1} with {len(batches)} feeds")
            tasks = [asyncio.create_task(process_feeds(batch, session)) for batch in batch_group]
            results = await asyncio.gather(*tasks)
            added = 0
            for r in results:
              if isinstance(r, Exception):
                print(f"batch task error: {r}", flush = True)
                continue
              for item in r:
                link = item.get('Link')
                if link and link not in seen_links:
                  seen_links.add(link)
                  all_articles.append(item)
                  added +=1
            tmp = partial_path.with_suffix(partial_path.suffix + '.part')
            try:
              with gzip.open(tmp, 'wt', encoding = 'utf-8') as f:
                json.dump(all_articles, f, ensure_ascii = False, separators = (',', ':'))
                tmp.replace(partial_path)
            finally:
              if tmp.exists():
                try:
                  tmp.unlink()
                except Exception:
                  pass
    return all_articles

def AAU_Press_Releases(max_articles=None, save_format='csv'):
    """Scrape press releases from AAU (Association of American Universities) and save to file
   
    Args:
        max_articles (int): Maximum number of articles to process. If None, process all found articles.
        save_format (str): Format to save results ('csv', 'json', 'both', or 'none')
    """
    print(f"Starting AAU Press Releases scraping...")
    base_url = "https://www.aau.edu"
    url = "https://www.aau.edu/newsroom/press-releases"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    }

    try:
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        # Find press release links - looking for links in the press releases listing
        article_links = []

        # Look for press release links in various possible locations
        link_selectors = [
            'a[href*="/press-release"]',
            'a[href*="/news/"]',
            'h2 a',  # common pattern for headlines
            'h3 a',
            '.views-row a',  # Drupal-based sites often use this
            '.node-title a',
            '.title a',
            'li a'  # list items often contain links
        ]

        for selector in link_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href and ('press-release' in href or 'news' in href):
                    full_url = href if href.startswith('http') else urljoin(base_url, href)
                    if 'aau.edu' in full_url and full_url not in article_links:
                        article_links.append(full_url)

        # Also look for any links that might contain press releases
        for a in soup.find_all('a', href=True):
            href = a['href']
            if any(pattern in href for pattern in ['/press-release', '/news/', '/article/']):
                full_url = href if href.startswith('http') else urljoin(base_url, href)
                if 'aau.edu' in full_url and full_url not in article_links:
                    article_links.append(full_url)

        # Remove duplicates
        article_links = list(dict.fromkeys(article_links))
        print(f"Found {len(article_links)} unique press release links")

        # Process articles - if max_articles is None, process all
        if max_articles is None:
            articles_to_process = len(article_links)
        else:
            articles_to_process = min(len(article_links), max_articles)

        print(f"Processing {articles_to_process} press releases...")

        press_releases = []

        for i, link in enumerate(article_links[:articles_to_process]):
            try:
                print(f"Processing press release {i+1}/{articles_to_process}: {link}")

                article_response = requests.get(link, headers=headers, timeout=60)
                article_response.raise_for_status()

                html_content = article_response.text
                soup = BeautifulSoup(html_content, 'html.parser')

                # Extract title
                title = "No title found"
                title_selectors = [
                    'h1',
                    'h1[class*="title"]',
                    'h1[class*="headline"]',
                    '.page-title',
                    '.article-title',
                    '.node-title',
                    'title'
                ]
                for selector in title_selectors:
                    element = soup.select_one(selector)
                    if element:
                        title_text = element.get_text(strip=True)
                        if title_text and len(title_text) > 5:
                            title = title_text
                            break

                # Extract publication date
                published = ""
                date_selectors = [
                    'time[datetime]',
                    '.date',
                    '.publish-date',
                    '.post-date',
                    '.article-date',
                    '.field-name-post-date',
                    'meta[property="article:published_time"]',
                    'meta[name="publish-date"]',
                    '.submitted'  # Drupal sites often use this
                ]
                for selector in date_selectors:
                    try:
                        if selector.startswith('meta'):
                            element = soup.select_one(selector)
                            if element and element.get('content'):
                                published = element['content']
                                break
                        else:
                            element = soup.select_one(selector)
                            if element:
                                if element.get('datetime'):
                                    published = element['datetime']
                                    break
                                else:
                                    date_text = element.get_text(strip=True)
                                    if date_text and any(char.isdigit() for char in date_text):
                                        published = date_text
                                        break
                    except:
                        continue

                # Format date
                if published:
                    try:
                        published_dt = pd.to_datetime(published, errors='coerce')
                        if not pd.isna(published_dt):
                            published = published_dt.strftime('%Y-%m-%d')
                        else:
                            published = ""
                    except:
                        published = ""

                # Extract summary
                summary = ""
                summary_selectors = [
                    'meta[property="og:description"]',
                    'meta[name="description"]',
                    '.field-name-body',  # Drupal body field
                    '.article-summary',
                    '.excerpt',
                    '.intro-text'
                ]
                for selector in summary_selectors:
                    try:
                        if selector.startswith('meta'):
                            element = soup.select_one(selector)
                            if element and element.get('content'):
                                summary = element['content']
                                break
                        else:
                            element = soup.select_one(selector)
                            if element:
                                summary_text = element.get_text(strip=True)
                                if summary_text:
                                    summary = summary_text
                                    break
                    except:
                        continue

                # Extract content using trafilatura
                text = trafilatura.extract(
                    html_content,
                    include_comments=False,
                    include_tables=False,
                    include_links=False,
                    favor_recall=True
                ) or ""

                # If trafilatura fails, try to extract content manually
                if not text:
                    content_selectors = [
                        '.field-name-body',  # Drupal content
                        '.article-content',
                        '.main-content',
                        '.content',
                        '#content'
                    ]
                    for selector in content_selectors:
                        element = soup.select_one(selector)
                        if element:
                            text = element.get_text(strip=True)
                            if text:
                                break

                # Simple entity extraction for universities and organizations
                def extract_simple_entities(text):
                    if not text:
                        return []
                    entities = []
                    patterns = [
                        r'[A-Z][a-z]+(?: [A-Z][a-z]+)* (?:University|College|Institute|School)',
                        r'[A-Z][a-z]+(?: [A-Z][a-z]+)* (?:Inc|LLC|Corp|Ltd|Foundation)',
                        r'Association of American Universities',
                        r'AAU',
                        r'National Science Foundation',
                        r'NSF',
                        r'National Institutes of Health',
                        r'NIH'
                    ]
                    for pattern in patterns:
                        try:
                            entities.extend(re.findall(pattern, text))
                        except:
                            continue
                    return list(set(entities))[:10]

                entities = extract_simple_entities(text)

                press_releases.append({
                    'Title': title,
                    'Link': link,
                    'Published': published,
                    'Summary': summary,
                    'Content': text,
                    'Source': 'AAU Press Releases',
                    'Entities': entities,
                    'Keyword': []
                })

            except Exception as e:
                print(f"✗ Error processing press release: {e}")
                continue
        return press_releases
    except Exception as e:
        print(f"Error: {e}")
        return []
def hullabaloo():
  print("Tulane Hullabaloo started", flush = True)
  url = "https://tulanehullabaloo.com/category/news/"
  response = requests.get(url)
  soup = BeautifulSoup(response.content, 'html.parser')
  articles = soup.find_all('div', class_='catlist-textarea-with-media')
  blocks = [block for block in articles if block.find('a')]
  links = [a['href'] for block in blocks for a in block.find_all('a', href=True)]
  
  
  hullabaloo = []
  for link in links:
      if re.search(r"staff_name", link):
          continue
      r = requests.get(link, timeout=30)
      r.raise_for_status()
      soup = BeautifulSoup(r.text, "html.parser")
      title = None
  
      # 1) Meta tags (most reliable)
      for sel in ['meta[property="og:title"]', 'meta[name="twitter:title"]']:
          m = soup.select_one(sel)
          if m and m.get("content"):
              title = m["content"].strip()
              break
  
      # 2) Common headline selectors on SNO/WordPress themes
      if not title:
          for sel in ["h1.sno-title",
                      "h1.entry-title",
                      "h1.headline",
                      "h1.post-title",
                      "header .sno-story-headline h1",
                      "h1"]:
              h = soup.select_one(sel)
              if h and h.get_text(strip=True):
                  title = h.get_text(strip=True)
                  break
  
      # 3) Fallback: feature image alt (present in your snippet)
      if not title:
          img = soup.select_one(".sno-story-photo-area img[alt]")
          if img and img.get("alt"):
              title = img["alt"].strip()
  
      # --- DATE & BYLINE (optional) ---
      date = soup.select_one(".sno-story-date .time-wrapper")
      date_text = date.get_text(strip=True) if date else None
  
      byline = soup.select_one(".sno-story-byline .byline-name")
      byline_text = byline.get_text(strip=True) if byline else None
  
      # --- BODY CONTENT ---
      body = soup.select_one("#sno-story-body-content")
      content_text = ""
      if body:
          # remove junk blocks you don’t want in the article text
          for junk_sel in [
              "script", "style",
              ".mailmunch-forms-before-post",
              ".mailmunch-forms-in-post-middle",
              ".mailmunch-forms-after-post",
              ".inline-slideshow-area",
              ".sno-story-photo-area",
          ]:
              for tag in body.select(junk_sel):
                  tag.decompose()
  
          # collect readable paragraphs / lists / subheads
          parts = []
          for el in body.find_all(["p", "li", "h2", "h3"]):
              txt = el.get_text(" ", strip=True)
              if txt:
                  parts.append(txt)
  
          content_text = "\n\n".join(parts)
  
      hullabaloo.append({
          "Title": title or "No Title Found",
          "Link": link,
          "Published": date_text or "No Date Found",
          "Summary": content_text[:200],
          "Content": content_text,
          "Source": "Tulane Hullabaloo"
      })
  print("Hulabaloo ended", flush = True)
  return hullabaloo
def Chronicle(max_articles=None, save_format='csv'):
  """Scrape articles from Chronicle of Higher Education and save to file
 
  Args:
      max_articles (int): Maximum number of articles to process. If None, process all found articles.
      save_format (str): Format to save results ('csv', 'json', 'both', or 'none')
  """
  print(f"Starting Chronicle scraping...")
  url = "https://www.chronicle.com/"

  headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
  }

  try:
      response = requests.get(url, headers=headers, timeout=60)
      response.raise_for_status()

      soup = BeautifulSoup(response.content, 'html.parser')

      # Find article links
      article_links = []
      for a in soup.find_all('a', href=True):
          href = a['href']
          if any(pattern in href for pattern in ['/article/', '/news/', '/story/', '/post/']):
              full_url = href if href.startswith('http') else urljoin(url, href)
              if 'chronicle.com' in full_url and full_url not in article_links:
                  article_links.append(full_url)

      # Remove duplicates
      article_links = list(dict.fromkeys(article_links))
      print(f"Found {len(article_links)} unique article links")

      # Process articles - if max_articles is None, process all
      if max_articles is None:
          articles_to_process = len(article_links)
      else:
          articles_to_process = min(len(article_links), max_articles)

      print(f"Processing {articles_to_process} articles...")

      rss_add = []

      for i, link in enumerate(article_links[:articles_to_process]):
          try:
              print(f"Processing article {i+1}/{articles_to_process}: {link}")

              article_response = requests.get(link, headers=headers, timeout=60)
              article_response.raise_for_status()

              html_content = article_response.text
              soup = BeautifulSoup(html_content, 'html.parser')

              # Extract title
              title = "No title found"
              title_selectors = ['h1', 'h1[class*="title"]', 'h1[class*="headline"]', 'title']
              for selector in title_selectors:
                  element = soup.select_one(selector)
                  if element:
                      title_text = element.get_text(strip=True)
                      if title_text and len(title_text) > 5:
                          title = title_text
                          break

              # Extract publication date
              published = ""
              date_selectors = [
                  'time[datetime]', '[class*="date"]', '[class*="publish"]',
                  'meta[property="article:published_time"]', 'meta[name="publish-date"]'
              ]
              for selector in date_selectors:
                  try:
                      if selector.startswith('meta'):
                          element = soup.select_one(selector)
                          if element and element.get('content'):
                              published = element['content']
                              break
                      else:
                          element = soup.select_one(selector)
                          if element:
                              published = element.get('datetime') or element.get_text(strip=True)
                              if published:
                                  break
                  except:
                      continue

              # Format date
              if published:
                  try:
                      published_dt = pd.to_datetime(published, errors='coerce')
                      if not pd.isna(published_dt):
                          published = published_dt.strftime('%Y-%m-%d')
                      else:
                          published = ""
                  except:
                      published = ""

              # Extract summary
              summary = ""
              summary_selectors = [
                  'meta[property="og:description"]', 'meta[name="description"]',
                  '[class*="summary"]', '[class*="excerpt"]'
              ]
              for selector in summary_selectors:
                  try:
                      if selector.startswith('meta'):
                          element = soup.select_one(selector)
                          if element and element.get('content'):
                              summary = element['content']
                              break
                      else:
                          element = soup.select_one(selector)
                          if element:
                              summary_text = element.get_text(strip=True)
                              if summary_text:
                                  summary = summary_text
                                  break
                  except:
                      continue

              # Extract content
              text = trafilatura.extract(
                  html_content,
                  include_comments=False,
                  include_tables=False,
                  include_links=False,
                  favor_recall=True
              ) or ""

              # Simple entity extraction
              def extract_simple_entities(text):
                  if not text:
                      return []
                  entities = []
                  patterns = [
                      r'[A-Z][a-z]+(?: [A-Z][a-z]+)* (?:University|College|Institute|School)',
                      r'[A-Z][a-z]+(?: [A-Z][a-z]+)* (?:Inc|LLC|Corp|Ltd)',
                      r'Dr\. [A-Z][a-z]+ [A-Z][a-z]+',
                      r'Prof\. [A-Z][a-z]+ [A-Z][a-z]+',
                  ]
                  for pattern in patterns:
                      try:
                          entities.extend(re.findall(pattern, text))
                      except:
                          continue
                  return list(set(entities))[:10]
              spacy_doc = nlp(text or '')
              ents = [ent.text for ent in spacy_doc.ents if ent.label_ in ('ORG','PERSON','GPE','LAW','EVENT','MONEY')]
              kws  = [kw for kw in keywords if kw in (title + ' ' + text).lower()]

              rss_add.append({
                  'Title': title,
                  'Link': link,
                  'Published': published,
                  'Summary': summary,
                  'Content': text,
                  'Source': 'The Chronicle of Higher Education',
                  'Entities': ents,
                  'Keyword': kws
              })
          except Exception as e:
              print(f"Error processing article {link}: {e}")   
      return rss_add

  except Exception as e:
      print(f"Error: {e}")
      return []
def highered():
  print("Inside Higher Ed started", flush = True)
  data = []
  for i in range(0, 10):
      url = f'https://www.insidehighered.com/news?page={i}'
      base_url = 'https://www.insidehighered.com'
      response = requests.get(url)
      soup = BeautifulSoup(response.content, 'html.parser')
      articles = soup.find_all('h4')
      articles = [a['href'] for article in articles for a in article.find_all('a', href = True)]

      for article in articles:
          url = base_url + article
          s = requests.Session()
          s.headers.update({"Cookie": "ZXlKMGVYQWlPaUpLVjFRaUxDSmhiR2NpT2lKSVV6STFOaUo5LmV5SnBjM01pT2lKb2RIUndjem92TDNkM2R5NXdaV3hqY204dVkyOXRMMkZ3YVM5Mk1TOXpaR3N2WTNWemRHOXRaWEl2Y21WbWNtVnphQ0lzSW1saGRDSTZNVGMxT1RVME9URXlNaXdpWlhod0lqb3hOelkwTnpNek1USXlMQ0p1WW1ZaU9qRTNOVGsxTkRreE1qSXNJbXAwYVNJNklrTTBjbWhLZFROT1lWUjJRMVZRTkVVaUxDSnpkV0lpT2lJNU1qUXlNVEk0SWl3aWNISjJJam9pTWpOaVpEVmpPRGswT1dZMk1EQmhaR0l6T1dVM01ERmpOREF3T0RjeVpHSTNZVFU1TnpabU55SjkuQlU2Y2IwOWRzMmJtUmpLZ1hZajFPYnpNMlkyWGEyVzlhWU5zOUY2LUdvYw=="})
          response = s.get(url)
          soup = BeautifulSoup(response.content, 'html.parser')
          title = soup.find('h1', class_='node-title normal-spacing').get_text(strip = True) if soup.find('h1', class_='node-title normal-spacing') else 'No Title Found'
          text = trafilatura.extract(response.text)
          needle_literal = "You have /5 articles left.\\nSign up for a free account or log in.\n"
          needle_newline = "You have /5 articles left.\nSign up for a free account or log in.\n"

          text = text.replace(needle_literal, '').replace(needle_newline, '')
          summary = soup.find('div', class_='node-lead normal-spacing').get_text(strip = True) if soup.find('div', class_='node-lead normal-spacing') else 'No Summary Found'

          published = soup.select_one('.node-created span').get_text(strip = True) if soup.select_one('.node-created span') else 'Unknown'
          if published != 'Unknown':
              published = pd.to_datetime(published, format = '%B %d, %Y', errors = 'coerce')
              published = published.strftime('%Y-%m-%d') if pd.notnull(published) else 'Unknown'
          data.append({
              'Title': title,
              'Link': url,
              'Published': published,
              'Summary': summary,
              'Content': text if text else 'No Content Found',
              'Source': 'Inside Higher Ed'
          })
  print("Inside Higher ed completed", flush = True)
  return data

def Whitehouse():
  print("Starting whitehouse", flush = True)
  url = 'https://www.whitehouse.gov/presidential-actions/executive-orders/'

  response = requests.get(url)
  soup = BeautifulSoup(response.content, 'html.parser')
  pagination = soup.find_all('a', class_='page-numbers')
  pagination = [a.get_text(strip = True) for a in pagination][-1]
  pagination = int(pagination)

  data = []
  for i in range(1, pagination):
      url = f'https://www.whitehouse.gov/presidential-actions/executive-orders/page/{i}/'
      response = requests.get(url)
      soup = BeautifulSoup(response.content, 'html.parser')
      articles = soup.find_all('h2', class_='wp-block-post-title')
      articles = [a['href'] for article in articles for a in article.find_all('a', href = True)]
      for article in articles:
          response = requests.get(article)
          soup = BeautifulSoup(response.content, 'html.parser')
          title = soup.find('h1', class_='wp-block-whitehouse-topper__headline').get_text(strip = True)
          text = trafilatura.extract(response.text)
          published = soup.find('div', class_='wp-block-post-date').get_text(strip = True)
          published = pd.to_datetime(published, format = '%B %d, %Y', errors = 'coerce')
          published = published.strftime('%Y-%m-%d')
          spacy_doc = nlp(text or '')
          ents = [ent.text for ent in spacy_doc.ents if ent.label_ in ('ORG','PERSON','GPE','LAW','EVENT','MONEY')]
          kws  = [kw for kw in keywords if kw in (title + ' ' + text).lower()]
          data.append({
              'Title': title,
              'Link': article,
              'Published': published,
              'Summary': text[:200] + '...',
              'Content': text,
              'Source': 'White House',
              'Entities': ents,
              'Keyword': kws
          })
  print("Whitehouse fniished", flush = True)
  return data
feeds = create_feeds(rss_feed)
articles = fetch_news(search, start_date, end_date)
articles = get_articles_with_full_content(articles, timezone=timezone_option)
unique_articles = []
seen_titles = set()
for article in articles:
    if article['Title'] not in seen_titles:
        unique_articles.append(article)
        seen_titles.add(article['Title'])
articles = unique_articles

cogr = COGR()
deloitte = Deloitte()
homeland = homeland_sec()
ace = Ace()
#data = Whitehouse()
chronicle = Chronicle(max_articles=None, save_format='none')
aau = AAU_Press_Releases(max_articles=None, save_format='none')
highered = highered()
hullabaloo = hullabaloo()

def run_with_deadline(coro, seconds = 7200):
  return asyncio.run(asyncio.wait_for(coro, timeout = seconds))
try:
    all_articles = run_with_deadline(batch_process_feeds(feeds, batch_size=5, concurrent_batches=3, deadline_seconds = 1700), seconds = 1800)
except asyncio.TimeoutError:
  print("RSS batch hard timeout")
  p = Path('Online_Extraction/partial_all_RSS.json.gz')
  if p.exists():
    with gzip.open(p, 'rt', encoding = 'utf-8') as f:
      all_articles = json.load(f)
  else:
    all_articles = []

all_articles += cogr
all_articles += articles
all_articles += deloitte
all_articles += homeland
all_articles += ace
all_articles += data
all_articles += chronicle
all_articles += aau
all_articles += highered
all_articles += hullabaloo

existing_articles = load_existing_articles()
new_articles = save_new_articles(existing_articles, all_articles)
