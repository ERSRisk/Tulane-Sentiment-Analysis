import aiohttp
import feedparser
import traceback
import spacy
import time
import subprocess
import asyncio
import json
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


rss_feed =   {"RSS_Feeds":[{
              "WHO": ["https://www.who.int/rss-feeds/news-english.xml"],
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
        
                           }]
              #"RSS_URLs":[{"AP News":["http://associated-press.s3-website-us-east-1.amazonaws.com/business.xml",
              #                       "http://associated-press.s3-website-us-east-1.amazonaws.com/climate-and-environment.xml",
               #                      "http://associated-press.s3-website-us-east-1.amazonaws.com/health.xml",
                #                     "http://associated-press.s3-website-us-east-1.amazonaws.com/politics.xml",
                 #                    "http://associated-press.s3-website-us-east-1.amazonaws.com/science.xml",
                  #                   "http://associated-press.s3-website-us-east-1.amazonaws.com/technology.xml",
                   #                  "http://associated-press.s3-website-us-east-1.amazonaws.com/us-news.xml",
                    #                 "http://associated-press.s3-website-us-east-1.amazonaws.com/world-news.xml"]}]
                           
        }
with sync_playwright() as p:
    browser = p.chromium.launch(headless = True)
    page = browser.new_page()
    page.goto("https://gohsep.la.gov/about/news/")
    page.wait_for_load_state("networkidle")
    news_items = page.locator("div.col-lg-9 ul li")
    extracted_gohsep_news = []

    for item in news_items.all():
        link = item.locator("a")
        url = link.get_attribute("href")
        url = url if url.startswith("http") else "https://gohsep.la.gov" + url
        extracted_gohsep_news.append({
            "source": "Gohsep",
            "url": url
        })

    for news in extracted_gohsep_news:
        page.goto(news["url"])
        page.wait_for_load_state("networkidle")
        try:
            paragraph = page.locator("p.MsoNormal span").all()
            content = " ".join([p.inner_text() for p in paragraph])
        except:
            content = "Content not found"

        news['content'] = content
        print(news['content'])
        rss_feed['Extracted_News'] = {}

        if news['source'] not in rss_feed['RSS_Feeds']:
          if news['source'] not in rss_feed["RSS_Feeds"][0]:
            rss_feed["RSS_Feeds"][0][news['source']] = []
          rss_feed["RSS_Feeds"][0][news['source']].append(news["url"])

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
  r = requests.get(url, headers = _gh_headers())
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
  r = requests.post(url, headers = _gh_headers(), json = payload)
  r.raise_for_status()
  return r.json()

def upload_asset(owner, repo, release, asset_name, data_bytes, content_type = 'application/gzip'):
  assets_api = release['assets_url']
  r = requests.get(assets_api, headers = _gh_headers())
  r.raise_for_status()
  for a in r.json():
    if a.get('name') == asset_name:
      del_r = requests.delete(a['url'], headers = _gh_headers())
      del_r.raise_for_status()
      break
  upload_url = release['upload_url'].split('{')[0]
  params = {"name":asset_name}
  headers = _gh_headers()
  headers['Content-type'] = content_type
  up = requests.post(upload_url, headers = headers, params = params, data = data_bytes)
  if not up.ok:
    raise RuntimeError(f"Upload failed{up.status_code}: {up.text[:500]}")
  return up.json()

def save_new_articles_to_release(all_articles:list, local_cache_path = 'Online_Extraction/all_RSS.json.gz'):
  buf = io.BytesIO()
  with gzip.GzipFile(fileobj = buf, mode = 'wb') as gz:
    gz.write(json.dumps(all_articles, ensure_ascii = False, indent = 2).encode("utf-8"))
  gz_bytes = buf.getvalue()

  Path(local_cache_path).parent.mkdir(parents=True, exist_ok = True)
  with open(local_cache_path, 'wb') as f:
    f.write(gz_bytes)

  rel = ensure_release(Github_owner, Github_repo, Release_tag)
  upload_asset(Github_owner, Github_repo, rel, Asset_name, gz_bytes)

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
  return for_rss

def homeland_sec():
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
  return nola_rss
def Ace():
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
      return acenet_data
def Deloitte():
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
        downloaded = trafilatura.fetch_url(article_url)
        if downloaded:
            extracted =  trafilatura.extract(downloaded)
            if extracted:
                return extracted
        return None
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
        return await asyncio.wait_for(
            asyncio.to_thread(fetch_content, url),
            timeout=30
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
                proc.communicate(input=text.encode()), timeout=30
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
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                text = await response.text()
                if 'xml' not in response.headers.get('Content-Type', ''):
                    print(f"Skipping non-XML content: {url}")
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

                doc = nlp(text)
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

        tasks = [process_entry(entry, name) for entry in feed_extract['entries']]
        entry_results = await asyncio.gather(*tasks)
        articles.extend([r for r in entry_results if r])

    return articles

COOKIE_HEADER = os.getenv("COOKIE_HEADER")
async def batch_process_feeds(feeds, batch_size = 15, concurrent_batches =5):
    all_articles = []
    batches = [feeds[i:i + batch_size] for i in range(0, len(feeds), batch_size)]
    headers = {
    "Cookie": COOKIE_HEADER,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    async with aiohttp.ClientSession(headers=headers) as session:
        for i in range(0, len(batches), concurrent_batches):
            batch_group = batches[i:i + concurrent_batches]
            print(f"Processing batch {i // batch_size + 1} with {len(batches)} feeds")
            tasks = [asyncio.create_task(process_feeds(batch, session)) for batch in batch_group]
            results = await asyncio.gather(*tasks)
            for batch_articles in results:
                all_articles.extend(batch_articles)
    return all_articles

feeds = create_feeds(rss_feed)
cogr = COGR()
deloitte = Deloitte()
homeland = homeland_sec()
ace = Ace()

try:
    all_articles = asyncio.run(batch_process_feeds(feeds, batch_size=5, concurrent_batches=2))
except Exception as e:
    print(f"Fatal error {e}") 

all_articles += cogr
all_articles += deloitte
all_articles += homeland
all_articles += ace

existing_articles = load_existing_articles()
new_articles = save_new_articles(existing_articles, all_articles)


