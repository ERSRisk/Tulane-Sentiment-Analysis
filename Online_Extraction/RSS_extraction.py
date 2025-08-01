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

rss_feed =   {"RSS_Feeds":[{
              "WHO": ["https://www.who.int/rss-feeds/news-english.xml"],
              "NIH": ["https://grants.nih.gov/news-events/nih-extramural-nexus-news/rss-feed"],
              "NOAA": ["noaa.gov/rss.xml"],
              "FEMA": ["https://www.fema.gov/feeds/news.rss", "https://www.fema.gov/feeds/disasters-major.rss", "https://www.fema.gov/feeds/disasters-fire.rss"],
              "NASA": ["https://www.nasa.gov/news-release/feed/"],
  "CISA": ["https://www.cisa.gov/news.xml"],
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
        
                           }],
              "RSS_URLs":[{"AP News":["http://associated-press.s3-website-us-east-1.amazonaws.com/business.xml",
                                     "http://associated-press.s3-website-us-east-1.amazonaws.com/climate-and-environment.xml",
                                     "http://associated-press.s3-website-us-east-1.amazonaws.com/health.xml",
                                     "http://associated-press.s3-website-us-east-1.amazonaws.com/politics.xml",
                                     "http://associated-press.s3-website-us-east-1.amazonaws.com/science.xml",
                                     "http://associated-press.s3-website-us-east-1.amazonaws.com/technology.xml",
                                     "http://associated-press.s3-website-us-east-1.amazonaws.com/us-news.xml",
                                     "http://associated-press.s3-website-us-east-1.amazonaws.com/world-news.xml"]}]
                           
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
'NYU'
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

def load_existing_articles():
    if os.path.exists('Online_Extraction/all_RSS.json'):
        with open('Online_Extraction/all_RSS.json', 'r', encoding = 'utf-8') as f:
            return json.load(f)
    return []

def save_new_articles(existing_articles, new_articles):
    existing_urls = {article['Link'] for article in existing_articles}
    unique_new_articles = [article for article in new_articles if article['Link'] not in existing_urls]
    
    print(f"Existing articles: {len(existing_articles)}")
    print(f"New unique articles: {len(unique_new_articles)}")
    
    if unique_new_articles:
        updated_articles = existing_articles + unique_new_articles
        with open('Online_Extraction/all_RSS.json', 'w', encoding='utf-8') as f:
            json.dump(updated_articles, f, indent=4)
        print(f"Saved {len(updated_articles)} articles.")
        return unique_new_articles
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
    
nlp = spacy.load('en_core_web_sm')
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
try:
    all_articles = asyncio.run(batch_process_feeds(feeds, batch_size=5, concurrent_batches=2))
except Exception as e:
    print(f"Fatal error {e}") 
existing_articles = load_existing_articles()
new_articles = save_new_articles(existing_articles, all_articles)
