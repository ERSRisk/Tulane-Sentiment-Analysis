import aiohttp
import feedparser
import traceback
import spacy
import time
import asyncio
import json
import re
import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import pickle
import trafilatura
import os

rss_feed =   {"The Advocate": ["https://www.theadvocate.com/search/?q=&t=article&l=35&d=&d1=&d2=&s=start_time&sd=desc&c%5b%20%5d=new_orleans/news*,baton_rouge/news/politics/legislature,baton_rouge/news/politics,new_orleans/opinion*,baton_rouge/opinion/stephanie_grace,baton_rouge/opinion/jeff_sadow,ba%20ton_rouge/opinion/mark_ballard,new_orleans/sports*,baton_rouge/sports/lsu&nk=%23tncen&f=rss",
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
        "Reuters": "https://ir.thomsonreuters.com/rss/news-releases.xml?items=15",
        "Economist": ["https://www.economist.com/finance-and-economics/rss.xml",
"https://www.economist.com/business/rss.xml",
"https://www.economist.com/united-states/rss.xml",
"https://www.economist.com/science-and-technology/rss.xml"]
        }

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
options = Options()
options.binary_location = '/usr/bin/google-chrome'
options.add_argument('--headless=new')
options.add_argument('--disable-gpu')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
options.add_argument("--window-size=1920,1080")
def get_cookies():
    driver = webdriver.Chrome(options=options)
    driver.get("https://login.nola.com/u/login?state=hKFo2SB4czdxOUVyNmdrVkY5Q0htcnl5Sk9pU1dEQkFWczlScqFur3VuaXZlcnNhbC1sb2dpbqN0aWTZIHBzOUdTQnBidGg0U3MtM25NdFotZDh6djZLS2tJVHZvo2NpZNkgTmFINm9Od0tubng2eWdIMjdMVFBZMVBBcGtSZE5USlU")
    time.sleep(10)
    driver.save_screenshot('screenshot.png')
    WebDriverWait(driver, 510).until(EC.presence_of_element_located((By.ID, "username"))).send_keys("Njenkins4@tulane.edu")
    driver.find_element(By.ID, "password").send_keys("ERSDepartment2023")
    driver.find_element(By.CSS_SELECTOR, 'button[type = "submit"]').click()
    time.sleep(60)
    driver.get("https://myaccount.nola.com/ta/dashboard")
    cookies = driver.get_cookies()
    auth_cookies = {}
    for cookie in cookies:
        if cookie['domain'].endswith('.nola.com'):
            auth_cookies[cookie['name']] = cookie['value']

    with open('nola_auth_headers.pkl', 'wb') as f:
        pickle.dump(auth_cookies, f)
    driver.quit()
    cookie_header = "; ".join([f"{key}={value}" for key, value in auth_cookies.items()])
    print(cookie_header)
    headers = {
    "Cookie": cookie_header,
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    return headers


def create_feeds(rss_feed):
    feeds = []
    for name, urls in rss_feed.items():
        if isinstance(urls, str):
            feeds.append({"source": name, "url": urls})
        else:
            for url in urls:
                feeds.append({"source": name, "url": url})
    return feeds

def load_existing_articles():
    if os.path.exists('all_RSS.json'):
        with open('all_RSS.json', 'r', encoding = 'utf-8') as f:
            return json.load(f)
    return []

def save_new_articles(existing_articles, new_articles):
    existing_urls = {article['Link'] for article in existing_articles}
    unique_new_articles = [article for article in new_articles if article['Link'] not in existing_urls]
    
    if unique_new_articles:
        updated_articles = existing_articles + unique_new_articles
        with open('all_RSS.json', 'w', encoding='utf-8') as f:
            json.dump(updated_articles, f, indent=4)
        return unique_new_articles
    return

def fetch_content(article_url):
    try:
        downloaded = trafilatura.fetch_url(article_url)
        if downloaded:
            return trafilatura.extract(downloaded)
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
    return await asyncio.to_thread(fetch_content, url)

async def process_feeds(feeds, session):
    articles = [] 
    for feed in feeds:
        name = feed["source"]
        url = feed["url"]   
        print(f"Fetching feed from {name}, {url}")
        if '/video/' in url or '/podcast/' in url:
            print(f"Skipping video or podcast feed: {url}")
            continue
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=150)) as response:
                text = await response.text()
                feed_extract = feedparser.parse(text)
                
        except Exception as e:
            print(f"Error fetching {url}: {repr(e)}")
            traceback.print_exc()
            continue
        if feed_extract.bozo:
            print(f"Error parsing feed {url}: {feed_extract.bozo_exception}")
            try:
                for entry in feed_extract.entries:
                    content = await fetch_article_content(entry.link)
                    text = await response.read()
                    feed_extract = feedparser.parse(text.decode("utf-8"))
                    article_data = {
                            "Title": entry.title,
                            "Link": entry.link,
                            "Published": get_available(entry, ['published', 'pubDate', 'updated']),
                            "Summary": get_available(entry, ["summary", "description", "content"]),
                            "Content": "Paywalled article" if any(p.lower() in name.lower() for p in paywalled) else content if content else get_available(entry, ["summary", "description", "content"])
                        }
                    articles.append(article_data)
            except Exception as e:
                print(f"Failed to fetch or parse feed {url} after retry: {e}")
            continue
        async def process_entry(entry, source):
            try:
                content = await fetch_article_content(entry.link)
                nlp = spacy.load('en_core_web_sm')
                text = content if content else get_available(entry, ["summary", "description", "content"])
                doc = nlp(text)
                relevant_entities = ['ORG', 'PERSON', 'GPE', 'LAW', 'EVENT', 'MONEY']
                entities = [ent.text for ent in doc.ents if ent.label_ in relevant_entities]
                text_to_check = (
                    entry.title,
                    get_available(entry, ["summary", "description", "content"]),
                    content or get_available(entry, ["summary", "description", "content"])
                )
                combined_text = " ".join(filter(None, text_to_check)).lower()
                
                matched_keywords = [keyword for keyword in keywords if keyword in combined_text]
                return {
                    "Title": entry.title,
                    "Link": entry.link,
                    "Published": get_available(entry, ['published', 'pubDate', 'updated']),
                    "Summary": get_available(entry, ["summary", "description", "content"]),
                    "Content": "Paywalled article" if any(p.lower() in name.lower() for p in paywalled) else content or get_available(entry, ["summary", "description", "content"]),
                    "Source": source,
                    "Keyword": matched_keywords,
                    "Entities": entities if entities else None
                }
            except Exception as e:
                print(f"Error processing entry {entry.link}: {e}")
                return None
        tasks = [process_entry(entry, name) for entry in feed_extract.entries]
        entry_results = await asyncio.gather(*tasks)
        articles.extend([r for r in entry_results if r])

    return articles

async def batch_process_feeds(feeds, batch_size = 15, concurrent_batches =5):
    all_articles = []
    batches = [feeds[i:i + batch_size] for i in range(0, len(feeds), batch_size)]
    async with aiohttp.ClientSession(headers=get_cookies()) as session:
        for i in range(0, len(feeds), concurrent_batches):
            batch_group = batches[i:i + concurrent_batches]
            print(f"Processing batch {i // batch_size + 1} with {len(batches)} feeds")
            tasks = [asyncio.create_task(process_feeds(batch, session)) for batch in batch_group]
            results = await asyncio.gather(*tasks)
            for batch_articles in results:
                all_articles.extend(batch_articles)
    return all_articles

feeds = create_feeds(rss_feed)
all_articles = asyncio.run(batch_process_feeds(feeds, batch_size=15, concurrent_batches=3))
existing_articles = load_existing_articles()
new_articles = save_new_articles(existing_articles, all_articles)
