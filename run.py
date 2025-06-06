import news as test
import datetime
from google import genai
import asyncio
from datetime import timedelta
import json
import toml
import tweets_extract as te
import rss

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


NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_NEWS")
GEMINI_API_KEY_X = os.getenv("GEMINI_API_KEY_X")
X_API_KEY = os.getenv("X_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY)
search = 'Tulane'
start_date = (datetime.date.today() - timedelta(days = 7))
start_date_X = (datetime.date.today() - timedelta(days = 6))
end_date = datetime.date.today()
timezone_option = 'CDT'

articles = test.fetch_news(search, start_date, end_date)
articles = test.get_articles_with_full_content(articles, timezone=timezone_option)
unique_articles = []
seen_titles = set()
for article in articles:
    if article['title'] not in seen_titles:
        unique_articles.append(article)
        seen_titles.add(article['title'])
articles = unique_articles

gemini_response_text = test.run_async_batches(articles, search, timezone_option, batch_size=10)
results = test.text_to_dataframe(gemini_response_text, articles)
existing_articles_news = test.load_existing_articles_news()
new_articles_news = test.save_new_articles_news(existing_articles_news, results)
with open('extracted_news.json', 'w') as f:
        json.dump(new_articles_news, f)

tweets = te.fetch_twits(search, start_date_X, end_date, 100)
df = asyncio.get_event_loop().run_until_complete(te.run_async_batches_X(tweets, search, batch_size=10))
existing_posts = te.load_existing_posts_X()
new_posts = te.save_new_posts_X(existing_posts, df)
with open('tweets.json', 'w') as f:
    json.dump(new_posts, f)

feeds = rss.create_feeds(rss_feed)
all_articles = asyncio.run(rss.batch_process_feeds(feeds, batch_size=15, concurrent_batches=3))
existing_articles = rss.load_existing_articles()
print(len(existing_articles))
new_articles = rss.save_new_articles(existing_articles, all_articles)
print(len(new_articles))
with open ("extracted_RSS.json", "w") as f:
    json.dump(new_articles, f)
