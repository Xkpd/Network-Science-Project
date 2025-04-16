import os
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup

# Paths
scientists_file = './output/scientists_cleaned.csv'
papers_test_output = 'papers_test.csv'

# Load only a few scientists for testing
print("🔍 Test mode: Loading first 2 scientists for debugging...")
if not os.path.exists(scientists_file):
    print(f"❌ Error: {scientists_file} not found.")
    exit()

scientists_df = pd.read_csv(scientists_file).head(2)

papers_data = []

# Scrape each test scientist
for _, row in scientists_df.iterrows():
    pid = row['pid']
    url = row['final_url']
    print(f"\n🧪 Testing: {pid} | {url}")

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        entries = soup.find_all('li', class_=lambda x: x and x.startswith('entry'))

        for entry in entries[:5]:  # Limit to 5 publications for brevity
            title_tag = entry.find('span', class_='title')
            year_tag = entry.find('span', class_='year')
            doi_tag = entry.find('a', title='DOI')
            author_tags = entry.find_all('span', itemprop='author')

            title = title_tag.text.strip() if title_tag else 'N/A'
            # Try to find the <span class="year">YYYY</span>
            year_tag = entry.find('span', class_='year')
            if year_tag:
                year = year_tag.text.strip()
            else:
                # fallback: find year at end of citation block
                year_match = re.search(r'(\b20\d{2}\b|\b19\d{2}\b)', entry.text)
                if year_match:
                    year = year_match.group(1)
            # Get DOI
            doi_tag = entry.find('a', href=re.compile(r'(doi\.org|arxiv\.org)'))
            doi = doi_tag['href'].strip() if doi_tag else 'N/A'

            authors = ', '.join([a.text.strip() for a in author_tags]) if author_tags else 'N/A'

            print(f"📄 Title: {title}")
            print(f"📅 Year: {year}")
            print(f"🔗 DOI: {doi}")
            print(f"👥 Authors: {authors}")
            print("---")

            papers_data.append({
                'Title': title,
                'Year': year,
                'DOI': doi,
                'Authors': authors,
                'file': f"{pid}.xml"
            })

    except Exception as e:
        print(f"❌ Error for {pid}: {e}")
        continue

# Save test output
papers_df = pd.DataFrame(papers_data)
papers_df.to_csv(papers_test_output, index=False)
print(f"\n✅ Test complete. Saved {len(papers_df)} papers to {papers_test_output}")
