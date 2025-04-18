import os
import re
import time
import requests
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial

# ========== File Paths ==========
input_file = './input/datascientists.xls'
scientists_output_file = './output/scientists_cleaned.csv'
papers_output_file = 'papers.csv'
error_log_file = 'error_log.txt'

# ========== Step 1: Load or Generate Cleaned Scientist Data ==========

if os.path.exists(scientists_output_file):
    print(f"Found existing '{scientists_output_file}', loading it...")
    scientists_df = pd.read_csv(scientists_output_file)
else:
    print("'scientists_cleaned.csv' not found. Starting from raw input...")

    initial_df = pd.read_excel(input_file)
    print("Collecting final DBLP URLs and PIDs...")
    pids = []
    final_urls = []
    errors_links = []

    for link in tqdm(initial_df['dblp']):
        while True:
            response = requests.get(link)
            if response.status_code == 429:
                print('Too many requests. Sleeping for 60 seconds...')
                time.sleep(60)
            else:
                break

        if response.status_code != 200:
            pids.append('Error')
            final_urls.append('Error')
            errors_links.append(link)
            continue

        final_url = response.url
        match = re.search(r'pid/(.*).html', final_url)

        if match:
            pid = match.group(1).replace('/', '-')
            pids.append(pid)
            final_urls.append(final_url)
        else:
            pids.append('Error')
            final_urls.append('Error')
            errors_links.append(link)

    # Save cleaned data
    cleaned_df = initial_df.copy()
    cleaned_df['pid'] = pids
    cleaned_df['final_url'] = final_urls
    cleaned_df = cleaned_df[(cleaned_df['pid'] != 'Error') & (cleaned_df['final_url'] != 'Error')]
    cleaned_df.to_csv(scientists_output_file, index=False)
    print(f"Saved {len(cleaned_df)} cleaned scientists to {scientists_output_file}")
    scientists_df = cleaned_df

# ========== Step 2: Scrape Papers (Parallelized) ==========

def scrape_scientist(row):
    papers = []
    pid = row['pid']
    url = row['final_url']

    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        entries = soup.find_all('li', class_=lambda x: x and x.startswith('entry'))

        for entry in entries:
            title_tag = entry.find('span', class_='title')
            year_tag = entry.find('span', class_='year')
            doi_tag = entry.find('a', title='DOI')
            author_tags = entry.find_all('span', itemprop='author')

            title = title_tag.text.strip() if title_tag else 'N/A'

            # Extract year (fallback to regex)
            if year_tag:
                year = year_tag.text.strip()
            else:
                year_matches = re.findall(r'\b(19\d{2}|20\d{2})\b', entry.text)
                valid_years = [int(y) for y in year_matches if int(y) <= 2025]
                if not valid_years:
                    with open("missing_year_fallback.txt", "a", encoding="utf-8") as log_file:
                        log_file.write(f"No valid year for entry in {pid}:\n{entry.text[:300]}\n\n")
                year = str(max(valid_years)) if valid_years else 'N/A'


            doi_tag = entry.find('a', href=re.compile(r'(doi\.org|arxiv\.org)'))
            doi = doi_tag['href'].strip() if doi_tag else 'N/A'

            authors = ', '.join([a.text.strip() for a in author_tags]) if author_tags else 'N/A'

            papers.append({
                'Title': title,
                'Year': year,
                'DOI': doi,
                'Authors': authors,
                'file': f"{pid}.xml"
            })

    except Exception as e:
        with open(error_log_file, 'a', encoding='utf-8') as f:
            f.write(f"Error for {pid} at {url}: {e}\n")

    return papers

if __name__ == '__main__':
    print("Scraping publications with multiprocessing...")

    with Pool(cpu_count()) as pool:
        all_papers_nested = list(tqdm(pool.imap(scrape_scientist, [row for _, row in scientists_df.iterrows()]), total=len(scientists_df)))

    # Flatten the nested list of lists
    all_papers = [paper for sublist in all_papers_nested if sublist for paper in sublist]

    # ========== Step 3: Save Results ==========
    papers_df = pd.DataFrame(all_papers, columns=['Title', 'Year', 'DOI', 'Authors', 'file'])
    papers_df.to_csv(papers_output_file, index=False)
    print(f"Saved {len(papers_df)} papers to {papers_output_file}")
    print(f"Errors (if any) logged in {error_log_file}")

