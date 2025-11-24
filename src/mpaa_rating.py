import pandas as pd
import requests
import time
import re
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_KEY = os.getenv("TMDB_API_KEY")  # Get from https://www.themoviedb.org/settings/api
INPUT_FILE = './data/movies.dat'
OUTPUT_FILE = './data/movies.dat'

# Load MovieLens 1M movies.dat
print("Loading movies.dat...")
movies = pd.read_csv(
    INPUT_FILE,
    sep='::',
    engine='python',
    header=None,
    names=['MovieID', 'Title', 'Genres'],
    encoding='latin-1'
)

# Extract year and clean title
print("Parsing titles and years...")
movies['Year'] = movies['Title'].str.extract(r'\((\d{4})\)')
movies['CleanTitle'] = movies['Title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()

# Fetch MPAA ratings from TMDb API
def get_mpaa_rating(title, year, api_key):
    """Fetch MPAA rating from TMDb API"""
    # Search for movie
    search_url = f"https://api.themoviedb.org/3/search/movie"
    search_params = {
        'api_key': api_key,
        'query': title,
        'year': year
    }
    
    try:
        # Get movie search results
        response = requests.get(search_url, params=search_params, timeout=5)
        data = response.json()
        
        if not data.get('results'):
            return 'N/A'
        
        # Get first result's ID
        movie_id = data['results'][0]['id']
        
        # Fetch release info for US certification
        release_url = f"https://api.themoviedb.org/3/movie/{movie_id}/release_dates"
        release_params = {'api_key': api_key}
        release_response = requests.get(release_url, params=release_params, timeout=5)
        release_data = release_response.json()
        
        # Find US certification (MPAA rating)
        for country in release_data.get('results', []):
            if country['iso_3166_1'] == 'US':
                for release in country['release_dates']:
                    if release.get('certification'):
                        return release['certification']
        
        return 'N/A'
        
    except Exception as e:
        return 'N/A'

# Add MPAA ratings with progress tracking
print(f"Fetching MPAA ratings for {len(movies)} movies...")
movies['MPAA_Rating'] = None

for idx in tqdm(range(len(movies)), desc="Fetching ratings"):
    row = movies.iloc[idx]
    movies.at[idx, 'MPAA_Rating'] = get_mpaa_rating(
        row['CleanTitle'], 
        row['Year'], 
        API_KEY
    )
    
    # Rate limiting (TMDb allows 40 requests per 10 seconds)
    time.sleep(0.3)

# Save updated dataset back to movies.dat with :: delimiter
print(f"\nSaving to {OUTPUT_FILE}...")
with open(OUTPUT_FILE, 'w', encoding='latin-1') as f:
    for idx, row in movies.iterrows():
        line = f"{row['MovieID']}::{row['Title']}::{row['Genres']}::{row['MPAA_Rating']}\n"
        f.write(line)

print("Done!")
print(f"\nRating distribution:")
print(movies['MPAA_Rating'].value_counts())