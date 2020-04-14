#%% [markdown]
# # Create your own horror movie!!
# ### Create a automatically generated horror movie screenplay,  look for movies from the public domain, and automatically let this script edit a new film.
#%% [markdown]
# ## Pre-production and screenplay  writing
#%% [markdown]
# ### Let's start by web scrapping horror movie screenplays
#%%
import requests
from bs4 import BeautifulSoup

SCREENPLAYS_URLS = ['http://www.horrorlair.com/moviescripts_a_f.html',
'http://www.horrorlair.com/moviescripts_g_n.html', 'http://www.horrorlair.com/moviescripts_o_z.html']

def scrape_screenplays(urls=SCREENPLAYS_URLS):
    for url in urls:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        print()
        #results = soup.find(id='ResultsContainer')
        #print(results.prettify())

scrape_screenplays(urls=['http://www.horrorlair.com/moviescripts_a_f.html'])

