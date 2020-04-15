#%% [markdown]
# # Create your own horror movie!!
# ### Create a automatically generated horror movie screenplay,  look for movies from the public domain, and automatically let this script edit a new film.
#%% [markdown]
# ## Pre-production and screenplay  writing
#%% [markdown]
# ### Let's start by web scraping horror movie screenplays
#%%
import requests
import urllib.request
import os
from bs4 import BeautifulSoup

SCREENPLAYS_URLS = ['https://www.simplyscripts.com/genre/horror-scripts.html']
FILE_TYPES = ['.pdf', '.txt']

# Folder with screenplays
SCREENPLAYS_FOLDER = 'screenplays'
if not os.path.exists(SCREENPLAYS_FOLDER):
    os.mkdir(SCREENPLAYS_FOLDER)

def scrape_screenplays(urls=SCREENPLAYS_URLS, screenplays_folder=SCREENPLAYS_FOLDER):
    file_names = []
    for url in urls:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        for link in soup.find_all('a'):
            file_link = link.get('href')
            try:
                if any(extension in file_link for extension in FILE_TYPES):
                    file = requests.get(file_link)
                    file_name = file_link.split('/')[-1]
                    file_names.append(file_name)
                    with open(screenplays_folder + '//' + file_name, 'wb') as f:
                        f.write(file.content)
            except Exception as identifier:
                print(identifier)
    return file_names

file_names = scrape_screenplays()

#%% [markdown]
# ### We continue by training a recurrent neural network with our data
#%%
# Read screenplay files from local folder
import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


for file_name in file_names:
    with open(SCREENPLAYS_FOLDER + '//' + file_name, 'rb') as f:
        print()




# %%
