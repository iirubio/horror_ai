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
from imdb import IMDb # documentation on https://imdbpy.github.io/


file_names = []
urls = ['https://www.imsdb.com/genre/Horror']
for url in urls:
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    for link in soup.find_all('a'):
        file_link = link.get('href')
        print(file_link)

#%%
import requests
import urllib.request
import os
from bs4 import BeautifulSoup

SCREENPLAYS_URLS = ['https://www.simplyscripts.com/genre/horror-scripts.html']
FILE_TYPES = ['.pdf', '.txt']
https://www.imsdb.com/genre/Horror
# Folder with screenplays
SCREENPLAYS_FOLDER = 'screenplays'
if not os.path.exists(SCREENPLAYS_FOLDER):
    os.mkdir(SCREENPLAYS_FOLDER)

def scrape_screenplays(urls=SCREENPLAYS_URLS, screenplays_folder=SCREENPLAYS_FOLDER):
    """Saves screenplays into .txt and .pdf files and returns file names

    Keyword Arguments:
        urls {string} -- [Urls to scrap] (default: {SCREENPLAYS_URLS})
        screenplays_folder {string} -- [Path to files] (default: {SCREENPLAYS_FOLDER})

    Returns:
        [list] -- [List of file names]
    """
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
# ### We continue to parse the names and get more info about the movies using IMDB's API
# ### Preliminary inspection shows a pattern '%20' that should be replaced with a blank space

#%%
import pandas as pd
from imdb import IMDb # documentation on https://imdbpy.github.io/

imdb_object = IMDb()


files_data = []
for file_name in file_names:
    movie_name = file_name[:-4].replace('%20', '') # Erase extension and %20 characters
    print(movie_name)
    movie_list = imdb_object.search_movie(movie_name)
    title = movie_name
    rating = None
    year = None #TODO: TURN INTO INT

    if movie_list:
        movie_id = movie_list[0].getID() # Assumes first movie on search is the correct one
        movie = imdb_object.get_movie(movie_id)
        # TODO: REFACTOR
        title = movie['title'] if movie.has_key('title') else movie_name
        rating = movie['rating'] if movie.has_key('rating') else None
        year = movie['year'] if movie.has_key('year') else None

    files_data.append({'file_name': file_name,
                        'title': title,
                        'rating': rating,
                        'year': year
                        })

# Save dataframe to csv
movies_df = pd.DataFrame(files_data)
movies_df.to_csv('screenplays_extra_data.csv') #TODO: Refactor screenplays

#%% [markdown]
# ### We filter further our new dataset (and do visualizations)
# ### We found a significant negative relation between year of publication and rating
# ### Reasons for it are not clear given the small dataset, it could be that only "bad"
# ### recent horror movies have their scripts online, unlike older movies.
#%%
# Read screenplay files from local folder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

movies_df = pd.read_csv('screenplays_extra_data.csv')
movies_df = movies_df.dropna() # Drop if any row value is NaN
#%%
# Ratings
plt.xlim(1, 10)
plt.xlabel('Rating')
plt.ylabel('Freq')
plt.title('Ratings histogram')
movies_df['rating'].hist()

#%%
# Year
plt.xlabel('Year')
plt.ylabel('Freq')
plt.title('Year histogram')
movies_df['year'].hist()
#%%
# Relationship between year and rating (in my small dataset of films with script)
import scipy.stats
ax = sns.regplot(x='year', y='rating', data=movies_df)
scipy.stats.linregress(movies_df['year'], movies_df['rating'])

#%% [markdown]
# ### A recurrent neural network is trained with the screenplays, to generate a new one
#%%
import tensorflow as ts
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_names = movies_df['file_name']
for file_name in file_names:
        # Read, then decode for py2 compat.
    text = open(SCREENPLAYS_FOLDER + '//' + file_name, 'rb').readlines()
    for line in text:
        print(line)
    # length of text is the number of characters in it
    #print ('Length of text: {} characters'.format(len(text)))
    #if 'bool(BeautifulSoup(text, "html.parser").find())': #TODO: A REGEX
        #print(file_name)





# %%
