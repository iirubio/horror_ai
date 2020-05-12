#%%
# # Create your own movie!!
# ### Create a automatically generated horror movie screenplay, look for movies from the public domain, and automatically let this script edit a new film.
# ### This is the first part, pre-production and screenplay writing!

#%%
# ## Production, filming (aka downloading youtube videos and segmenting scenes)
# ### Public domain horror movies: https://www.youtube.com/playlist?list=PL2P-7ibfKhkuAkVr0P7bXPYQAri353k-S
#%%

#%%
from pytube import YouTube
YouTube('https://youtu.be/9bZkp7q19f0').streams.get_highest_resolution().download()

yt = YouTube('http://youtube.com/watch?v=9bZkp7q19f0')
yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution')[-1].download()