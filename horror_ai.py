#%%
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

SCREENPLAYS_URLS = ['https://www.simplyscripts.com/genre/horror-scripts.html',
                    'https://www.simplyscripts.com/genre/musical-scripts.html']
FILE_TYPES = ['.pdf', '.txt']

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
    movie_count = 0
    for url in urls:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        for link in soup.find_all('a'):
            file_link = link.get('href')
            try:
                if any(extension in file_link for extension in FILE_TYPES):
                    file = requests.get(file_link)
                    if '<html>' not in file:
                        file_name = file_link.split('/')[-1]
                        file_names.append(file_name)
                        with open(screenplays_folder + '//' + file_name, 'wb') as f:
                            f.write(file.content)
                            movie_count += 1
                            print(str(movie_count))
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


def get_imdb_info(file_names, data_labels=['title', 'rating', 'year', 'genre']):
    imdb_object = IMDb()
    files_data = []
    for file_name in file_names:
        movie_name = file_name[:-4].replace('%20', '') # Erase extension and %20 characters
        print(movie_name)
        movie_list = imdb_object.search_movie(movie_name)
        imdb_info = {'file_name': file_name, 'title': None, 'rating': None, 'year': None, 'genre': None}


        if movie_list:
            movie_id = movie_list[0].getID() # Assumes first movie on search is the correct one
            movie = imdb_object.get_movie(movie_id)
            for key, value in imdb_info.items():
                value = movie[key] if movie.has_key(key) else None

        files_data.append(imdb_info)
    return files_data

def get_file_info(file_names, screenplays_folder=SCREENPLAYS_FOLDER):
    has_html_column = []
    is_txt = []
    for file_name in file_names:
        text_has_html = False
        text = open(screenplays_folder + '//' + file_name, 'rb').readlines()

        for line in text:
            if '<html>' in str(line):
                text_has_html = True
                break
        is_txt.append(True if 'txt' in file_name else False)
        has_html_column.append(text_has_html)
    return has_html_column, is_txt


#%%

files_data = get_imdb_info(file_names=file_names)
#%%

has_html_column, is_txt = get_file_info(file_names=file_names)


# # # Save dataframe to csv
movies_df = pd.DataFrame(files_data)
movies_df['has_html'] = has_html_column
movies_df['is_txt'] = is_txt
movies_df.to_csv('screenplays_data3.csv') #TODO: Refactor screenplays

#%%
#movies_df.to_csv('screenplays_extra_data2.csv')



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
#%%
movies_df = movies_df.dropna() # Drop if any row value is NaN
# #%%
# # Ratings
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

SCREENPLAYS_FOLDER = 'screenplays'

file_names = movies_df[(~movies_df['has_html']) & (movies_df['is_txt'])]['file_name']

#%%
# Scripts into one string variable
SCREENPLAYS_FOLDER = 'screenplays'

def create_training_dataset(movies_df, scripts_by_genre=5, genres=['horror']):
    # 5 scripts of each genre for RNN to learn from

    all_scripts_text = ''
    for index, file_name in file_names.iteritems():
        print(file_name)
        text = str(open(SCREENPLAYS_FOLDER + '//' + file_name, 'rb').read().decode('utf-8', 'ignore')).encode("utf-8")
        print(text)
        all_scripts_text += str(text) + ' '
        print ('Length of text: {} characters'.format(len(text)))

    len(all_scripts_text)

#%%

# Read, then decode for py2 compat.
# TODO: TURN IT INTO STRING HERE
text = open(SCREENPLAYS_FOLDER + '//' + test_file, 'rb').read().decode('utf-8','ignore').encode("utf-8")
# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))


# %%
# The unique characters in the file
vocab = sorted(set(str(text)))
#%%
# RNN
# Vectorize text
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
#%%
text_as_int = np.array([char2idx[c] for c in str(text)])

#%%


#%%
import tensorflow as tf
# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)
#%%
# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

#%%
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))


#%%
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

#%%
for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
  print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

#%%
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))


# %%
# Batch size
BATCH_SIZE = 64

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset

# %%
# Build the model
# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

#%%
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)


#%%
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


# %%
model.summary()


# %%
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()


# %%
sampled_indices

# %%print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))

print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))


# %%
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())


# %%
model.compile(optimizer='adam', loss=loss)
#%%
import os
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

#%%
EPOCHS=10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


# %%
# Generate text
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))


# %%
def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 2.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


# %%
print(generate_text(model, start_string=u"ANDY"))


# %%
d = {'col1': [1, 2], 'col2': ['hola', 'jaja']}

df = pd.DataFrame(data=d)

# %%
