
## **Topic Modeling**





# Most of codes are from the link.
!pip install spacy-langdetect
!pip install language-detector
!pip install symspellpy
!pip install sentence-transformers
!pip install stop-words

# linear algebra
# data processing, CSV file I/O (e.g. pd.read_csv)
import os

import json
import pandas as pd
from tqdm import tqdm
import numpy as np
from nltk.corpus import wordnet
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
# import umap.umap_ as umap

# Commented out IPython magic to ensure Python compatibility.
import gensim
from gensim.utils import simple_preprocess
from gensim import corpora, models
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

import datetime
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
# %matplotlib inline
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib inline
from plotly.offline import plot
import plotly.graph_objects as go

import nltk

"""## **Pre-Trained Model**"""

import pandas as pd
import pickle
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', category=Warning)

import argparse

#def model(): #:if __name__ == '__main__':

method = "LDA_BERT"
ntopic = 6
    #parser = argparse.ArgumentParser(description='contextual_topic_identification tm_test:1.0')
    #parser.add_argument('--fpath', default='/kaggle/working/train.csv')
    #parser.add_argument('--ntopic', default=10,)
    #parser.add_argument('--method', default='TFIDF')
    #parser.add_argument('--samp_size', default=20500)
    #args = parser.parse_args()
data = meta
    #pd.read_csv('/kaggle/working/train.csv')
data = data.fillna('')  # only the comments has NaN's
rws = data['Submission_HTML_Removed_coding'] # csv sentense column
samp_size = len(rws)
sentences, token_lists, idx_in = preprocess(rws, samp_size=samp_size, replace=False) 
tm = Topic_Model(k = ntopic, method = 'LDA_BERT') # pre-training
    # Fit the topic model by chosen method
tm.fit(sentences, token_lists)
    # Evaluate using metrics
#with open("/kaggle/working/{}.file".format(tm.id), "wb") as f:
  #    pickle.dump(tm, f, pickle.HIGHEST_PROTOCOL)
print('Coherence:', get_coherence(tm, token_lists, 'c_v'))
#  print('Silhouette Score:', get_silhouette(tm))
    # visualize img

"""##  Preprocessing and Predicting with the pre-trained model"""

stc= rws.apply(lambda x: preprocess_sent(x)) # preprocessing sentences
tokens = stc.apply(lambda x: preprocess_word(x)) # tokenizing

"""Saving the model"""

import pickle

filename = 'finalized_LDAmodel.sav'
pickle.dump(tm, open(filename, 'wb'))

data_sentence=[x for x in stc if x is not None]
data_tokens =[x for x in tokens if x is not None]
# Empty sentence and tokens are eliminated.

"""Load the model"""

loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(data_sentence, data_tokens, out_of_sample=True)
print(result)


# We might use this later to deal with empty array for sentences and tokens.

topicnames = ["Topic"]
docnames = ["Doc" + str(i) for i in range(len(result))]
df_document_topic = pd.DataFrame(np.round(result, 2), columns = topicnames, index = docnames)
print(df_document_topic)

"""Sorting topic counts"""

# Descending order
unsorted = [0] * ntopic

for topic in df_document_topic["Topic"]:
  unsorted[topic] = unsorted[topic] + 1
print(unsorted)
topic_distribution = sorted(unsorted,reverse=True)
print(topic_distribution)

xindex = []
for i in topic_distribution:
  for j in range(ntopic):
    if i == unsorted[j]:
      print()
      xindex.append(j)
      continue
print(xindex)
print(topic_distribution)

"""## **Visualization for our models**"""

plt.figure(figsize=(20,6))
x = np.arange(len(xindex))
plt.title('No. of comments vs. topics', fontsize = 14)
plt.ylabel('Number of comments', fontsize = 14)
plt.xlabel('Topics', fontsize = 14)
plt.bar(x, topic_distribution)
plt.xticks(x, xindex)
plt.show()
print(topic_distribution)

"""**pyLDAvis**"""

!pip install pyLDAvis
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# feed the LDA model into the pyLDAvis instance
lda_out = gensim.models.ldamodel.LdaModel(corpus=tm.corpus, num_topics=6, id2word=tm.dictionary, random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=6,
                                           alpha='auto',
                                           per_word_topics=True)
doc_lda = lda_out[tm.corpus]

pyLDAvis.enable_notebook()
lda_viz = gensimvis.prepare(lda_out, tm.corpus, tm.dictionary)

#lda_model.fit_transform(data_vectorized)

def format_topics_sentences(ldamodel=tm.ldamodel, corpus=corpus, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=tm.ldamodel, corpus=tm.corpus, texts=data_sentence)

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic.head(100)

count = [0] * ntopic
for topic in df_document_topic["Topic"]:
  count[topic] = count[topic] + 1
print(count)

topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
