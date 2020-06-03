
#https://maksimekin.github.io/COVID19-Literature-Clustering/COVID19_literature_clustering.html

import numpy as np
import pandas as pd
import glob
import json
import time
import string
import re

import nltk
from nltk.corpus import stopwords
stop = set(stopwords.words('english'))

start = time.time()

import matplotlib.pyplot as plt
plt.style.use('ggplot')

root_path  = 'data'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype = {
                        'pubmed_id': str,
                        'Microsoft Academic Paper ID' : str,
                        'doi' : str})

all_json = glob.glob(f'{root_path}//**/*.json', recursive = True)
#print(len(all_json))
all_json = all_json[:20000]

# helper class to read the file
class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            for entry in content['abstract']:
                self.abstract.append(entry['text'])
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
first_row = FileReader(all_json[0])
print(first_row)

 # add in helper function
def get_breaks(content, length):
    data = ""
    words = content.split(' ')
    total_chars = 0

    # fix in the breaks
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            data = data + "<br>"  + words[i]
            total_chars = 0
        else:
            data = data + " " + words[i]
    return data


print(all_json[0])

# create an array for each of the keys in the dict
dict_ = {'paper_id' : [], 'doi' : [], 'abstract' : [], 'body_text' :[],
        'authors' : [] , 'title' :  [], 'journal' : [], 'abstract_summary' : []}

for i, entry in enumerate(all_json):
    # generate a print out - creates a class for each instance
    if i % (len(all_json) // 10) == 0:
        print(f' processing : {i} of {len(all_json)}')

    try: # processes each json entry
        content = FileReader(entry)
    except Exception as e:
        print(str(e))
        continue # invalid file format

    # get the metadata
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    # no metadata throw a print and move on
    if len(meta_data) == 0:
        #print("no meta here - skipping")
        continue

    dict_['abstract'].append(content.abstract)
    dict_['paper_id'].append(content.paper_id)
    dict_['body_text'].append(content.body_text)

    # generate a new column for the abstract
    if len(content.abstract) == 0:
        dict_['abstract_summary'].append("Not provided")
    elif len(content.abstract.split(' ')) > 100:
        info = content.abstract.split(' ')[:100]
        summary = get_breaks(' '.join(info), 40 )
        dict_['abstract_summary'].append(summary + "...")

    else:
        # abstract is long enough
        summary = get_breaks(content.abstract, 40 )
        dict_['abstract_summary'].append(summary)

    # get metadata
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    try:
        # parse additional authors
        authors = meta_data['authors'].values[0].split(';')
        if len(authors) > 2:
            dict_['authors'].append(get_breaks('. '.join(authors), 40))
        else:
            # authors fit in the plit
            dict_['authors'].append(". ".join(authors))
    except Exception as e:
            # only one author - or NaN
            dict_['authors'].append(meta_data['authors'].values[0])
    # add the title information , add break when needed
    try:
        title = get_breaks(meta_data['title'].values[0], 40 )
        dict_['title'].append(title)
    except Exception as e: # title absent
        dict_['title'].append(meta_data['title'].values[0])

    # add the journal information
    dict_['journal'].append(meta_data['journal'].values[0])

    #add doi
    dict_['doi'].append(meta_data['doi'].values[0])

df_covid = pd.DataFrame(dict_, columns= ['paper_id', 'doi', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary'])
print(df_covid.head())



print(df_covid.head())
print(df_covid.info())

print(df_covid['abstract'].describe(include = 'all'))

# document set comes with duplicates
df_covid.drop_duplicates(['abstract', 'body_text'], inplace=True)
print(df_covid['abstract'].describe(include='all'))
print(df_covid['body_text'].describe(include = "all"))


# covert to a similar maroner
df = df_covid
del df_covid

print(df.head())
print(df.columns)

def createVSM(doc):
    # remove punctuation
    removeSyms = string.punctuation
    pattern = r"[{}]".format(removeSyms)
    doc = re.sub(pattern, " ", doc.strip().lower())

    # cast to a VSM
    doc = doc.split()
    return doc

# this method takes the text and cast it to VSM
all_rows = []
for row in df['body_text']:
    row = createVSM(row)
    all_rows.append(row)
    #print(row)


df['body_text'] = all_rows
# drops all duplicate instances
df.dropna(inplace=True)

# method removes stopwords
def removeStopWords(row):
    # stop gap measure to retain terms method etc
    stop = set(stopwords.words('english'))

    # remove all unnecessary stop words
    custom_stop_words = [
        'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure',
        'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'medrxiv', 'license', 'fig', 'fig.',
        'al.', 'Elsevier', 'PMC', 'CZI', 'www' , "http"
    ]
    # generic list of stop terms
    stop = list(stop)
    stop = stop + custom_stop_words

    row = [ word for word in row if word not in stop]

    row =  " ".join(row)
    return row

# calls expression and removes stoppies
# in this method we lazily convert back to string
# for a more robuts approach organise properly
df['body_text'] = df['body_text'].apply(removeStopWords)

print(df['body_text'][0])
print(len(df['body_text'][0].split(" ")))
# convert the text to tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
def vectorize(text , maxx_feature):
    vectorizer = TfidfVectorizer(max_features = maxx_feature)
    X = vectorizer.fit_transform(text)
    return X

text = df['body_text'].values
# converts the text into a statistical representation
X = vectorize(text, 2 **12)
#
from sklearn.cluster import KMeans
# make the assumption that there are 20 clusters
k = 5
kmeans = KMeans(n_clusters= k)
y_pred = kmeans.fit_predict(X.toarray())
df['y']= y_pred


# plot the result to TSNE
from sklearn.manifold import TSNE
tsne = TSNE(verbose = 1, perplexity = 100, random_state = 42)
X_embedded = tsne.fit_transform(X.toarray())

# plot that mother
from matplotlib import pyplot as plt
import seaborn as sns
# sns.set(rc= {"figure.figsize": (15,15)})
# palette = sns.color_palette("bright", 1)
# sns.scatterplot(X_embedded[:,0], X_embedded[:,1], palette = palette)
# plt.title("tsne without labels")
# plt.savefig("tsneCovid.png")
# plt.show()

sns.set(rc={'figure.figsize': (15,15)})

# colours
palette = sns.hls_palette(len(list(set(y_pred))), l=.4, s=.9 )

# #plot
# print(len(y_pred))
# print(len(X_embedded))
# sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue = y_pred, legend = 'full', palette=palette)
# plt.title('tsne with kmeans Labels')
# plt.savefig('plots/imporve_cluster_tsne.png')
# plt.legend(labels=["a","a","a","a","a"])
# plt.show()


from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


vectorizers = []
for ii in range(0, k):
    # create vectorizer
    vectorizers.append(CountVectorizer(min_df = .1, max_df = 0.9, stop_words = 'english'
                                        , lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}'))

print(df.columns)
#vectorisse data per cluster
vectorized_data = []
for current_cluster, cvec in enumerate(vectorizers):
    try:
        vectorized_data.append(cvec.fit_transform(df.loc[df['y'] == current_cluster, 'body_text']))
    except:
        print("not enough instances of clusters " + str(current_cluster))
        vectorized_data.append(None)

print(len(vectorized_data))

NUM_TOPICS_PER_CLUSTER = k
lda_models = []
for ii in range(0, k):
    # build lda
    lda = LatentDirichletAllocation(n_components = NUM_TOPICS_PER_CLUSTER,
                                        max_iter=10, learning_method = 'online',
                                        verbose= False, random_state = 42)
    lda_models.append(lda)

print(lda_models[0])

# apply the lda model to each of the clustering
clusters_lda_data = []
for current_cluster, lda in enumerate(lda_models):
    print("current_cluster: " + str(current_cluster))
    if vectorized_data[current_cluster] != None:
        clusters_lda_data.append((lda.fit_transform(vectorized_data[current_cluster])))

# extract keywords per cluster
# function printing keywords for each cluster
def selection_topics(model, vectorizer, top_n=3):
    current_words = []
    keywords = []
    for idx, topic in enumerate(model.components_):
        words  = [(vectorizer.get_feature_names()[i], topic[i]) for i in
                                                topic.argsort()[:-top_n - 1:-1]]
        for word in words:
            if word[0] not in current_words:
                keywords.append(word)
                current_words.append(word[0])
    keywords.sort(key = lambda x : x[1])
    keywords.reverse()
    return_values = []
    for ii in keywords:
        return_values.append(ii[0])
    return return_values

all_keywords = []
for current_vectorizer, lda in enumerate(lda_models):
    print("current cluster: " + str(current_vectorizer))

    if vectorized_data[current_vectorizer] != None:
        all_keywords.append(selection_topics(lda, vectorizers[current_vectorizer]))

print(all_keywords[0][:10])

i = 0
for array in all_keywords:
    print(array[:5])
    print(i)
    i = i + 1

print(type(all_keywords))
print(len(all_keywords))
print(all_keywords)
print(len(y_pred))
print(len(X_embedded))
sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue = y_pred, legend = 'full', palette=palette)
plt.title('tsne with kmeans Labels')
plt.savefig('plots/imporve_cluster_tsne2.png')
plt.legend(labels=[all_keywords[0][:5],all_keywords[1][:5],all_keywords[2][:5],all_keywords[3][:5],all_keywords[4][:5]])
plt.show()




print(10*"**")
print(time.time() - start)
