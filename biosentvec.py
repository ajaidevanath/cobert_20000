import matplotlib.pyplot as plt
from sklearn.cluster import KMeans,DBSCAN,AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')
import string
from cluster import doc_cluster,article_cluster

import sent2vec 
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from scipy.spatial import distance
from nltk.corpus import stopwords
 
stop_words = stopwords.words('english')

import nltk
nltk.download('punkt')
nltk.download('stopwords')

# loading the biosent vec weights
model_path = '/Users/ajai_devanathan/Desktop/project_ir/Biosentvec/BioSentVec_PubMed_MIMICIII-bigram_d700.bin'
model = sent2vec.Sent2vecModel()
try:
    model.load_model(model_path)
except Exception as e:
    print(e)
print('model successfully loaded')


def preprocess_sentence(text):
    tokens = [token for token in word_tokenize(text) if token not in string.punctuation and token not in stop_words]
    return ' '.join(tokens)

sentence = preprocess_sentence('while having coronoavirus it is better to wash hands')
print(sentence)

sentence_vector = model.embed_sentence(sentence)
print(sentence_vector)

# Reading the dataset
df = pd.read_json('~/Desktop/project_ir/CORD-NER-full.json',lines=True)

# Filling all the portions where are this no body
for i in range(len(df)):
    if df['body'][i] =='':
        df['body'][i] = df['abstract'][i]
df = df[df['body']!='']

df.reset_index(inplace=True)

# entitity list and bar graph

entity =[]
for i in range(0,df.shape[0]):
    for j in range(0,len(df.entities[i])):
         entity.append(df.entities[i][j]['type'])

entity = pd.Series(entity)
entity = entity.value_counts()
entity 

# plotting the bar graph of the entities and counts/frequencies
plt.figure(figsize=(20,10))
plt.bar(entity.index[:10],height=entity.values[:10],color='red')
plt.xticks(rotation=45)
plt.show()

# we will need to extract the sentences from the text column and then embed them using the model
articles_cluster = []
for i in range(0,df.shape[0]):
    articles_cluster.append(model.embed_sentence(preprocess_sentence(df.title[i])))

articles_vector = np.array(articles_cluster)#convert the articles cluster to an numpy array

df_articles = pd.DataFrame(index = np.arange(df.shape[0]),columns=np.arange(0,articles_vector.shape[2]))
df_articles.shape

for i in range(df.shape[0]):
    df_articles.iloc[i] = articles_vector[i]

wcss = []
kmeans = KMeans(n_clusters=15, max_iter=300, n_init=10, random_state=0)
for i in range(1,15):
    kmeans.set_params(n_clusters=i)
    kmeans.fit(df_articles)
    wcss.append(kmeans.inertia_)

# elbow plot to check the optimal number of clusters
plt.plot(wcss)
plt.xlim(0,)

# fitting the model
kmeans = KMeans(n_clusters=12, max_iter=300, n_init=10, random_state=0)
kmeans.fit(df_articles)

# function for prediction of any query
def cluster_precict(x):
    x = np.array(x,dtype=np.float)
    pred = kmeans.predict(x)
    return pred 
cluster_precict(sentence_vector)# testing a prediction

label = kmeans.fit_predict(df_articles)
df_articles['label'] = label
df_articles.to_csv('article_title_final.csv')

# K nearest Neighbours
df_articles_knn = df_articles.drop(['label'], axis =1)
df_articles_knn_target = df_articles['label']
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=15,metric='cosine')
knn.fit(df_articles_knn,df_articles_knn_target)






