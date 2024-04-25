# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 17:05:05 2022

@author: pecco
"""

# MODELLING_DATA_WEL HA NLP, BOW E EMOZIONI, MANCA SOLO TOPIC

from nrclex import NRCLex
import pandas as pd
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack
import gensim.corpora as corpora
from gensim.test.utils import common_texts
from gensim.utils import simple_preprocess
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
import gensim
from nltk.stem import WordNetLemmatizer
import random

df = pd.read_csv("Welfake_prData.csv")
df_train = pd.read_csv('train_wel.csv')
df_test = pd.read_csv('test_wel.csv')

stop_words = set(stopwords.words('english'))

tokens = []
for i in df_train['processed_text']:
    word_tokens = word_tokenize(i)
    tokens.append(word_tokens)
    
tokens_all = []
for i in df['processed_text']:
    word_tokens = word_tokenize(i)
    tokens_all.append(word_tokens)
    

# EMOZIONI PRIMARIE E SENTIMENT
emotion_list = []
for i in df['processed_text']:
    emotion = NRCLex(i)
    emotion_list.append(emotion.affect_frequencies)


em_dict = pd.DataFrame(emotion_list)
em_dict1 = em_dict.drop(columns='anticip')

new_list = pd.DataFrame()
for i in em_dict1:
    new_list = pd.concat([new_list,em_dict1[i].fillna(0)], axis=1)
    
new_list = pd.concat([new_list, df['label']], axis=1)
new_list.to_csv('emotions_data_wel.csv', index=False)


### LATENT DIRICHLET ALLOCATION 
lem = WordNetLemmatizer()
train_data_lda = []
for i in tokens:
    new_lem = []
    for words in i:
        new_lem.append(lem.lemmatize(words))
    train_data_lda.append(new_lem)
    
data_lda = []
for i in tokens_all:
    new_lem = []
    for words in i:
        new_lem.append(lem.lemmatize(words))
    data_lda.append(new_lem)


id2word = corpora.Dictionary(train_data_lda)
id2word_all= corpora.Dictionary(data_lda)
id2word_all.filter_extremes(no_below=5)

corpus = [id2word_all.doc2bow(text) for text in train_data_lda]
corpus_all = [id2word_all.doc2bow(text) for text in data_lda]

random.seed(12345)
lda_train = gensim.models.LdaModel(
                            corpus=corpus,
                            num_topics=8,
                            id2word=id2word_all, eta=0.1, iterations=100)
lda_train.show_topics()

t = lda_train.update(corpus_all)

top_names = [0,1,2,3,4,5,6,7]
topics = pd.DataFrame(index=df.index,columns=top_names)
for i in range(len(corpus_all)):
    tops = lda_train.get_document_topics(corpus_all[i])
    for j in tops:
        topics.iloc[i,j[0]]=j[1]
        
topics = topics.fillna(0)
topics.columns = ['PoliticaEstera','Attualita','CampagnaTrump','Russia',
                  'Repubblicani','Arresti','InvestigazioneClinton','Societa']
topics.to_csv('topics.csv',index=False)
    
# tops_train = lda_train.get_document_topics(corpus)



##### CREAZIONI VARIABILI ESPLICATIVE
#1 lunghezza del testo
length = []
for i in df['text']:
    length.append(len(i))


#2,3 freq parole di lunghezza >11, <7
# testo processato ma senza stemming
freq11 = []
freq5 = []
# freq14 = []
# freq15 = []
for i in tokens:
    c11 = 0
    c7 = 0
    # c14 = 0
    # c15 = 0
    for t in i:
        if len(t) >= 11:
            c11 += 1
        elif len(t) <= 5:
            c7 += 1
        # elif len(t) == 14:
        #     c14 += 1
        # elif len(t) >= 15:
        #     c15 += 1
    freq11.append(c11/len(i))
    freq5.append(c7/len(i))
    # freq14.append(c14/len(i))
    # freq15.append(c15/len(i))
        
#4,5,6 numero medio di parole per frase, % punteggiatura, numero citazioni
# testo non processato
tokens_np = []
for i in df['text']:
    word_tokens = word_tokenize(i)
    tokens_np.append(word_tokens)
    

nwords_tot = []
npunct_tot = []
nquotes_tot = []
for i in tokens_np:
    count = 0
    npunct = 0
    nquotes = 0
    nwords = []
    for x in i:
        if x =='.' or x ==';' or x =='!' or x=='...' or x==':':
            nwords.append(count)
            count = 0
            npunct += 1
        elif x == '“':
            npunct += 1
            nquotes += 1
        elif x ==',':
            npunct += 1
        else: 
            count = count + 1
    if nwords == []:
        nwords.append(count)
    npunct_tot.append(npunct)
    nquotes_tot.append(nquotes)
    nwords_tot.append(nwords)
    

def Average(lst):
    return sum(lst)/len(lst)

av_nwords = []                
for i in nwords_tot:
    av_nwords.append(Average(i))     

# numero di lettere maiuscole
nupper = []
for i in tokens:
    s = 0
    for x in i:
        s += sum(1 for c in x if c.isupper())
    nupper.append(s)


#7 BAG OF WORDS, TFIDF
# calcolo top words nelle fake news
df_fake=df_train[df_train['label']==1]
fake_vect = TfidfVectorizer()
count = fake_vect.fit_transform(df_fake['processed_text'])
fake_words = list(fake_vect.get_feature_names())
#dtm_fake = pd.DataFrame(count.todense(), columns = fake_words)
trans_f = count.transpose()
sm_f=[]
sm_f = trans_f.sum(axis=1)
t = pd.DataFrame()
names = pd.DataFrame(fake_words)
sums = pd.DataFrame(sm_f)
t = pd.concat([names,sums], axis=1)
t.columns = ['names','sums']

# sm = t.getcol(0).todense()
# top = np.sort(sm,axis=0)[::-1]

top_fake_m = t.sort_values(by='sums', ascending=False).head(20)
top_fake = list(top_fake_m['names'])

# dtm corpus
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(df['processed_text'])
top_words = list(vectorizer.get_feature_names())
#dtm = pd.DataFrame(tfidf.todense(), columns=top_fake)
# tfidf.gelcol(top_fake[0])
# j = 0
ind=[]
for x in top_fake:
    for i in range(len(top_words)):
        if top_words[i] == x:
            ind.append(i)


d = pd.DataFrame()
for i in ind:
    c = tfidf[:,i]
    c = pd.DataFrame(c.todense())
    d = pd.concat([d,c], axis = 1)

d.columns = top_fake
d.to_csv('tfidf.csv',index=False)


# TOPIC MODELLING

# CREAZIONE NUOVO DATA FRAME CON VARIABILI ESPLICATIVE
df1 = pd.DataFrame(
    {'label' : list(df['label']),
     'length' : length,
     'perc5' : freq5,
     'perc11' : freq11,
     'av_words' : av_nwords,
     'punct': npunct_tot,
     'quotes':nquotes_tot
     })

d = pd.read_csv('tfidf.csv')
df2 = pd.concat([df1,d], axis=1)
df3 = pd.concat([df2,d_em], axis=1)
# df2 = pd.concat([df1a, new_list], axis = 1)
df3.to_csv('modelling_data_wel.csv', index = False)


###########
data = pd.read_csv('modelling_data_wel.csv')
data[]
emotions = pd.read_csv('emotions_data_wel.csv')
d_em = emotions.drop(columns='label')
topics = pd.read_csv('topics.csv')                                
data1 = pd.concat([data,d], axis=1)
data2 = pd.concat([data1, topics], axis = 1)
data2.to_csv('modelling_data1_wel.csv', index = False)
##
data = pd.read_csv('modelling_data1_wel.csv')
data1 = pd.concat([df['Unnamed: 0'], data], axis=1)
data1.columns = ['index', 'label', 'length', 'perc5', 'perc11', 'av_words', 'punct',
       'quotes', 'trump', 'us', 'people', 'said', 'one', 'would', 'like',
       'also', 'president', 'even', 'time', 'clinton', 'state', 'new', 'many',
       'could', 'donaldtrump', 'government', 'media', 'get', 'fear', 'anger',
       'trust', 'surprise', 'positive', 'negative', 'sadness', 'disgust',
       'joy', 'anticipation']

data1.to_csv('modelling_data1_wel.csv', index = False)
# train e test per modelling data

train = pd.DataFrame()
for i in df_train['Unnamed..0']:
    train = pd.concat([train,data.loc[data['index']==i]], axis=0)

test = pd.DataFrame()
for i in df_test['Unnamed..0']:
    test = pd.concat([test,data.loc[data['index']==i]], axis=0)

test.to_csv('test_modelling.csv', index=False)
train.to_csv('train_modelling.csv', index=False)


# data = pd.read_csv('modelling_data1_wel.csv')
# data.columns
# data1 = data.loc[:,'trump':'get']
# remove = data1.columns
# data2 = data.drop(columns=remove)
# data3 = pd.concat([data2, d], axis=1)
# data3.to_csv('modelling_data1_wel.csv', index = False)
# data3.columns
