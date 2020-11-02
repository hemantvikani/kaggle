# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 10:40:15 2020

@author: heman
"""

import numpy as np
import spacy
import pandas as pd

df = pd.read_csv('Musical_instruments_reviews.csv')

df.head(10)

print ('Total ratings per rating:','\n',df.overall.value_counts())

#Number of unique instrument ids

print('Number of unique instrument ids',len(df.asin.unique()))
print('total number of rows',df.shape[0])

df['reviews'] = df['reviewText'] + ' ' + df['summary']
del df['reviewText']
del df['summary']

#rename overall to rating

df.rename(columns = {'overall' : 'rating'}, inplace = True)

df['reviews'].isnull().sum()

#drop rows with missing reviews

df.dropna(axis=0,inplace =True)

df['reviews'].isnull().sum()

import seaborn as sns 
import matplotlib.pyplot as plt

#look at top 20 instrument reviews
#look at bottom 20 instrument reviews

top_20 = df.asin.value_counts().head(20)
btm_20 = df.asin.value_counts().tail(20)

#create pivot table to plot
top_20_df = pd.DataFrame()
top_20_ids=list(top_20.index)

for i in top_20_ids:
    top_20_df = top_20_df.append(df[df['asin']==i],ignore_index = True)
    
    
table = pd.pivot_table(top_20_df,values = 'rating',index = top_20_df['asin'],aggfunc=np.mean)

#create figure
plt.figure(figsize=(10,6))
sns.barplot(x=table.index, y='rating', data=table)
plt.xticks(rotation=90)
plt.xlabel('Instrument ID')
plt.ylabel('Average Rating')
plt.title('Instruments with the Highest Number of Ratings (Top 20)')
plt.tight_layout()
plt.show()


#Plot ave rating for 20 Instruments with lower number of ratings
btm_20_df=pd.DataFrame()
btm_20_ids=list(btm_20.index)
for i in btm_20_ids:
    btm_20_df=btm_20_df.append(df[df['asin']==i],ignore_index=True)
table_btm = pd.pivot_table(btm_20_df, values='rating',index=btm_20_df['asin'],aggfunc=np.mean)

plt.figure(figsize=(10,6))
sns.barplot(x=table_btm.index, y='rating', data=table_btm)
plt.xticks(rotation=90)
plt.xlabel('Instrument ID')
plt.ylabel('Average Rating for Instrument')
plt.title('Instruments with Fewest Number of Ratings (Bottom 20)')
plt.tight_layout()
plt.show()


#Plot ratings percentages

t = pd.DataFrame(data = df['rating'].value_counts(normalize=True)*100)
plt.figure(figsize=(10,6))
sns.barplot(x=t.index, y=t.rating,palette="Blues_d")
plt.xlabel('Rating',fontsize=20)
plt.ylabel('Percent of Total Ratings',fontsize=20)
plt.show()


#Drop columns not using for analysis

col_to_drop = ['reviewerID','asin','reviewerName','helpful','unixReviewTime','reviewTime']
instrument_reviews = df.drop(columns = col_to_drop,axis =1)
instrument_reviews.head()


#Create sentiment column

instrument_reviews['sentiment'] = instrument_reviews['rating'].map({5:2,4:2,3:1,2:0,1:0})

instrument_reviews.head() 

instrument_reviews.sentiment.value_counts(normalize=True)*100


#Data to plot

labels = ['Positive', 'Neutral','Negative']
sizes = [instrument_reviews['sentiment'].value_counts(normalize=True)]

labels_rating = ['5','4','3','2','1']
sizes_rating = [instrument_reviews['rating'].value_counts(normalize=True)]



plt.figure(figsize=(10,6))
colors=['olive','yellow','lightcoral']
 
plt.pie(instrument_reviews['sentiment'].value_counts(normalize=True),colors=colors,labels=['Positive','Neutral','Negative'],autopct='%1.2f%%',shadow=True)
plt.title('Sentiment',fontsize=20)

plt.show()



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.base import TransformerMixin

from sklearn.pipeline import Pipeline

#import models to test
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
#import metrics

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import string
import re
import spacy
spacy.load('en')
from spacy.lang.en import English
parser = English()

#splitting into train and valid
from sklearn.model_selection import train_test_split
X_train,X_valid,y_train,y_valid = train_test_split(instrument_reviews['reviews'],instrument_reviews['sentiment'],random_state=42,test_size=0.3)
print('Training data shape:',X_train.shape)
print('Testing data shape:',X_valid.shape)

#define function to clean and tokenize data
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”",".",""]


class CleanTextTransformer(TransformerMixin):
   def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]
   def fit(self, X, y=None, **fit_params):
        return self
def get_params(self, deep=True):
        return {}
    
def cleanText(text):    
    text = text.strip().replace("\n", " ").replace("\r", " ")
    text = text.lower()
    return text

def tokenizeText(sample):
    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    tokens = [tok for tok in tokens if tok not in STOPLIST]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    return tokens


#Create string of Positive, Neutral, Negative words for Wordcloud
pos = X_train[y_train[y_train == 2].index]
neut = X_train[y_train[y_train == 1].index]
neg = X_train[y_train[y_train == 0].index]

X_train.shape,pos.shape,neut.shape,neg.shape


#Create text for wordcloud for each sentiment
pos_words=''
for w in pos.apply(cleanText).apply(tokenizeText):
     pos_words+=" ".join(w)
print('There are {} positive words'.format(len(pos_words)))


neut_words=''
for w in neut.apply(cleanText).apply(tokenizeText):
     neut_words+=" ".join(w)
print('There are {} neutral words'.format(len(neut_words)))        

neg_words=''
for w in neg.apply(cleanText).apply(tokenizeText):
     neg_words+=" ".join(w)
print('There are {} negative words'.format(len(neg_words)))   


#Models to test
clf = LinearSVC(max_iter=10000)
xgb = XGBClassifier(n_estimators = 100, learning_rate=0.1)
rfc = RandomForestClassifier(n_estimators=100)
lr = LogisticRegression(max_iter=500)
mnb = MultinomialNB()

models = [clf, xgb, rfc, lr, mnb]

# def printNMostInformative(vectorizer, clf, N):
#     feature_names = vectorizer.get_feature_names()
#     coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
#     topNeg = coefs_with_fns[:N]
#     topPos = coefs_with_fns[:-(N + 1):-1]
#     print("Negative best: ")
#     for feat in topNeg:
#         print(feat)
#     print("Positive best: ")
#     for feat in topPos:
#         print(feat)
# print("Top 10 features used to predict: ")        
# printNMostInformative(vectorizer, clf, 10)
# pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer)])
# transform = pipe.fit_transform(X_train, y_train)
# vocab = vectorizer.get_feature_names()
# for i in range(len(X_train)):
#     s = ""
#     indexIntoVocab = transform.indices[transform.indptr[i]:transform.indptr[i+1]]
#     numOccurences = transform.data[transform.indptr[i]:transform.indptr[i+1]]
#     for idx, num in zip(indexIntoVocab, numOccurences):
#         s += str((vocab[idx], num))



#Create loop to get accuracy and classification report for models
for model in models:
    
    vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
    pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_valid)
    print('model:',model,'\t',"accuracy:", accuracy_score(y_valid, preds))
    print('Classification Report','\n',50*'-','\n',classification_report(y_valid, preds),'\n',50*'-')
    
    
    
model = XGBClassifier(subsample=0.8)

vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])

param_grid = { 
    'model__n_estimators': range(100,1001,300),
    'model__max_depth': [8],
    'model__learning_rate':[0.1]}
from sklearn.model_selection import GridSearchCV
CV = GridSearchCV(pipe, param_grid, n_jobs= 1)
                  
CV.fit(X_train, y_train)  
print(CV.best_params_)    
print(CV.best_score_)




model = XGBClassifier(subsample=0.8)

vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])

param_grid = { 
    'model__n_estimators': range(100,1001,300),
    'model__max_depth': [5],
    'model__learning_rate':[0.1]}
from sklearn.model_selection import GridSearchCV
CV = GridSearchCV(pipe, param_grid, n_jobs= 1)
                  
CV.fit(X_train, y_train)  
print(CV.best_params_)    
print(CV.best_score_)



model = XGBClassifier(subsample=0.8)

vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])

param_grid = { 
    'model__n_estimators': range(100,1001,300),
    'model__max_depth': [5],
    'model__learning_rate':[0.1]}
from sklearn.model_selection import GridSearchCV
CV = GridSearchCV(pipe, param_grid, n_jobs= 1)
                  
CV.fit(X_train, y_train)  
print(CV.best_params_)    
print(CV.best_score_)



model = XGBClassifier(subsample=0.8)

vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])

param_grid = { 
    'model__n_estimators': [700],
    'model__max_depth': [5,8],
    'model__learning_rate':[0.1,0.2]}
from sklearn.model_selection import GridSearchCV
CV = GridSearchCV(pipe, param_grid, n_jobs= 1)
                  
CV.fit(X_train, y_train)  
print(CV.best_params_)    
print(CV.best_score_)



model = MultinomialNB()

vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])

param_grid = { 
    'model__alpha': np.linspace(0.5,1.6,6),
    'model__fit_prior': [True, False]}
from sklearn.model_selection import GridSearchCV
CV = GridSearchCV(pipe, param_grid, n_jobs= 1)
                  
CV.fit(X_train, y_train)  
print(CV.best_params_)    
print(CV.best_score_)
      
      
model = RandomForestClassifier()

vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])

param_grid = { 
    'model__n_estimators': [100],
    'model__max_features': ['auto'],
    'model__max_depth' : [2,3,4],
    'model__criterion' :['gini'],
    'model__class_weight': [None]}
from sklearn.model_selection import GridSearchCV
CV = GridSearchCV(pipe, param_grid, n_jobs= 1)
                  
CV.fit(X_train, y_train)  
print(CV.best_params_)    
print(CV.best_score_)



model = LogisticRegression(max_iter=500)
vectorizer = CountVectorizer(tokenizer=tokenizeText, ngram_range=(1,1))
pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer), ('model', model)])
# pipe.fit(X_train, y_train)
# preds = pipe.predict(X_valid)
# print("accuracy:", accuracy_score(y_valid, preds))
# print('Classification Report','\n',50*'-','\n',classification_report(y_valid, preds),'\n',50*'-')

#
param_grid = {'model__C': (0.01,0.1,1),'model__class_weight': [None,'balanced']}
from sklearn.model_selection import GridSearchCV
CV = GridSearchCV(pipe, param_grid, n_jobs= 1)
                  
CV.fit(X_train, y_train)  
print(CV.best_params_)    
print(CV.best_score_)