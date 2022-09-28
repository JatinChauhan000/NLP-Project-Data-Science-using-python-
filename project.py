import string
import seaborn as sns
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import bs4 
import textblob
from textblob import TextBlob
from bs4 import BeautifulSoup
import warnings
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay 
warnings.filterwarnings('ignore')
%matplotlib inline


os.chdir('C:\\Users\\91964\\Downloads')

df = pd.read_excel('Tweet_NFT.xlsx')

df.head()

df.columns

text_df = df.drop(['tweet_created_at', 'tweet_intent'],axis = 1)

text_df.head()

print(text_df['tweet_text'].iloc[0],"\n")
print(text_df['tweet_text'].iloc[1],"\n")
print(text_df['tweet_text'].iloc[2],"\n")
print(text_df['tweet_text'].iloc[3],"\n")
print(text_df['tweet_text'].iloc[4],"\n")


text_df.info()


from nltk.corpus import stopwords


stop_words = stopwords.words('english')


def data_processing(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+https|S+", '',text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','',text)
    text = re.sub(r'[^\w\s]','',text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)


text_df.tweet_text = text_df['tweet_text'].apply(data_processing)


text_df = text_df.drop_duplicates('tweet_text')


text_df['tweet_text'] = text_df['tweet_text'].str.replace(r"[\"\0-9\'\_\|\?\=\.\<\>\@\#\*\,]", ' ')


stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data


text_df['tweet_text'] = text_df['tweet_text'].apply(lambda x: stemming(x))
                                                    # to define a function in one line
    
    
text_df.head()
    
    
print(text_df['tweet_text'].iloc[0],"\n")
print(text_df['tweet_text'].iloc[1],"\n")
print(text_df['tweet_text'].iloc[2],"\n")
print(text_df['tweet_text'].iloc[3],"\n")
print(text_df['tweet_text'].iloc[4],"\n")


text_df.info()


def polarity(text):
    return TextBlob(text).sentiment.polarity


text_df['polarity'] = text_df['tweet_text'].apply(polarity)


text_df.head()


def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"
    
    
text_df['sentiment'] = text_df['polarity'].apply(sentiment)


text_df.head(15)


fig = plt.figure(figsize=(5,5))
sns.countplot(x='sentiment', data = text_df)


pos_tweets = text_df[text_df.sentiment == 'Positive']
pos_tweets = pos_tweets.sort_values(['polarity'], ascending = False)
pos_tweets.head(15)


tweet1_text = ' '.join([word for word in pos_tweets['tweet_text']])
tweet1_text


neg_tweets = text_df[text_df.sentiment == 'Negative']
neg_tweets = neg_tweets.sort_values(['polarity'], ascending = False)
neg_tweets.head(15)


tweet2_text = ' '.join([word for word in neg_tweets['tweet_text']])
tweet2_text


neu_tweets = text_df[text_df.sentiment == 'Neutral']
neu_tweets = neu_tweets.sort_values(['polarity'], ascending = False)
neu_tweets.head(15)


tweet3_text = ' '.join([word for word in neu_tweets['tweet_text']])
tweet3_text


text_df.head(15)


fig = plt.figure(figsize=(7,7))
colors = ("yellowgreen","gold","red")
wp = {'linewidth':4, 'edgecolor':"white"}
tags = text_df['sentiment'].value_counts()
explode = (0.1,0.1,0.1)
tags.plot(kind='pie', autopct='%1.1f%%', shadow=True, colors = colors, 
         startangle = 90, wedgeprops = wp, explode = explode, label='')
plt.title('Distribution of Sentiments')


vect = CountVectorizer(ngram_range=(1,2)).fit(text_df['tweet_text'])
       # it will give some features words to train our model
    
    
feature_names = vect.get_feature_names()
print("features names:\n {}".format(feature_names[100:150]))


# Train and Test your data
# divide the data in X and Y

X = text_df['tweet_text']
Y = text_df['sentiment']
X = vect.transform(X)


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
                                                                    # 20%
    
    
print("size of x_train:", (x_train.shape))
print("size of y_train:", (y_train.shape))
print("size of x_test:", (x_test.shape))
print("size of y_test:", (y_test.shape))


# Train your data using LogisticRegression
# LogisticRegression = predict the probability in binary form

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
logreg_pred = logreg.predict(x_test)
logreg_acc = accuracy_score(logreg_pred, y_test)
print("test accuracy: {:.2f}%".format(logreg_acc*100))
# 96.91% accuracy


# another way to check accuracy
# classification_report to check accuracy

print(confusion_matrix(y_test, logreg_pred))
print('\n')
print(classification_report(y_test, logreg_pred))

# 97% accuracy


