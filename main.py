'''

Sentiment analysis on IMDB movie reviews using NLTK library's VADER lexicon.
Data source: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

'''

#Library Import#
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#File opener
df = pd.read_csv('IMDB Dataset.csv')

#Sentiment analysis function
def sentiment_analyze(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    compound_score = score['compound']
    return compound_score

#Analysis of the dataframe
df['post-sentiment'] = df['sentiment'].apply(sentiment_analyze)

#Test
#print(df.head(50))


import matplotlib.pyplot as plt

#Counter of each instance of positive and negative sentiment

sentiment_counts = pd.cut(df['post-sentiment'],
                          bins=3,
                          labels = ['Negative','Neutral','Positive']).value_counts()

#Sentiment Plot
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.title('Sentiment Analysis')
plt.show()

#Print the number of instances of each sentiment
for sentiment, count in sentiment_counts.items():
    print(f"{sentiment}:{count}")

#CSV Extraction
#df.to_csv('IMDB Dataset_Sentiment Analyzed.csv')



