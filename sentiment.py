import pandas as pd
import numpy as np
import tweepy 
from textblob import TextBlob
consumerKey =""
consumerSecret = ""
accessToken =""
accessTokenSecret =""
   
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)
authenticate.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(authenticate, wait_on_rate_limit = True) # api object
post = api.user_timeline(screen_name="elonmusk", count = 100, lang ="en", tweet_mode="extended")
i=1
for tweet in post[:5]:
    print(str(i) +') '+ tweet.full_text + '\n')
    i= i+1

twitter = pd.DataFrame([tweet.full_text for tweet in post], columns=['Tweets'])
def cleanTxt(text):
    text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
    text = re.sub('#', '', text) # Removing '#' hash tag
    text = re.sub('RT[\s]+', '', text) # Removing RT
    text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
    return text
twitter['Tweets'] = twitter['Tweets'].apply(cleanTxt)
twitter.head()
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity
twitter['Subjectivity'] = twitter['Tweets'].apply(getSubjectivity)
twitter['Polarity'] = twitter['Tweets'].apply(getPolarity)
twitter
​def getAnalysis(score):
    if score < 0:
        return 'Negative'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Positive'
twitter['Analysis'] = twitter['Polarity'].apply(getAnalysis)
positive = twitter.loc[twitter['Analysis'].str.contains('Positive')]
positive.drop(['Subjectivity','Polarity'], axis=1, inplace=True)
positive.head()
negative = twitter.loc[twitter['Analysis'].str.contains('Negative')]
negative.drop(['Subjectivity','Polarity'], axis=1, inplace=True)
negative.head()
neutral = twitter.loc[twitter['Analysis'].str.contains('Neutral')]
neutral.drop(['Subjectivity','Polarity'], axis=1, inplace=True)
neutral.head()
print(str(round((positive.shape[0]/twitter.shape[0])*100, 1))+' %')
print(str(round((negative.shape[0]/twitter.shape[0])*100, 1))+' %')
twitter['Analysis'].value_counts()
