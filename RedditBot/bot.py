import praw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import time
import random
import pickle
import re

from collections import Counter
from nltk.corpus import stopwords
import ssl
import string
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', download_dir='/Users/arnav/Desktop/MachineLearning/ML_CSE343 Project/redditbots')
stop_words = set(stopwords.words('english'))

# Create the Reddit instance by authenticating and calling the API
reddit = praw.Reddit(
    client_id="bjQT7ACBPqV257sJfNJwdQ",
    client_secret="5bApHvGRe5DDbRUNbYBQCFaR6M_NAQ",
    user_agent="<console:BotBhai:1.0>",
    username="MLproject_test",
    password="deadline"
)

# Function to get a random quote
def get_suicide_risk(current_post):
    
    # Load the vectorizer and the model
    vectorizer = pickle.load(open("/Users/arnav/Desktop/MachineLearning/ML_CSE343 Project/redditbots/vectorizer.sav", "rb"))
    model = pickle.load(open("/Users/arnav/Desktop/MachineLearning/ML_CSE343 Project/redditbots/finalized_model.sav", "rb"))

    # convert to dataframe
    df = pd.DataFrame([current_post], columns=['text'])

    print(1)
    print(df)
    # Remove URLs
    df['text'] = df['text'].apply(lambda x: re.sub(r'http\S+', '', x))

    # Remove HTML tags
    df['text'] = df['text'].apply(lambda x: re.sub(r'<.*?>', '', x))

    # Remove special characters
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    # Remove Emojis
    df['text'] = df['text'].apply(lambda x: x.encode('ascii', 'ignore').decode('ascii'))

    # Remove non-ASCII characters
    df['text'] = df['text'].apply(lambda x: x.encode("ascii", "ignore").decode())

    # Remove leading and trailing whitespaces
    df['text'] = df['text'].apply(lambda x: x.strip())

    # Remove numbers
    df['text'] = df['text'].apply(lambda x: re.sub(r'\d+', '', x))

    # Remove multiple spaces
    df['text'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', x))

    # Remove all items starting with @
    df['text'] = df['text'].apply(lambda x: re.sub(r'@\w+', '', x))

    # Remove all items starting with #
    df['text'] = df['text'].apply(lambda x: re.sub(r'#\w+', '', x))

    # Remove all items starting with &
    df['text'] = df['text'].apply(lambda x: re.sub(r'&\w+', '', x))

    # Remove all non-alphanumeric characters
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^A-Za-z0-9 ]+', '', x))

    # Remove all booleans
    df['text'] = df['text'].apply(lambda x: re.sub(r'true|false', '', x))

    # Remove all punctuations
    df['text'] = df['text'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

    # Remove all stopwords
    df['text'] = df['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    
    # Convert to lowercase
    df['text'] = df['text'].apply(lambda x: x.lower())

    # # Ensure only strings are present
    # df['text'] = df['text'].apply(lambda x: isinstance(x, str))

    print(2)
    # Preprocessing the data
    # Lemmatization

    lemmatizer = WordNetLemmatizer()
    df['text'] = df['text'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))

    print(3)
    # Stemming
    stemmer = PorterStemmer()
    df['text'] = df['text'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

    # Tokenization
    tokenizer = TweetTokenizer()
    df['text'] = df['text'].apply(lambda x: tokenizer.tokenize(x))

    # Convert the list of tokens into a string
    df['text'] = df['text'].apply(lambda x: ' '.join(x))

    # Get the current post
    current_post = df['text']
    print("\n\n\nInside the Model:\n\n\n")
    print(current_post)

    # Vectorize the post content
    current_post = vectorizer.transform(current_post)

    # Predict the risk level
    risk_level = model.predict(current_post)

    print(f"\n\nRisk Level: {risk_level}")

    return risk_level[0]

# Monitor mentions and reply to comments
def check_mentions():
    for mention in reddit.inbox.mentions(limit=None):
        try:
            if mention.new:
                print(f"New mention from {mention.author.name}: {mention.body}")

                submission = mention.submission
                post_content = submission.selftext
                #print(f"Post Content:\n{post_content}")
                suicide_risk=get_suicide_risk(post_content)
                if(suicide_risk == 'suicide'):
                    reply_message = f"Thank you for tagging me! I've noted some concerning content in the post. If you or someone you know needs support, consider reaching out to a mental health professional.\nSpeak with someone today: 9152987821"
                else:
                    reply_message = f"Thank you for tagging me! This text looks safe from my point of view."

                # Reply to the mention
                mention.reply(reply_message)
                print(f"Replying to {mention.author.name}'s comment with: {reply_message}")

                # Mark the mention as read to avoid processing it again
                mention.mark_read()
        except Exception as e:
            print(f"An unexpected exception occurred: {e}")

def main():
    while True:
        check_mentions()
        print("Sleeping for 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    main()
