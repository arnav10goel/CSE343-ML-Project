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

print(stop_words)

stri = "hiffhji jfrkf rkjforfj orfkrof"

