import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV # preprocessing libraries
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression # Model
from sklearn.metrics import confusion_matrix, plot_confusion_matrix # for plotting
from sklearn.naive_bayes import MultinomialNB # Model
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # transforming
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier # Classification Problem
from sklearn.ensemble import BaggingClassifier


from nltk.corpus import stopwords, words
def get_posts(subreddit, number=100):
    '''
    This function is purposed to grab a number of posts based 
    on how much you want. It also can do it based on the 
    subreddit you want to. 
    
    Note: the number of posts received may vary from the
    number requested.
    --------------------------------------------------------
    subreddit: string; name of subreddut
    number: int
    '''
    url = 'https://api.pushshift.io/reddit/comment/search'
    
    url = 'https://api.pushshift.io/reddit/search/submission'
    params = {
            'subreddit': subreddit,
            'size': 100
        } # 'before' is added later in the while loop
    df = pd.DataFrame() # empty dataframe for concatenating
    returned = pd.DataFrame()
    while True: # Need a break out of this
        time.sleep(3)
        
        res = requests.get(url, params)
        data = res.json()
        posts = data['data']
        df = pd.DataFrame(posts)
        
        params['before'] = df.iloc[-1:,:]['created_utc']
        
        returned = pd.concat([returned, df[['subreddit','selftext','title']]], axis=0)
        returned.drop_duplicates(inplace=True)
        
        if len(returned) > number:
            break
      
    returned.reset_index(inplace=True,drop=True)
    return returned[:number]

def get_comments(subreddit, number=100):
    '''
    This function is purposed to grab a number of comments based 
    on how much you want. It also can do it based on the 
    subreddit you want to. 
    
    Note: the number of posts received may vary from the
    number requested.
    --------------------------------------------------------
    subreddit: string; name of subreddut
    number: int
    '''
    url = 'https://api.pushshift.io/reddit/comment/search'
    params = {
    'subreddit': subreddit, 
    'size': 100 
    }
    df = pd.DataFrame() # empty dataframe for concatenating
    returned = pd.DataFrame()
    while True: # Need a break out of this
        time.sleep(3)
        
        res = requests.get(url, params)
        data = res.json()
        posts = data['data']
        df = pd.DataFrame(posts)
        
        params['before'] = df.iloc[-1:,:]['created_utc']
        
        returned = pd.concat([returned, df[['subreddit','body']]], axis=0)
        returned.drop_duplicates(inplace=True)
        
        returned = returned[returned['body'] != '[deleted]']
        
        if len(returned) > number:
            break
            
    returned.reset_index(inplace=True,drop=True)
    
    return returned[:number]


def concat_df(dataframes):
    '''
    dataframes: list of dataframes
    '''
    df = pd.concat(dataframes, axis=0)
    df.reset_index(inplace=True,drop=True)

    return df


def check_words(sentence):
    '''
    Takes in a sentence, then will separate the words.
    Once that is complete, it will check to see if the word
    is in the big list of words found in nltk.corpus words.words()
    ---------------------------------------------------------------
    Returns the word back or None/''
    '''
    tokenizer = RegexpTokenizer(r'\w+')
    splits = tokenizer.tokenize(sentence) # Sentence >> list of words
    results = []
    for s in splits:
        if s.lower() in words.words():
            results.append(s)

    return results




'''
Code further down is a failed attempt at trying to make
my life easier. Please do not grade me for including it. I just
don't want to lost the data. 

It is to improve later on.
'''



def print_menu(number):
    if number == 1:
        print('Transformer Menu:')
        print('1. CountVectorizer()')
        print('2. TfidVectorizer()')
        print('Type the number')
    if number == 2:
        print('Model Menu:')
        print('1. Logistic Regression()')
        print('2. MultinomialNB()')
    if number == 3:
        print('We have made it to parameters.\nFrom here on out make sure that you input everything in a list')
        print('Give me a list of the amount of max features to include per model.')
    if number == 4:
        print('Give a list for max_df and min_df.')
        print('min_df is a list of integers.')
        print('max_df is a list of floats between 0 and 1')
    if number == 5:
        print('Give a list of tuples for n-grams.')
    if number == 6:
        print('Give a list of stopwords')
        
transformer = {1:CountVectorizer(), 2:TfidfVectorizer()}
models = {1:LogisticRegression(), 2:MultinomialNB()}

def modeling(X_train, y_train):
    # Select a Transformer
    print_menu(1)
    print('Type in the number.')
    input1 = int(input('Which transformer do you want to use: '))

    # Select a model
    print_menu(2)
    print('Type in the number.')
    input2 = int(input('Which model do you want to use: '))

    # Max Features Parameters
    print_menu(3)
    print('For defaults, enter []')
    input3 = input('List of max features goes here: ')

    # Min_df and Max_df
    print_menu(4)
    print('For defaults, enter []')
    input4 = input('List of min_df goes here: ')
    input5 = input('List of max_df goes here: ')

    # ngram range
    print_menu(5)
    print('For defaults, enter []')
    input6 = input('List of n=grams goes here: ')

    # stop words
    print_menu(6)
    print('For defaults, enter []')
    input7 = input('List of stopwords goes here: ')

    pipe = Pipeline({
        ('trans', transformer[input1]),
        ('model', models[input2])
    })

    params = {'trans__max_features': input3,
             'trans__min_df': input4,
             'trans__max_df': input5,
             'trans__ngram_range': input6,
             'trans__stop_words': input7}

    input7 = int(input('How many cross val folds? '))
    gs = GridSearchCV(pipe, params, cv = input7, verbose = 1)

    return gs




