import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re, collections
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import cohen_kappa_score
import enchant 
import pandas as pd
import spacy
import re
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
import urllib.request

dataframe = pd.read_csv('training_set_rel3.tsv', encoding = 'latin-1',sep='\t')
dataframe = dataframe[['essay_id','essay_set','essay','domain1_score']]

dataframe = dataframe[(dataframe['essay_set'] == 1)]
dataframe.dropna(axis=1, how='all', inplace=True)

dataframe.set_index('essay_id',inplace=True, drop=True)

def get_number_of_characters(df):
    chars = []
    for essay in df['essay']:
        chars.append(len(essay))
        
    return df.assign(num_of_chars = chars) 

dataframe = get_number_of_characters(dataframe)

print("noc done")

tokenizer = RegexpTokenizer(r'\w+')
def get_number_of_words(regExTokenizer, df):
    length = []
    for essay in df['essay']:
        length.append(len(regExTokenizer.tokenize(essay)))
        
    return df.assign(num_of_words = length) 

dataframe = get_number_of_words(tokenizer, dataframe)

print("now done")

def get_number_of_long_words(regExTokenizer,df):
    length = []
    for essay in df['essay']:
        long_words = 0
        for word in regExTokenizer.tokenize(essay):
            if len(word)>8:
                long_words+=1
        length.append(long_words)
    return df.assign(num_of_long_words = length)
dataframe = get_number_of_long_words(tokenizer, dataframe)

print("nolw done")

def get_number_of_short_words(regExTokenizer,df):
    length = []
    for essay in df['essay']:
        short_words = 0
        for word in regExTokenizer.tokenize(essay):
            if len(word)<=3:
                short_words+=1
        length.append(short_words)
    return df.assign(num_of_short_words = length)
dataframe = get_number_of_short_words(tokenizer, dataframe)

print("nosw done")

def get_most_frequent_word_length(regExTokenizer,df):
    length = []
    for essay in df['essay']:
        len_dict = {}
        for word in regExTokenizer.tokenize(essay):
            if len(word) in len_dict:
                len_dict[len(word)] = len_dict[len(word)]+1
            else:
                len_dict[len(word)] = 1
        maxwords = max(len_dict,key=len_dict.get)
        length.append(maxwords)
    return df.assign(most_frequent_word_length = length)
dataframe = get_most_frequent_word_length(tokenizer, dataframe)

print("mfwl done")

def get_average_word_length(df):
    length = []
    for i in range(0,df.shape[0]):
        row = df.iloc[i]
        average_word_length = (row['num_of_chars']-row['num_of_words'])/row['num_of_words']
        length.append(average_word_length)
    return df.assign(average_word_length = length)
dataframe = get_average_word_length(dataframe)

print("awl done")

nlp = spacy.load('en_core_web_md')
def get_number_of_sentences(df):
    _byRow_sents = []
    for essay in df['essay']:
        sents = []
        parsed_essay = nlp(essay)
        for num, sentence in enumerate(parsed_essay.sents):
            sents.append(sentence)
        _byRow_sents.append(len(sents))   
    return df.assign(num_of_sentences = _byRow_sents)

dataframe = get_number_of_sentences(dataframe)

print("nos done")

def get_number_of_sentence_attributes(regExTokenizer,df):
    #number of long sentences
    #number of short sentences
    #average sentence length
    #most frequent sentence length
    num_of_long_sents = []
    num_of_short_sents = []
    average_sents_length = []
    most_frequent_sents_length = []
    for essay in df['essay']:
        long_sents = 0
        short_sents = 0
        sent_dict = {}
        num_of_sents = 0
        num_of_words = 0
        parsed_essay = nlp(essay)
        for num, sentence in enumerate(parsed_essay.sents):
            tokenized = len(regExTokenizer.tokenize(sentence.text))
            if tokenized > 25:
                long_sents+=1
            if tokenized < 10:
                short_sents+=1
            num_of_words+=tokenized
            num_of_sents+=1
            if tokenized in sent_dict:
                sent_dict[tokenized] = sent_dict[tokenized]+1
            else:
                sent_dict[tokenized] = 1
        num_of_long_sents.append(long_sents)
        num_of_short_sents.append(short_sents)
        average_sents_length.append(num_of_words/num_of_sents)
        maxwords = max(sent_dict,key=sent_dict.get)
        most_frequent_sents_length.append(maxwords)
    df = df.assign(num_of_long_sents = num_of_long_sents)
    df = df.assign(num_of_short_sents = num_of_short_sents)
    df = df.assign(average_sents_length = average_sents_length)
    df = df.assign(most_frequent_sents_length = most_frequent_sents_length)
    return df

dataframe = get_number_of_sentence_attributes(tokenizer,dataframe)

print("nosa done")

def get_number_of_different_words(regExTokenizer,df):
    length = []
    for essay in df['essay']:
        word_dict = {}
        for word in regExTokenizer.tokenize(essay):
            if word in word_dict:
                word_dict[word] = word_dict[word]+1
            else:
                word_dict[word] = 1
        length.append(len(word_dict))
    return df.assign(num_of_different_words = length)
dataframe = get_number_of_different_words(tokenizer, dataframe)

print("nodw done")

from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize

def get_number_of_stopwords(df):
    stop_words = set(stopwords.words('english'))
    length = []
    for essay in df['essay']:
        word_tokens = word_tokenize(essay)
        filtered_sentence = [w for w in word_tokens if w in stop_words]
        length.append(len(filtered_sentence))
    return df.assign(num_of_stopwords = length)
dataframe = get_number_of_stopwords(dataframe)

print("nosw done")

from readcalc import readcalc
import textstat
def get_readability_measures(df):
    gunning_fox_index = []
    flesch_reading_ease = []
    flesch_kincaid_grade_level = []
    dale_schall_index = []
    automated_readability_index = []
    SMOG = []
    LIX = []
    for essay in df['essay']:
        flesch_kincaid_grade_level.append(textstat.flesch_kincaid_grade(essay))
        gunning_fox_index.append(textstat.gunning_fog(essay))
        flesch_reading_ease.append(textstat.flesch_reading_ease(essay))
        dale_schall_index.append(textstat.dale_chall_readability_score(essay))
        automated_readability_index.append(textstat.automated_readability_index(essay))
        SMOG.append(textstat.smog_index(essay))
        calc = readcalc.ReadCalc(essay)
        LIX.append(calc.get_lix_index())
    df = df.assign(gunning_fox_index = gunning_fox_index)
    df = df.assign(flesch_reading_ease = flesch_reading_ease)
    df = df.assign(flesch_kincaid_grade_level = flesch_kincaid_grade_level)
    df = df.assign(dale_schall_index = dale_schall_index)
    df = df.assign(automated_readability_index = automated_readability_index)
    df = df.assign(SMOG = SMOG)
    df = df.assign(LIX = LIX)
    return df
dataframe = get_readability_measures(dataframe)

print("rm done")

dataframe.to_csv("allFeaturesSet1.csv",index=False)