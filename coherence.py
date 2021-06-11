import math
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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
tokenizer = RegexpTokenizer(r"[\w']+")
from sklearn.metrics.pairwise import euclidean_distances


dataframe = pd.read_csv('training_set_rel3.tsv', encoding = 'latin-1',sep='\t')
dataframe = dataframe[['essay_id','essay_set','essay','domain1_score']]

dataframe = dataframe[(dataframe['essay_set'] == 8)]
dataframe.dropna(axis=1, how='all', inplace=True)
dataframe.set_index('essay_id',inplace=True, drop=True)

countdown = 0
def get_string_arrays(regExTokenizer, df):
    array_of_strings = []
    tempnum = 0
    for essay in df['essay']:
        # print(tempnum)
        tempnum+=1

        words = regExTokenizer.tokenize(essay)
        # sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        # words = sent_detector.tokenize(essay.strip())
        if(tempnum==1532):
            print(essay,words)
        if len(words)<=3:
            array_of_strings.append([])
            continue;
        array = []
        size = len(words)//4
        curr = 0
        flag = 0
        while flag == 0:
            new = ' '
            for i in range(curr,curr+size):
                
                    # print(words[i])
                if i>=len(words):
                    flag = 1
                    break
                
                new = new+' '+words[i]
            
            curr+=10
            array.append(new)
        
        array_of_strings.append(array)
        # array_of_strings+=array
        
    return array_of_strings
array_of_strings = get_string_arrays(tokenizer,dataframe)

print(countdown)
countdown+=1

tfidfvectorizer = TfidfVectorizer(analyzer='word')
cosine_array = []
for i in range(0,len(array_of_strings)):

    
    # print(i)
    if(len(array_of_strings[i])<=2) :
        
        cosine_array.append(np.empty([1,1],dtype=int))
        continue
    if(i==1115):
        print(len(array_of_strings[i]))
    tfidf_wm = tfidfvectorizer.fit_transform(array_of_strings[i])
    similarity_matrix = cosine_similarity(tfidf_wm)

    cosine_array.append(similarity_matrix)

print(countdown)
countdown+=1

def get_avg_distance_bw_neighbours(cosine_array):
    array = []
    for i in range(0,len(cosine_array)):
        j = 0
        dist = 0
        while j < cosine_array[i].shape[0]-1:
            dist+=cosine_array[i][j][j+1]
            j+=1
        if (cosine_array[i].shape[0]-1)==0:
            array.append(0)
        else :
            array.append(dist/(cosine_array[i].shape[0]-1))
    return array
avg_distance_bw_neighbours = get_avg_distance_bw_neighbours(cosine_array)
dataframe=dataframe.assign(avg_distance_bw_neighbours=avg_distance_bw_neighbours)

print(countdown)
countdown+=1

def get_maxmin_distance_bw_neighbours(cosine_array):
    maxarray = []
    minarray = []
    for i in range(0,len(cosine_array)):
        j = 0
        maxn = 0
        minn = 1
        while j < cosine_array[i].shape[0]-1:
            maxn = maxn if maxn > cosine_array[i][j][j+1] else cosine_array[i][j][j+1]
            minn = minn if minn < cosine_array[i][j][j+1] else cosine_array[i][j][j+1]
            j+=1
        maxarray.append(maxn)
        minarray.append(minn)
    return maxarray,minarray
max_distance_bw_neighbours,min_distance_bw_neighbours = get_maxmin_distance_bw_neighbours(cosine_array)
dataframe=dataframe.assign(max_distance_bw_neighbours=max_distance_bw_neighbours)
dataframe=dataframe.assign(min_distance_bw_neighbours=min_distance_bw_neighbours)

print(countdown)
countdown+=1

def get_avg_distance_bw_points(cosine_array):
    array = []
    for i in range(0,len(cosine_array)):
        dist = 0
        for j in range(0,cosine_array[i].shape[0]):
            for k in range(0,cosine_array[i].shape[1]):
                dist+=cosine_array[i][j][k]
        array.append(dist/(cosine_array[i].shape[0]*cosine_array[i].shape[1]))
    return array
avg_distance_bw_points = get_avg_distance_bw_points(cosine_array)
dataframe=dataframe.assign(avg_distance_bw_points=avg_distance_bw_points)

print(countdown)
countdown+=1

def get_max_distance_bw_points(cosine_array):
    array = []
    for i in range(0,len(cosine_array)):
        maxn = 0
        for j in range(0,cosine_array[i].shape[0]):
            for k in range(0,cosine_array[i].shape[1]):
                if j!=k:
                    maxn = cosine_array[i][j][k] if maxn < cosine_array[i][j][k] else maxn
        array.append(maxn)
    return array
max_distance_bw_points = get_max_distance_bw_points(cosine_array)
dataframe=dataframe.assign(max_distance_bw_points=max_distance_bw_points)

print(countdown)
countdown+=1

def get_clark_and_evans(cosine_array):
    array = []
    for i in range(0,len(cosine_array)):
        dist = 0
        for j in range(0,cosine_array[i].shape[0]):
            curr = 1
            if j-1>=0 and cosine_array[i][j-1][j] < curr :
                curr = cosine_array[i][j-1][j]
            if j+1<cosine_array[i].shape[0] and cosine_array[i][j][j+1] < curr:
                curr = cosine_array[i][j][j+1]
            dist+=curr
        array.append(2*dist/math.sqrt(cosine_array[i].shape[0]))
    return array
clark_and_evans = get_clark_and_evans(cosine_array)
dataframe=dataframe.assign(clark_and_evans=clark_and_evans)

print(countdown)
countdown+=1

def get_avg_distance_to_nearest_neighbours(cosine_array):
    array = []
    for i in range(0,len(cosine_array)):
        dist = 0
        for j in range(0,cosine_array[i].shape[0]):
            curr = 1
            if j-1>=0 and cosine_array[i][j-1][j] < curr :
                curr = cosine_array[i][j-1][j]
            if j+1<cosine_array[i].shape[0] and cosine_array[i][j][j+1] < curr:
                curr = cosine_array[i][j][j+1]
            dist+=curr
        array.append(dist/cosine_array[i].shape[0])
    return array
avg_distance_to_nearest_neighbours = get_avg_distance_to_nearest_neighbours(cosine_array)
dataframe=dataframe.assign(avg_distance_to_nearest_neighbours=avg_distance_to_nearest_neighbours)

print(countdown)
countdown+=1

def get_cumulative_frequency_distribution(cosine_array):
    array = []
    for i in range(0,len(cosine_array)):
        dist = 0
        for j in range(0,cosine_array[i].shape[0]):
            curr = 1
            if j-1>=0 and cosine_array[i][j-1][j] < curr :
                curr = cosine_array[i][j-1][j]
            if j+1<cosine_array[i].shape[0] and cosine_array[i][j][j+1] < curr :
                curr = cosine_array[i][j][j+1]
            if curr <= avg_distance_to_nearest_neighbours[i] :
                dist+=1
        array.append(dist/cosine_array[i].shape[0])
    return array
cumulative_frequency_distribution = get_cumulative_frequency_distribution(cosine_array)
dataframe=dataframe.assign(cumulative_frequency_distribution=cumulative_frequency_distribution)

print(countdown)
countdown+=1

euclidean_array = []
for i in range(0,len(array_of_strings)):
    if(len(array_of_strings[i])<=2) :
        
        euclidean_array.append(np.empty([1,1],dtype=int))
        continue
    tfidf_wm = tfidfvectorizer.fit_transform(array_of_strings[i])
    similarity_matrix = euclidean_distances(tfidf_wm)
    euclidean_array.append(similarity_matrix)

print(countdown)
countdown+=1

avg_distance_bw_neighbours_euclid = get_avg_distance_bw_neighbours(euclidean_array)
dataframe=dataframe.assign(avg_distance_bw_neighbours_euclid=avg_distance_bw_neighbours_euclid)

print(countdown)
countdown+=1

max_distance_bw_neighbours_euclid,min_distance_bw_neighbours_euclid = get_maxmin_distance_bw_neighbours(euclidean_array)
dataframe=dataframe.assign(max_distance_bw_neighbours_euclid=max_distance_bw_neighbours_euclid)
dataframe=dataframe.assign(min_distance_bw_neighbours_euclid=min_distance_bw_neighbours_euclid)

print(countdown)
countdown+=1

avg_distance_bw_points_euclid = get_avg_distance_bw_points(euclidean_array)
dataframe=dataframe.assign(avg_distance_bw_points_euclid=avg_distance_bw_points_euclid)

print(countdown)
countdown+=1

centroid_array = []
for i in range(0,len(array_of_strings)):
    if(len(array_of_strings[i])<=2) :
        
        centroid_array.append(0)
        continue
    tfidf_wm = tfidfvectorizer.fit_transform(array_of_strings[i])
    temp = tfidf_wm[0]
    for j in range(1,tfidf_wm.shape[0]):
        temp+=tfidf_wm[j]
    centroid_array.append(temp/tfidf_wm.shape[0])

print(countdown)
countdown+=1

euclidcentroid_array = []
for i in range(0,len(array_of_strings)):
    if(len(array_of_strings[i])<=2) :
        
        euclidcentroid_array.append(np.empty([1,1],dtype=int))
        continue

    tfidf_wm = tfidfvectorizer.fit_transform(array_of_strings[i])
    similarity_matrix = euclidean_distances(tfidf_wm,centroid_array[i])
    euclidcentroid_array.append(similarity_matrix)

print(countdown)
countdown+=1

def get_avg_distance_bw_centroid_and_points(euclidcentroid_array):
    array = []
    for i in range(0,len(euclidcentroid_array)):
        dist = 0
        for j in range(0,len(euclidcentroid_array[i])):
            dist+=euclidcentroid_array[i][j]
        array.append(dist/len(euclidcentroid_array[i]))
    return array
avg_distance_bw_centroid_and_points = get_avg_distance_bw_centroid_and_points(euclidcentroid_array)
dataframe=dataframe.assign(avg_distance_bw_centroid_and_points=avg_distance_bw_centroid_and_points)

print(countdown)
countdown+=1

def get_maxmin_distance_bw_centroid_and_points(euclidcentroid_array):
    maxarray = []
    minarray = []
    for i in range(0,len(euclidcentroid_array)):
        mx = 0
        mn = 1000
        for j in range(0,len(euclidcentroid_array[i])):
            if mx < euclidcentroid_array[i][j]:
                mx = euclidcentroid_array[i][j]
            if mn > euclidcentroid_array[i][j]:
                mn = euclidcentroid_array[i][j]
        maxarray.append(mx)
        minarray.append(mn)
    return maxarray,minarray
max_distance_bw_centroid_and_points,min_distance_bw_centroid_and_points = get_maxmin_distance_bw_centroid_and_points(euclidcentroid_array)
dataframe=dataframe.assign(max_distance_bw_centroid_and_points=max_distance_bw_centroid_and_points)
dataframe=dataframe.assign(min_distance_bw_centroid_and_points=min_distance_bw_centroid_and_points)

print(countdown)
countdown+=1


dataframe.to_csv("CoherenceFeaturesSet8.csv",index=False)