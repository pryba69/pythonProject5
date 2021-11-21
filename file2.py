# libraries to be included in the project
import string
import pandas as pd
from pandas import Series
from textblob import TextBlob
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

# variable to store stopwords - list of meaningless words that should be removed
stop_set = set(stopwords.words('english'))

# import the data from csv files using pandas
df_train = pd.read_csv("c:\\Users\\Piter\\Downloads\\CourseworkC\\train.csv", names=['opinion', 'label'])
df_test = pd.read_csv("c:\\Users\\Piter\\Downloads\\CourseworkC\\test.csv", names=['opinion', 'label'])
df_validate = pd.read_csv("c:\\Users\\Piter\\Downloads\\CourseworkC\\val.csv", names=['opinion', 'label'])

# https://www.codegrepper.com/code-examples/python/remove+punctuation+in+dataframe+column
def remove_punctuations(text):
   for punctuation in string.punctuation:
       text = text.replace(punctuation, '')
   return text

#https://www.codegrepper.com/code-examples/python/remove+punctuation+in+dataframe+column
df_train['opinion'] = df_train['opinion'].apply(remove_punctuations)
df_test['opinion'] = df_test['opinion'].apply(remove_punctuations)
df_validate['opinion'] = df_validate['opinion'].apply(remove_punctuations)

# https://www.codegrepper.com/code-examples/python/how+to+remove+numbers+from+string+in+python+dataframe
df_train['opinion'] = df_train['opinion'].str.replace('\d+', '')
df_test['opinion'] = df_test['opinion'].str.replace('\d+', '')
df_validate['opinion'] = df_validate['opinion'].str.replace('\d+', '')

# https://www.datasnips.com/58/remove-stop-words-from-text-in-dataframe-column/
df_train['opinion'] = df_train['opinion'].apply(lambda s: ' '.join([word for word in s.split() if word not in (stop_set)]))
df_test['opinion'] = df_test['opinion'].apply(lambda s: ' '.join([word for word in s.split() if word not in (stop_set)]))
df_validate['opinion'] = df_validate['opinion'].apply(lambda s: ' '.join([word for word in s.split() if word not in (stop_set)]))

# checking if the imports were success
print("FIRST CHECK:")
print(df_train.iloc[0])
print(df_test.iloc[0])
print(df_validate.iloc[0])

# checking if the imports were success
print("FIRST CHECK:")
print(df_train.head())
print(df_test.head())
print(df_validate.head())