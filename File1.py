# libraries to be included in the project
import regex
import string
import numpy
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from collections import Counter
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

# variable to store stopwords - list of meaningless words that should be removed
stop_set = set(stopwords.words('english'))

# import the data from csv files using pandas
df_train = pd.read_csv("c:\\Users\\Piter\\Downloads\\CourseworkC\\train.csv", names=['opinion', 'label'])
df_test = pd.read_csv("c:\\Users\\Piter\\Downloads\\CourseworkC\\test.csv", names=['opinion', 'label'])
df_validate = pd.read_csv("c:\\Users\\Piter\\Downloads\\CourseworkC\\val.csv", names=['opinion', 'label'])


# online source: https://www.codegrepper.com/code-examples/python/remove+punctuation+in+dataframe+column
def remove_punctuations(text):
   for punctuation in string.punctuation:
       text = text.replace(punctuation, '')
   return text


# online source: https://www.codegrepper.com/code-examples/python/remove+punctuation+in+dataframe+column
df_train['opinion'] = df_train['opinion'].apply(remove_punctuations)
df_test['opinion'] = df_test['opinion'].apply(remove_punctuations)
df_validate['opinion'] = df_validate['opinion'].apply(remove_punctuations)

# online source: https://www.codegrepper.com/code-examples/python/how+to+remove+numbers+from+string+in+python+dataframe
df_train['opinion'] = df_train['opinion'].str.replace('\d+', '')
df_test['opinion'] = df_test['opinion'].str.replace('\d+', '')
df_validate['opinion'] = df_validate['opinion'].str.replace('\d+', '')

# online source: https://www.datasnips.com/58/remove-stop-words-from-text-in-dataframe-column/
df_train['opinion'] = df_train['opinion'].apply(lambda s: ' '.join([word for word in s.split() if word not in stop_set]))
df_test['opinion'] = df_test['opinion'].apply(lambda s: ' '.join([word for word in s.split() if word not in stop_set]))
df_validate['opinion'] = df_validate['opinion'].apply(lambda s: ' '.join([word for word in s.split() if word not in stop_set]))

# checking if the imports were success
print("FIRST CHECK:")
print(df_train.head())
print(df_test.head())
print(df_validate.head())

# spliting the above files into required datasets
# since we have different files, we do not need to use
# train_test_split module
opinions_train = df_train['opinion'].values
opinions_test = df_test['opinion'].values
opinions_validate = df_validate['opinion'].values

# checking if the imports were success
print("SECOND CHECK:")
print(opinions_train[0])
print(opinions_test[0])
print(opinions_validate[0])

# defining y datasets
y_train = df_train['label'].values
y_test = df_test['label'].values
y_validate = df_validate['label'].values

# checking if the imports were success
print("THIRD CHECK:")
print(y_train[0])
print(y_test[0])
print(y_validate[0])

# define Tokenizer - a function assigning values to words
# num_words parameter specifies the number of words
# to be used for analysis using the matrix
tk = Tokenizer(num_words=20000)  # it can be any number

# fitting the data using Tokenizer
tk.fit_on_texts(opinions_train.tolist())

# check how many words have been fit
print('Unique words found: %d' % len(tk.word_index))

# defining the x datasets
x_train = tk.texts_to_sequences(opinions_train)
x_test = tk.texts_to_sequences(opinions_test)
x_validate = tk.texts_to_sequences(opinions_validate)

# checking if the imports were success
print("THIRD CHECK:")
print(opinions_train[0])
print(x_train[0])
print(opinions_test[0])
print(x_test[0])
print(opinions_test[0])
print(x_validate[0])

# defining vocabulary size
vocabulary_size = len(tk.word_index) + 1  # word index starts at 0 so we need to add 1 to get the real size

# checking the vocabulary size
print('Vocabulary size: %d' % vocabulary_size)

# setting the maximum_word_length - variable keeping the number of characters
# that a vector will contain while analysing words
maximum_word_length = 50  # it can be any number

# converting the words the x datasets into vectors
# using the post truncation processing to make certain
# that all vectors have the same length - padding

x_train = pad_sequences(x_train, padding='post', maxlen=maximum_word_length)
x_test = pad_sequences(x_test, padding='post', maxlen=maximum_word_length)
x_validate = pad_sequences(x_validate, padding='post', maxlen=maximum_word_length)

print('Train data len:' + str(len(x_train)))
print('Class distribution' + str(Counter(y_train)))
print('Validate data len:' + str(len(x_validate)))
print('Class distribution' + str(Counter(y_validate)))


# defining a function to classify the layers: Embedding, LSTM, Dense
def input_classify():
    number_of_embedding_dimentions = 50  # variable
    model = Sequential()  # model
    model.add(layers.Embedding(vocabulary_size, number_of_embedding_dimentions,
                               input_length=maximum_word_length))  # Embedding
    model.add(
        layers.LSTM(4, return_sequences=True, dropout=0.2))  # LSTM layers returning output to following LSTM layers
    model.add(layers.LSTM(4, dropout=0.2))  # the last LSTN layer returning output to the Dense layer
    model.add(layers.Dense(1, activation='sigmoid'))  # the Dense layer
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])  # model compilation characteristics
    model.summary()  # create the summary
    return model


# create model
model = input_classify()

# create trained version of the model
training = model.fit(x_train, y_train, epochs=20, verbose=True, validation_split=0.1, batch_size=1)

# loss and accuracy variables for trained model
train_loss, train_accuracy = model.evaluate(x_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(train_accuracy))

# loss and accuracy variables for test model
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Test Accuracy: {:.4f}".format(test_accuracy))

# loss and accuracy variables for validate model
validate_loss, validate_accuracy = model.evaluate(x_validate, y_validate, verbose=False)
print("Validate Accuracy: {:.4f}".format(validate_accuracy))


# function to create a visualisation of the training model results
def model_visualisation(training):
    plt.plot(training.history['val_loss'], label='Validation loss)')  # presenting the 'val_loss' stats calculated in epochs
    plt.plot(training.history['accuracy'], label='Accuracy')  # presenting the 'accuracy' stats calculated in epochs
    plt.title('Sentiment analysis') # displaying the title of the plot
    plt.ylabel('Value')  # displaying the name of the y-axis
    plt.xlabel('Epoch')  # displaying the name of the x-axis
    plt.legend(loc="lower right")  # location of the legend
    plt.show()  # displaying the plot


# calling the function
model_visualisation(training)