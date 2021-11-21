import pandas as pd
from keras.models import Sequential
from keras import layers

df = pd.read_csv("c:\\Users\\Piter\\Downloads\\Dimitra1\\amazon_cells_labelled.txt", names=['sentence', 'label'], sep='\t')
#We can see how the first example looks like below
print(df.iloc[0])

# We store the reviews and labels in two arrays as follows:
reviews = df['sentence'].values
labels = df['label'].values

from sklearn.model_selection import train_test_split

reviews_train, reviews_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.20)

from keras.preprocessing.text import Tokenizer

#define the tokenizer: https://keras.io/preprocessing/text/
tokenizer = Tokenizer(num_words=2000)

#Use tokenisation only on the training data!
tokenizer.fit_on_texts(reviews_train)

X_train = tokenizer.texts_to_sequences(reviews_train)
X_test = tokenizer.texts_to_sequences(reviews_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

print(reviews_train[0])
print(X_train[0])
vocab_size

from keras.preprocessing.sequence import pad_sequences

maxlen = 50

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)


def lstm_text_classifier():
    embedding_dim = 50

    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))  # The embedding layer
    model.add(layers.LSTM(4, return_sequences=True, dropout=0.2))
    # To be able to "stack" multiple LSTM layers, we need to return a 3d array output for each input time step
    # We can do this by setting return_sequences to True as above.

    model.add(layers.LSTM(4, dropout=0.2))  # Our last LSTM layer, by removing return_sequences=True,
    # we can pass the output directly to the dense layer. Otherwise, we could have used a Flatten layer.

    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

model = lstm_text_classifier()
training = model.fit(X_train, y_train, epochs=10, verbose=True, validation_split=0.1, batch_size=1)
#details about the model: https://keras.io/models/model/

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

import matplotlib.pyplot as plt

def plot_history(training):
# Plot history: MAE
  plt.plot(training.history['val_loss'], label='Validation loss)')
  plt.plot(training.history['accuracy'], label='Accuracy')
  plt.title('Sentiment analysis')
  plt.ylabel('value')
  plt.xlabel('epoch')
  plt.legend(loc="lower right")
  plt.show()

plot_history(training)