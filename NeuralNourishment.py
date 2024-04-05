import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

'''PREPROCESSING'''
print('Loading dataset...')
df = pd.read_csv('RecipeNLG_dataset.csv')

print('Preprocessing...')
all_text = ' '.join(df['ingredients'].apply(lambda x: ' '.join(eval(x))).tolist())

chars = sorted(list(set(all_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))

seq_length = 100
dataX = []
dataY = []
for i in range(0, len(all_text) - seq_length, 1):
    seq_in = all_text[i:i + seq_length]
    seq_out = all_text[i + seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
X = np.reshape(dataX, (n_patterns, seq_length, 1)) / float(len(chars))
y = to_categorical(dataY)

'''MODEL'''
print('Creating model...')
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
model.save('neural_nourishment.keras')

'''TRAINING'''
print('Training...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train, epochs=15, batch_size=128, validation_data=(X_test, y_test))
