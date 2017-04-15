import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import RMSprop

data = []

with open('data/turkish.txt') as textfile:
    for word in textfile:
        data.append((word.replace('\n', ''), 0))

with open('data/english.txt') as textfile:
    for word in textfile:
        data.append((word.replace('\n', ''), 1))

random.shuffle(data)

test = list(data[:500])
data = list(data[500:])

words = [record[0] for record in data]
labels = [record[1] for record in data]

test_words = [record[0] for record in test]
test_labels = [record[1] for record in test]

char_pool = sorted(set(''.join(words)))
longest = sorted(words, key=len)[-1]
maxlen = len(longest)
word_count = len(data)

# Turkce ve ingilizce.
n_classes = 2

print('Karakter havuzu: {}'.format(", ".join(char_pool)))
print('En uzun kelime: {}'.format(longest))
print('En uzun kelimenin uzunlugu: {}'.format(maxlen))
print('Kelime adedi: {}'.format(word_count))

char_indices = dict((c, i) for i, c in enumerate(char_pool))
indices_char = dict((i, c) for i, c in enumerate(char_pool))

x_data = np.zeros((word_count, maxlen, len(char_pool)), dtype=np.bool)
y_data = np.zeros((word_count, n_classes))

for i_word, word in enumerate(words):
    for i_char, char in enumerate(word):
        x_data[i_word, i_char, char_indices[char]] = 1

for i_label, label in enumerate(labels):
    y_data[i_label, label] = 1

model = Sequential()
model.add(LSTM(8, input_shape=(maxlen, len(char_pool))))
model.add(Dense(n_classes))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)

for iteration in range(1):
    model.fit(x_data, y_data, batch_size=64, nb_epoch=1)

def predict(word):
    '''Verilen kelimenin dillere gore aitlik olasiliklarini hesaplar.'''
    processed_word = np.zeros((1, maxlen, len(char_pool)))
    for i_char, char in enumerate(word):
        processed_word[0, i_char, char_indices[char]] = 1
    prediction = model.predict(processed_word, verbose=0)[0]

    result = {'turk': prediction[0], 'ing': prediction[1]}

    return result

correct_count = 0

for word, label in zip(test_words, test_labels):
    prediction = predict(word)

    pred = 0

    if prediction["turk"] > prediction["ing"]:
        pred = 0
    else:
        pred = 1

    if pred == label:
        correct_count += 1
print(f"Dogruluk oranı {correct_count}/500 = {correct_count/5}")
