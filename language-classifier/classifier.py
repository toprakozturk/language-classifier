import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import RMSprop
import os

data = [] 

#open and read all the words from the txt file
with open('data/turkish.txt') as textfile:
    for word in textfile:
        data.append((word.replace('\n', ''), 0))

with open('data/english.txt') as textfile:
    for word in textfile:
        data.append((word.replace('\n', ''), 1))

#Shuffle data to maintain homogenity
random.shuffle(data)

#Test train split data
test = list(data[:500])
data = list(data[500:])

#Seperate labels and words from each other
words = [record[0] for record in data]
labels = [record[1] for record in data]

test_words = [record[0] for record in test]
test_labels = [record[1] for record in test]

char_pool = sorted(set(''.join(words)))
longest = sorted(words, key=len)[-1]
maxlen = len(longest)
word_count = len(data)

n_classes = 2

print('Character pool: {}'.format(", ".join(char_pool)))
print('Longest word: {}'.format(longest))
print('Length of the longest word: {}'.format(maxlen))
print('Data size: {} words.'.format(word_count))

char_indices = dict((c, i) for i, c in enumerate(char_pool))
indices_char = dict((i, c) for i, c in enumerate(char_pool))

#Transform the data as required to one hot encodings
x_data = np.zeros((word_count, maxlen, len(char_pool)), dtype=np.bool)
y_data = np.zeros((word_count, n_classes))


#Transforming training words to one hot encoding
for i_word, word in enumerate(words):
    for i_char, char in enumerate(word):
        x_data[i_word, i_char, char_indices[char]] = 1

for i_label, label in enumerate(labels):
    y_data[i_label, label] = 1

#Create the model and compile it.
model = Sequential()
model.add(LSTM(8, input_shape=(maxlen, len(char_pool))))
model.add(Dense(n_classes))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#save different model in Models for later use
base= 'Models'
for iteration in range(1): #You can tune iterations as per the required
    model.fit(x_data, y_data, batch_size=64, nb_epoch=1)
    name='model_iteration_{}.h5'.format(iteration+1)
    model.save(os.path.join(base, name))


def predict(word):
    processed_word = np.zeros((1, maxlen, len(char_pool)))
    for i_char, char in enumerate(word):
        processed_word[0, i_char, char_indices[char]] = 1
    prediction = model.predict(processed_word, verbose=0)[0]

    result = {'tur': prediction[0], 'eng': prediction[1]}

    return result

correct_count = 0

for word, label in zip(test_words, test_labels):
    prediction = predict(word)
    if prediction["tur"] > prediction["eng"]:
        pred = 0
    else:
        pred = 1

    if pred == label:
        correct_count += 1

word_list = [
    # supposed to be Turkish
    'altinvarak',
    'bulutsuzluk',
    'farmakoloji',
    'toprak',
    'hanimeli',
    'imkansiz',

    # supposed to be English
    'tensorflow',
    'jabba',
    'magsafe',
    'pharmacology',
    'parallax',
    'wabby',
    'querein',

    # curiosity
    'terminal', # an actual word in both languages
    'ahahahah',
    'ahahahahahahahah',
    'rawr',
]

for word in word_list:
    prediction = predict(word)
    print('%s\t\tTUR:%.2f\tENG:%.2f'%(word, prediction['tur'], prediction['eng']))

print("Overall Accuracy: %.2f/500 "%(correct_count))
