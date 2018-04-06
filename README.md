
# Language Classifier

Tries to predict which language a word belongs using LSTMs. 

Dataset contains 100K words. 50K from each: Turkish and English. I haven't included accented letters in Turkish like `ü, ö, ı, ç, ş`, because most of the Turkish words have one of them so it'd make the prediction lot easier. Instead, I've used the most approximate standard Latin letter - like `o` for `ö`.

English X, Q and W (Turkish alphabet doesn't present them) weren't touched since their frequency amongst English words are low.

# Installing

* Clone the repository.
* `cd language-classifier`
* `pipenv install -r %% pipenv shell`
* Then run language-classifier/classifier.py

Requires Python 3.6+

# Explanation

## Imports

* Keras: Deep learning framework
* Numpy: Data storing and manipulation
* Random: Just to shuffle the dataset.

```python
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import RMSprop
```

## Exploring and Preparing the Data

Importing the dataset.

```python
data = []

with open('turkish.txt') as textfile:
    for word in textfile:
        data.append((word.replace('\n', ''), 0))

with open('english.txt') as textfile:
    for word in textfile:
        data.append((word.replace('\n', ''), 1))

random.shuffle(data)
```

Exploring

```python
words = [record[0] for record in data]
labels = [record[1] for record in data]

char_pool = sorted(set(''.join(words)))
longest = sorted(words, key=len)[-1]
maxlen = len(longest)
word_count = len(data)

n_classes = 2

print('Character pool: {}'.format(", ".join(char_pool)))
print('Longest word: {}'.format(longest))
print('Length of the longest word: {}'.format(maxlen))
print('Data size: {} words.'.format(word_count))
```

So the result is..

```
Character pool: a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z
Longest word: trinitrophenylmethylnitramine
Length of the longest word: 29
Data size: 99957
```

Tokenizing char-wise.

```python
char_indices = dict((c, i) for i, c in enumerate(char_pool))
indices_char = dict((i, c) for i, c in enumerate(char_pool))
```

Prepearing the training data. Basically creating a whole size 0 filled tensor, and then filling it with data as the data contains sequential one-hot arrays. Makes it easier for me.

```python
x_data = np.zeros((word_count, maxlen, len(char_pool)), dtype=np.bool)
y_data = np.zeros((word_count, n_classes))

for i_word, word in enumerate(words):
    for i_char, char in enumerate(word):
        x_data[i_word, i_char, char_indices[char]] = 1

for i_label, label in enumerate(labels):
    y_data[i_label, label] = 1
```

## The Predictive Model

```python
model = Sequential()
model.add(LSTM(16, input_shape=(maxlen, len(char_pool))))
model.add(Dense(n_classes))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)
```

## Training

```python
for iteration in range(3):
    model.fit(x_data, y_data, batch_size=128, nb_epoch=1)
```

## Launch!

```python
def predict(word):
    processed_word = np.zeros((1, maxlen, len(char_pool)))
    for i_char, char in enumerate(word):
        processed_word[0, i_char, char_indices[char]] = 1
    prediction = model.predict(processed_word, verbose=0)[0]
    
    result = {'Turkish': prediction[0], 'English': prediction[1]}

    return result
```

Throw any word you want inside this list. It'll be our playing dataset.

```python
# [!] be sure they are all lower-case.
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
    print('{}: {}'.format(word, prediction))
```

## Results

```
altinvarak:	    TUR: 0.98	ENG: 0.02
bulutsuzluk:    TUR: 0.99	ENG: 0.01
farmakoloji:    TUR: 0.97	ENG: 0.03
toprak:	        TUR: 0.90	ENG: 0.10
hanimeli:	    TUR: 0.97	ENG: 0.03
imkansiz:	    TUR: 0.99	ENG: 0.01
tensorflow:	    TUR: 0.00	ENG: 1.00
jabba:	        TUR: 0.75	ENG: 0.25
magsafe:	    TUR: 0.59	ENG: 0.41
pharmacology:   TUR: 0.00	ENG: 1.00
parallax:	    TUR: 0.00	ENG: 1.00
wabby:	        TUR: 0.00	ENG: 1.00
querein:	    TUR: 0.00	ENG: 1.00
terminal:	    TUR: 0.20	ENG: 0.80
ahahahah:	    TUR: 0.83	ENG: 0.17
ahahahahahahahah:TUR: 0.80	ENG: 0.20
rawr:	        TUR: 0.00	ENG: 1.00

Overall Accuracy: 457/500 (91.4)
```
