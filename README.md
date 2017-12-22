
# Kelime Sınıflandırıcı

Bu proje, verilen bir kelimenin hangi dilde olmaya daha yatkın olduğunu tahmin etmek için LSTM (long short term memory) denen derin öğrenme metodonu kullanıyor.

Veri olarak 100.000 adet, her iki dilden de 50.000 adet kelime kullanıldı.

Türkçe'ye özel karakterler olan ü, ö, ı, ç, ş gibi harfleri en yakın genel latin karşılığına çevirerek kullandım. Bunun yanında, İngilizce'de bu karakterler nispeten daha az olduğundan, İngilize kelimelerdeki x, w ve q karakterlerine dokunmadım.

## Kullanılan Modüller

* Keras: YSA matematik modellemesi
* Numpy: Veri işleme
* Random: Listeleri karistirmak icin.


```python
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import RMSprop
```

## Veriyi Programa Aktarma

### Dosyadan Listeye

Kelimeleri etiketleriyle birilkte data listesine ekliyorum.

Sinif etiketlerini belirtmesi icin 0 ve 1 kullandim.

Öğrenme sırasında verilerin sıralı düzen içinde verilmesi modelin optimizasyonunun kötü olmasına sebebiyet verebilir. Dolayısıyla listeyi karıştırmakta fayda var.


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

### Öncü Veri Taraması


```python
words = [record[0] for record in data]
labels = [record[1] for record in data]

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
```

    Karakter havuzu: a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z
    En uzun kelime: trinitrophenylmethylnitramine
    En uzun kelimenin uzunlugu: 29
    Kelime adedi: 99957


### Ön İşleme

Tüm karakterlere bir index atıyorum (ve vice-versa).


```python
char_indices = dict((c, i) for i, c in enumerate(char_pool))
indices_char = dict((i, c) for i, c in enumerate(char_pool))
```

Kelime/Karakter tipindeki veriyi sayısal diziler şekline getiriyorum.


```python
# Her harf icin maxlen uzunlugunda 0'lar ile dolu bir vektor
# Her kelime icin icinde o listelerin oldugu baska bir liste
# Tum veriler icin de kelimelere ait listeleri barindiran ana bir liste
# olusturacak sekilde 3 boyutlu bos bir dizi tanimlaniyor.
x_data = np.zeros((word_count, maxlen, len(char_pool)), dtype=np.bool)

# Toplam kelime sayisi ve cikti sayisina gore bir liste tanimlaniyor.
y_data = np.zeros((word_count, n_classes))

# [0, 0, ..., 0] seklinde olan karakter vektorundeki bir degeri karak-
# terin index'ine gore 1 yapiyor. Dolayisiyla en sonunda elde bir ke-
# lime icin sirayla dizilmis one-hot diziler kaliyor.
for i_word, word in enumerate(words):
    for i_char, char in enumerate(word):
        x_data[i_word, i_char, char_indices[char]] = 1

# [0, 0] olan etiket listesini duruma gore [0, 1] veya [1, 0] haline
# getiriyor.
for i_label, label in enumerate(labels):
    y_data[i_label, label] = 1
```

## Modeli Yaratma


```python
model = Sequential()
model.add(LSTM(16, input_shape=(maxlen, len(char_pool))))
model.add(Dense(n_classes))
model.add(Activation('softmax'))

# geri donuslu yapay aglar icin genelde kullanilan optimizer
optimizer = RMSprop(lr=0.01)

model.compile(loss='categorical_crossentropy', optimizer=optimizer)
```

## Ogrenim


```python
for iteration in range(3):
    model.fit(x_data, y_data, batch_size=128, nb_epoch=1)
print('Ogrenim tamamlandi!')
```

## Deneme!


```python
def predict(word):
    '''Verilen kelimenin dillere gore aitlik olasiliklarini hesaplar.'''
    processed_word = np.zeros((1, maxlen, len(char_pool)))
    for i_char, char in enumerate(word):
        processed_word[0, i_char, char_indices[char]] = 1
    prediction = model.predict(processed_word, verbose=0)[0]
    
    result = {'turk': prediction[0], 'ing': prediction[1]}

    return result
```


```python
# [!] kelimelerin kucuk harflerle yazilmis olmasi gerekiyor.
word_list = [
    'toprak',
    'enginar',
    'ornitorenk',
    'frontier',
    'television',
    'facebook',
    # anlamsiz kelimeler
    'ahahahah',
    'xtr',
    'rabara',
    'fizyoloji',
    'physiology'
]

for word in word_list:
    prediction = predict(word)
    print('{}: {}'.format(word, prediction))
```

    toprak: {'turk': 0.92372286, 'ing': 0.076277196}
    enginar: {'turk': 0.28950188, 'ing': 0.71049809}
    ornitorenk: {'turk': 0.025209611, 'ing': 0.97479033}
    frontier: {'turk': 0.000279842, 'ing': 0.99972016}
    television: {'turk': 0.00035036309, 'ing': 0.99964964}
    facebook: {'turk': 0.00049878447, 'ing': 0.99950123}
    ahahahah: {'turk': 0.98772639, 'ing': 0.012273617}
    xtr: {'turk': 0.00017934592, 'ing': 0.99982065}
    rabara: {'turk': 0.72516716, 'ing': 0.2748329}
    fizyoloji: {'turk': 0.99369228, 'ing': 0.0063076839}
    physiology: {'turk': 0.00017088205, 'ing': 0.99982905}


# Sonuclar
* x, q ve w gibi Ingilice'ye has karakterlerin varligini kavramis
* Turkcede -loji, Ingilizce'de -logy gibi eklerin kelime siniflandirmasinda onemli yer tuttugunu anlamis.
* Gülmek daha Türkçemsi bir eylemmiş...
