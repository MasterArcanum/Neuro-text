import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
import requests

# Замените на URL текстового файла
url = "https://gist.githubusercontent.com/Semionn/bdcb66640cc070450817686f6c818897/raw/f9e8c888a771dd96f54562a9b050acd1138cc7a9/war_and_peace.ru.txt"

# Загрузка содержимого файла
response = requests.get(url)
text = response.text.lower()

# Далее вы можете обрабатывать текст как обычно


#text = open("path_to_your_text_file.txt", "r").read().lower()  # Замените на путь к вашему текстовому файлу
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 40
step = 3
sequences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sequences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

X = np.zeros((len(sequences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sequences), len(chars)), dtype=np.bool)

for i, sequence in enumerate(sequences):
    for t, char in enumerate(sequence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')


model.fit(X, y, batch_size=128, epochs=10)  # Вы можете изменить количество эпох или размер пакета

def generate_text(seed_text, length=400):
    generated = seed_text
    for i in range(length):
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(seed_text):
            x_pred[0, t, char_indices[char]] = 1.
            
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.argmax(preds)
        next_char = indices_char[next_index]
        
        seed_text = seed_text[1:] + next_char
        generated += next_char
    return generated

start_index = np.random.randint(0, len(text) - maxlen - 1)
seed_text = text[start_index: start_index + maxlen]
print(generate_text(seed_text))
