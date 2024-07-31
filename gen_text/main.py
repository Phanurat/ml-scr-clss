import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer

# ข้อความตัวอย่าง
text = """สวัสดีครับ"""

# การประมวลผลข้อความ
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
sequences = tokenizer.texts_to_sequences([text])[0]

vocab_size = len(tokenizer.word_index) + 1
sequence_length = 5

# สร้าง sequences
sequences_list = []
for i in range(sequence_length, len(sequences)):
    seq = sequences[i-sequence_length:i+1]
    sequences_list.append(seq)

sequences_list = np.array(sequences_list)
X, y = sequences_list[:,:-1], sequences_list[:,-1]
y = to_categorical(y, num_classes=vocab_size)

# สร้างและฝึกอบรมโมเดล
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=sequence_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, y, epochs=100, verbose=2)

# ฟังก์ชันสำหรับสร้างข้อความใหม่
def generate_text(model, tokenizer, sequence_length, seed_text, num_words):
    result = []
    in_text = seed_text
    for _ in range(num_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = pad_sequences([encoded], maxlen=sequence_length, truncating='pre')
        y_pred = np.argmax(model.predict(encoded, verbose=0), axis=-1)
        predicted_word = ''
        for word, index in tokenizer.word_index.items():
            if index == y_pred:
                predicted_word = word
                break
        in_text += ' ' + predicted_word
        result.append(predicted_word)
    return ' '.join(result)

# สร้างข้อความใหม่
seed_text = "เป็นไงบ้าง"
generated_text = generate_text(model, tokenizer, sequence_length, seed_text, num_words=50)
print(generated_text)
