from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

data = pd.DataFrame({
    'text': [
        "I love this movie",
        "This is a bad product",
        "Awesome experience",
        "Worst thing I ever bought",
        "I am very happy today",
        "I hate this so much",
        "What a great day!",
        "Terrible and boring",
        "Amazing and fantastic!",
        "Disgusting and awful"
    ],
    'label': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
})

texts=data['text'].tolist()
lables=data['label'].tolist()

max_words=1000
tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences=tokenizer.texts_to_sequences(texts)

max_len=10
padded=pad_sequences(sequences,maxlen=max_len)

X_train,X_test,y_train,y_test=train_test_split(padded,lables,random_state=42,test_size=0.2)
X_train=np.array(X_train,dtype='int32')
X_test=np.array(X_test,dtype='int32')
y_train=np.array(y_train)
y_test=np.array(y_test)

model=Sequential()
model.add(Embedding(input_dim=max_words,output_dim=64,input_length=max_len))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=2,epochs=10,validation_data=(X_test,y_test))

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded_seq = pad_sequences(seq, maxlen=max_len)
    prediction = model.predict(padded_seq)[0][0]
    if prediction > 0.5:
        print(f"'{text}' ➝ Positive ({prediction:.2f})")
    else:
        print(f"'{text}' ➝ Negative ({prediction:.2f})")
print("\n--- Predictions ---")
predict_sentiment("I really enjoyed this")
predict_sentiment("This is terrible and awful")
predict_sentiment("I absolutely loved it")