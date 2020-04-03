# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data,train_labels),(test_data,test_labels) = data.load_data(num_words = 10000)


word_index = data.get_word_index()

word_index={k:(v+3) for k,v in word_index.items()}

word_index["<PAD>"] = 0 
word_index["<START>"] = 1
word_index["<UK>"] = 2 
word_index["<UNUSED>"] = 3

reverse_word_index  = dict([(value,key) for (key,value) in word_index.items()])

#We Need to lock the max len value
train_data = keras.preprocessing.sequence.pad_sequences(train_data,value = word_index["<PAD>"],padding = "post",maxlen = 250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,value = word_index["<PAD>"],padding = "post",maxlen = 250)

def decodeReview(text):
    return " ".join([reverse_word_index.get(i,"?") for i in text])


#Model Creation Starts from here:::
    
model = keras.Sequential()
model.add(keras.layers.Embedding(88000,16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16,activation = "relu"))
model.add(keras.layers.Dense(1,activation = "sigmoid"))

model.summary()

model.compile(optimizer = "adam",loss = "binary_crossentropy",metrics = ["accuracy"])


x_Val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]


fitModel = model.fit(x_train,y_train,epochs=60,batch_size = 512,validation_data = (x_Val,y_val),verbose = 1)

result = model.evaluate(test_data,test_labels)
print(result)

model.save("model.h5")
