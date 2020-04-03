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

model = keras.models.load_model("model.h5")

def review_encode(s):
    encoded = [1]
    for word in s:
        if word in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded
            

with open("test.txt",encoding = "utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",","").replace(".","").replace("(","").replace(")","").replace(":","").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode],value = word_index["<PAD>"],padding = "post",maxlen = 250)
        predict = model.predict(encode)
        print(line)
        #print(encode)
        print(predict[0])
        if predict[0] < 0.2:
            print("Our model predicts 1 Star to this movie")
        elif (predict[0] > 0.2) &  (predict[0] < 0.4):
            print("Our model predicts 2 Star to this movie")
        elif (predict[0] > 0.4) &  (predict[0] < 0.6):
            print("Our model predicts 3 Star to this movie")
        elif (predict[0] > 0.6) &  (predict[0] < 0.8):
            print("Our model predicts 4 Star to this movie")
        elif predict[0] > 0.8:
            print("Our model predicts 5 Star to this movie")
        else:
            print("NO")
        
with open("test2.txt",encoding = "utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",","").replace(".","").replace("(","").replace(")","").replace(":","").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode],value = word_index["<PAD>"],padding = "post",maxlen = 250)
        predict = model.predict(encode)
        print(line)
        #print(encode)
        print(predict[0])
        if predict[0] < 0.2:
            print("Our model predicts 1 Star to this movie")
        elif (predict[0] > 0.2) &  (predict[0] < 0.4):
            print("Our model predicts 2 Star to this movie")
        elif (predict[0] > 0.4) &  (predict[0] < 0.6):
            print("Our model predicts 3 Star to this movie")
        elif (predict[0] > 0.6) &  (predict[0] < 0.8):
            print("Our model predicts 4 Star to this movie")
        elif predict[0] > 0.8:
            print("Our model predicts 5 Star to this movie")
        else:
            print("NO")
        
        
with open("test3.txt",encoding = "utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",","").replace(".","").replace("(","").replace(")","").replace(":","").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode],value = word_index["<PAD>"],padding = "post",maxlen = 250)
        predict = model.predict(encode)
        print(line)
        #print(encode)
        print(predict[0])
        if predict[0] < 0.2:
            print("Our model predicts 1 Star to this movie")
        elif (predict[0] > 0.2) &  (predict[0] < 0.4):
            print("Our model predicts 2 Star to this movie")
        elif (predict[0] > 0.4) &  (predict[0] < 0.6):
            print("Our model predicts 3 Star to this movie")
        elif (predict[0] > 0.6) &  (predict[0] < 0.8):
            print("Our model predicts 4 Star to this movie")
        elif predict[0] > 0.8:
            print("Our model predicts 5 Star to this movie")
        else:
            print("NO")
        