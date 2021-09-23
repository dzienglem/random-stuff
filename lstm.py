from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
#from keras.optimizers import Adam
from keras.utils.data_utils import get_file
import matplotlib.pyplot as plt   # 追加
import numpy as np
import random
import sys
import io
from keras.callbacks import ModelCheckpoint,CSVLogger

import jieba
import jieba.analyse
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, word2vec
import glob
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import axes3d, Axes3D
import os
import re
import numpy as np
import jieba.posseg as pseg
import warnings
import datetime
 
segment_path = 'segment.txt'

sentences = word2vec.LineSentence(segment_path )
# model = word2vec.Word2Vec(sentences, hs=1,min_count=3,window=10,vector_size=100)
w2vmodel = word2vec.Word2Vec(sentences, hs=1,window=10,min_count=0)

maxlen = 8
with io.open(segment_path, encoding='utf-8') as f:
    text = " ".join(f.read().split("\n"))
    #text= re.sub('[^\s\u4e00-\u9fa5]+','',text)
    text = re.sub('[^，。「」\s\u4e00-\u9fa5]+','',text)
    text = text.split(" ")
print('corpus length:', len(text))

text_char = []
for i in range(len(text)):
  text[i] = text[i].replace(u'\xa0', u'')
  if text[i] != "":
    text_char.append(text[i])

# print (text_char[971375:971378])

print('Vectorization...')
x=[]
y=[]

for i in range(0,len(text_char)-maxlen-1,1):
  s=[]
  for j in range(maxlen):
    s.append(w2vmodel.wv[text_char[i+j]])
  x.append(s)
  y.append(w2vmodel.wv[text_char[i+j+1]])
  #print (i)

x = np.array(x)
y=np.array(y)

# print (x.shape)
# print (y.shape)


 # build the model: a single LSTM
print('Build model...')
model = Sequential()
#model.add(LSTM(64, input_shape=(maxlen, y.shape[1]),return_sequences=True))# 不能叠两个lstm
model.add(LSTM(64, input_shape=(maxlen, y.shape[1])))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('softmax'))
 
#optimizer = RMSprop(lr=0.01)
model.compile(loss='cosine_similarity', optimizer='sgd')



def on_epoch_end(epoch, logs):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)
 
    start_index = random.randint(0, len(text_char) - maxlen - 1)
    #start_index = 0  # 毎回、先頭から文章生成
    for diversity in [0.2]:  # diversity = 0.2 のみとする
        print ("time:",datetime.datetime.now())
 
        sentence = []
        for i in range(maxlen):
          sentence.append(text_char[start_index + i])


        print('----- Generating with seed: "' + "".join(sentence) + '"')
        sys.stdout.write("".join(sentence))
 
        for i in range(20):

            x_pred = []
            for char in sentence:
              x_pred.append(w2vmodel.wv[char])

            x_pred = np.array(x_pred)[np.newaxis,:,:]
 
            preds = model.predict(x_pred, verbose=0)[0]

            next_char = w2vmodel.wv.similar_by_vector(preds, topn=1, restrict_vocab=None)[0][0]
 

            sentence.append(next_char)
            sentence = sentence[1:]

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
 
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)



model_checkpoint = ModelCheckpoint("./model.hdf5", monitor='cosine_similarity',mode="max",verbose=0, save_best_only=True)
csv_logger = CSVLogger("./loss.log")
 
history = model.fit(x, y,
                    batch_size=128,
                    epochs=100,
					verbose=0,
                    callbacks=[print_callback, model_checkpoint, csv_logger])
 
# Plot Training loss & Validation Loss
loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, "bo", label = "Training loss" )
plt.title("Training loss")
plt.legend()
plt.savefig("loss.png")
plt.close()
 
 