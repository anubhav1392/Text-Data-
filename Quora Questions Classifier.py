####### Quora Classifier ################
import pandas as pd
import os
import glob
import re
import numpy as np
#from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
#from textblob import TextBlob
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from collections import Counter
from keras.models import Sequential
from keras.layers import Flatten,Embedding,LSTM,Dense,Dropout,Conv1D,MaxPooling1D
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


path=r'C:\Users\Anu\Downloads\Compressed\all_5'
word_embedding_path=r'C:\Users\Anu\Documents\embeddings\glove.840B.300d\glove.840B.300d.txt'
files=glob.glob(os.path.join(path,'*.csv'))

train_data=pd.read_csv(files[2])
test_data=pd.read_csv(files[1])

train_data=shuffle(train_data)
labels=train_data['target']

#Text Cleaning
def text_cleaning(documents):
    filtered_text=[]
    tmp=[]
    tmp_sp=[]
    #stop_words
    lm=WordNetLemmatizer()
    sp=stopwords.words('english')
    unnecessary_words=['how','when','what','which']
    for sample in unnecessary_words:
        sp.append(sample)
    for i,document in enumerate(documents):
        print('Cleaning Document %d'%(i+1))
        for word in document.split():
            if word not in sp:
                tmp.append(''.join(lm.lemmatize(word.lower(),pos='v')))
        tmp_sp.append(' '.join(tmp))
        tmp=[]
    for sample in tmp_sp:
        filtered_text.append(re.sub(r'[^\w\s]','',sample))
    return filtered_text


train_data_1=text_cleaning(train_data['question_text'])

#Total Words present
words=[]
for sample in train_data_1:
    words.extend(sample.split())
word_counts=Counter(words).most_common()

########################################################
#Parsing Text Data and Word Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

print('Text Data Vectorization..')
tokenize=Tokenizer(num_words=30000)
tokenize.fit_on_texts(train_data_1)
sequences=tokenize.texts_to_sequences(train_data_1)
train_data_prepd=pad_sequences(sequences,maxlen=15)
word_index=tokenize.word_index



import gensim
path=r'C:\Users\Anu\Documents\embeddings\GoogleNews-vectors-negative300\GoogleNews-vectors-negative300.bin'
google_vec=gensim.models.KeyedVectors.load_word2vec_format(path,binary=True)

embedd_matrix=np.zeros((30000,300))
print('Loading Embedding Matrix')
for word,i in word_index.items():
    print('.')
    if i<30000:
        if word in google_vec.vocab:
            embedding_vector=google_vec.word_vec(word)
            embedd_matrix[i]=embedding_vector
print('Null word embeddings: %d' % np.sum(np.sum(embedd_matrix, axis=1) == 0))
            

########################################
print('Model Training')
model=Sequential()
model.add(Embedding(30000,300))
model.add(Dropout(0.3))
model.layers[0].set_weights([embedd_matrix])
model.layers[0].trainable = False
model.add(LSTM(64,activation='tanh',kernel_initializer='he_normal',return_sequences=True,use_bias=True))
model.add(LSTM(128,activation='tanh',kernel_initializer='he_normal',use_bias=True,return_sequences=True))
model.add(LSTM(256,activation='tanh',kernel_initializer='he_normal',use_bias=True))
model.add(Dense(1,activation='sigmoid'))
model.layers[0].set_weights([embedd_matrix])
model.layers[0].trainable=False
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
model.summary()

history=model.fit(train_data_prepd,labels,epochs=5,batch_size=50,validation_split=0.2)
model.save(r'C:\Users\Anu\Downloads\Compressed\all_5\model.h5')

##################################################################################
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'b',color='red', label='Training acc')
plt.plot(epochs, val_acc, 'b',color='blue', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'b', color='red', label='Training loss')
plt.plot(epochs, val_loss, 'b',color='blue', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()










        
        
    


    