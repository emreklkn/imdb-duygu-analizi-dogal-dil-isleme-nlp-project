import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
#pad sequences kütüphanesinin amacı bazı cümleler 6 kelime bazıları 4 vs vs bunları tek bir sayıya eşitliyeceğiz mesala az olanlara
#boş kelime ekleyeceğiz , kerasta hepsinin girdisinin aynı olması gerekiyor
from keras.models import Sequential 
from keras.layers import Embedding, SimpleRNN, Dense, Activation## embedding int leri belirli boyutta yoğunluk vektörlerine çeviricek



(X_train, Y_train), (X_test, Y_test) = imdb.load_data(path = "ibdb.npz",
                                                       num_words= None,
                                                       skip_top = 0,
                                                       maxlen = None,
                                                       seed = 113,
                                                       start_char = 1,
                                                       oov_char = 2,
                                                       index_from = 3)

print("Type: ", type(X_train))
print("Type: ", type(Y_train))

print("X train shape: ",X_train.shape)
print("Y train shape: ",Y_train.shape)

# %% EDA - keşifsel veri analizi

print("Y train values: ",np.unique(Y_train))
print("Y test values: ",np.unique(Y_test))

unique, counts = np.unique(Y_train, return_counts = True)
print("Y train distribution: ",dict(zip(unique,counts)))

unique, counts = np.unique(Y_test, return_counts = True)
print("Y testdistribution: ",dict(zip(unique,counts)))

plt.figure()
sns.countplot(Y_train)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y train")

plt.figure()
sns.countplot(Y_test)
plt.xlabel("Classes")
plt.ylabel("Freq")
plt.title("Y test")

d = X_train[0]
print(d)
print(len(d))

#train ve test için boş bir liste yap
review_len_train = []
review_len_test = []
# yorumların kaç tane kelimeden oluştuğuna bakıyoruz
for i, ii in zip(X_train, X_test):
    review_len_train.append(len(i))
    review_len_test.append(len(ii))
    
    
# NORMAL DAĞILIMA MI SAHİP BAKIYORUZ 
sns.distplot(review_len_train, hist_kws = {"alpha":0.3})
sns.distplot(review_len_test, hist_kws = {"alpha":0.3})

    
# mode medyan ve mean bakıyoruz
print("Train mean:", np.mean(review_len_train))
print("Train median:", np.median(review_len_train))
print("Train mode:", stats.mode(review_len_train))



#kelimeler index halinde verilmiş mesala 1 index i "the" kelimesine karşılık geliyor
# number of words
word_index = imdb.get_word_index()
print(type(word_index))
print(len(word_index))


# indexlerin hangi kelimeye denk geldiğine nasıl bakabiliriz ?
for keys, values in word_index.items():
    if values == 22:## 22 indexi hangi kelimeye denk geliyorsa onu getirir
        print(keys)
        
        
#herhangi bir bireyi getirme yani cümleyi
def whatItSay(index = 24):
    
    reverse_index = dict([(value, key) for (key, value) in word_index.items()])
    decode_review = " ".join([reverse_index.get(i - 3, "!") for i in X_train[index]])
    print(decode_review)
    print(Y_train[index])
    return decode_review

decoded_review = whatItSay(36)




#preprocess - işleme

num_words = 15000 # sınırlandırıyoruz 15000 adet yorum var artık
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=num_words)

maxlen = 130 # max uzunluk 130 oluyor aşağıdaki 2 satırda da bunu uyguluyor
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

print(X_train[5])

for i in X_train[0:10]:# ilk 10 nu kelime sayısını getiriyor hepsi 130 olmuş
    print(len(i))

decoded_review = whatItSay(5)# 5.yorumu getiriyor boş kısımlara ! geliyor
    


# %% RNN

rnn = Sequential()
rnn.add(Embedding(num_words, 32, input_length = len(X_train[0])))
rnn.add(SimpleRNN(16, input_shape = (num_words,maxlen), return_sequences= False, activation= "relu"))
rnn.add(Dense(1))
rnn.add(Activation("sigmoid"))

print(rnn.summary())
rnn.compile(loss = "binary_crossentropy", optimizer="rmsprop",metrics= ["accuracy"])

history = rnn.fit(X_train, Y_train, validation_data= (X_test, Y_test), epochs=5, batch_size= 128, verbose=1)

score = rnn.evaluate(X_test, Y_test)
print("Accuracy: %",score[1]*100)

plt.figure()
plt.plot(history.history["acc"], label = "Train")
plt.plot(history.history["val_acc"], label = "Test")
plt.title("Acc")
plt.ylabel("Acc")
plt.xlabel("Epochs")
plt.legend()
plt.show()

plt.figure()
plt.plot(history.history["loss"], label = "Train")
plt.plot(history.history["val_loss"], label = "Test")
plt.title("Acc")
plt.ylabel("Acc")
plt.xlabel("Epochs")
plt.legend()
plt.show()    
        