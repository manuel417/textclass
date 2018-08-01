import keras
import keras.preprocessing.text as kpt
from keras.preprocessing.text import Tokenizer
import numpy as np
from sklearn import svm
max_words = 10

def main():

    tweets = [['Trump is crazy'], ['trump is bitching all the asdasda in live'], ['Soccer is too slow'], ['Waste time in World Cup rum booze']]
    train_y = np.array([1, 1, 0, 0])
    train_x = [x[0] for x in tweets]
    tokenizer = Tokenizer(num_words=max_words, oov_token='unk')
    print(train_x)
    tokenizer.fit_on_texts(train_x)
    dictionary = tokenizer.word_index
    print("dictionary: ", dictionary)

    def convert_text_to_index_array(text):
        # one really important thing that `text_to_word_sequence` does
        # is make all texts the same length -- in this case, the length
        # of the longest text in the set.
        result =[]
        for word in kpt.text_to_word_sequence(text):
            print("word: ", word)
            x = dictionary.get(word, 0)
            print("x: ", x)
            result.append(x)
        return result
        #return [dictionary[word] for word in kpt.text_to_word_sequence(text)]

    allWordIndices = tokenizer.texts_to_sequences(train_x)

    allWordIndices = np.asarray(allWordIndices)
    print("allWord 1: ", allWordIndices)
    train_x = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
    print("train_x", train_x)
    print("type x: ", type(train_x))
    print("type y: ", type(train_y))

    # Sciki Learn
    clf = svm.SVC()
    clf.fit(train_x, train_y)

    pred_tweet = ['Trump is live asdasda tu eres juan', 'Trump is asdasda illary', 'Trump is slow Soccer asdasda']
    allWordIndices = tokenizer.texts_to_sequences(pred_tweet)
    allWordIndices = np.asarray(allWordIndices)
    print("allWord 2: ", allWordIndices)
    pred_X = tokenizer.sequences_to_matrix(allWordIndices, mode='binary')
    print("pred X: ", pred_X)
    P = clf.predict(pred_X)
    print("P: ", P)

if __name__ == "__main__":
    main()