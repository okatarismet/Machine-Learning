import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def naive_bayes(X_train,Y_train,n_gram,stop_words):
    if (stop_words == True):
        stop = ENGLISH_STOP_WORDS 
    else:
        stop = None
    cv = CountVectorizer( ngram_range=n_gram,
    token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', stop_words=stop)
    X_train_cv = cv.fit_transform(X_train.to_numpy().astype('U'))
    Y_train_array = Y_train.to_numpy()
    # using np.ones instead of np.zeroes comes for Laplace smoothing
    P_word_real = np.ones(X_train_cv.shape[1])
    P_word_fake = np.ones(X_train_cv.shape[1])
    uniq_len = X_train_cv.shape[1]

    P_real = uniq_len
    P_fake = uniq_len

    for i in range(X_train_cv.shape[0]):
        # Getting each words occurence in real or fake situations
        if(Y_train_array[i] == 1):
            # for P(word|real)
            P_word_real += X_train_cv[i].toarray()[0]
            P_real += 1
        else:
            # for P(word|fake)
            P_word_fake += X_train_cv[i].toarray()[0]
            P_fake += 1
    P_word_real /= P_real
    # now P_World_real is a vector which contains all the conditional probabilities for each word like 
    #[P("ali"|real),P("ahmet"|real),P("veli"|real),P("hasan"|real)]
    P_word_fake /= P_fake
    # P_World_fake is a vector which contains all the conditional probabilities for each word like 
    #[P("ali"|fake),P("ahmet"|fake),P("veli"|fake),P("hasan"|fake)]
    return P_word_real,P_word_fake,P_real,P_fake,uniq_len,cv
