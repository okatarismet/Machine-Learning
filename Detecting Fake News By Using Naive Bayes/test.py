import numpy as np
results = []
def All_Results():
    for i in results:
        print(i)
def run_test(msg,X_test_array,Y_test_array,P_word_real, P_word_fake, P_real, P_fake, uniq_len ,cv):
    acc = 0
    for i in range(X_test_array.shape[0]):
        text_cv = cv.transform(X_test_array[i].split(" ")).toarray()[0]
        # Each word times its conditional probability if "ali come ali" P(ali|real)*2 P(come|real)*1
        temp = (text_cv * P_word_real)
        # for logarithmic caluclations we have to change zeroes with ones
        temp[temp == 0] = 1
        #  We compute the log probabilities to prevent numerical underflow when calculating multiplicative probabilities.
        # ((P_real - uniq_len)/uniq_len)) is the P(real) part
        real = np.exp(np.sum(np.log(temp) * ((P_real - uniq_len)/uniq_len)))

        # SAME FOR FAKES
        temp = (text_cv * P_word_fake)
        temp[temp == 0] = 1
        fake = np.exp(np.sum(np.log(temp) * ((P_fake - uniq_len)/uniq_len)))

        predict =  1 if real>fake else 0
        if(predict-Y_test_array[i]== 0):
            acc+=1
    print(msg)
    print("Correct classified news:{} out of {}".format(acc,X_test_array.shape[0]))
    print("Accuracy: {}".format(acc/X_test_array.shape[0]*100))
    print("####################################")
    results.append(msg)
    results.append("Accuracy: {}".format(acc/X_test_array.shape[0]*100))
    results.append("Correct classified news: {} out of {}".format(acc,X_test_array.shape[0]))
    results.append("####################################")
    
    
    return 0
def understanding_data(P_word_real,P_word_fake,cv):
    feature_names = cv.get_feature_names()
    part1_real = P_word_real
    part1_real = part1_real.argsort()[-3:][::-1]
    print("3 most appeared words in real news:")
   
    for i in part1_real:
        print("\t- "+feature_names[i])
       

    part1_fake = P_word_fake
    part1_fake = part1_fake.argsort()[-3:][::-1]
    print("3 most appeared words in fake news:")
   
    for i in part1_fake:
        print("\t- "+feature_names[i])
        
    return 0


