


def understanding_data(P_word_real,P_word_fake,cv,N,state):
    feature_names = cv.get_feature_names()
    if(state == "real"):
        part1_real = P_word_real
        part1_real = part1_real.argsort()[-N:][::-1]
        print("  Word/WordPairs")
        for i in range(len(part1_real)):
            name = feature_names[part1_real[i]]
            print("{} {}".format(i,name))
    else:
        part1_fake = P_word_fake
        part1_fake = part1_fake.argsort()[-N:][::-1]
        print("  Word/WordPairs")
        for i in range(len(part1_fake)):
            name = feature_names[part1_fake[i]]
            print("{} {}".format(i,name))
   