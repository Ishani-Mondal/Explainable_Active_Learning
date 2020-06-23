#!/usr/bin/env python
# coding: utf-8

# In[113]:
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state
import sklearn
from sklearn.metrics import roc_auc_score
import random
from scipy.stats import entropy
import numpy as np
import logging

import pickle
import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D
import numpy as np
import numpy.random as npr
import argparse
import shap
import matplotlib.pyplot as plt
import image
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import shap

def getMNIST():
    t0 = time.time()
    train_samples = 60000
    # Load data from https://www.openml.org/d/554
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X.reshape((X.shape[0], -1))
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_samples, test_size=10000)
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("==================")
    X_train=list(X_train)
    X_test=list(X_test)
    print(len(X_train))
    print(len(X_test))
    print("==================")
    
    return X_train, y_train, X_test, y_test

def make_Subset(X_train, y_train, X_test, y_test, seed_size):
    templist = []
    for tup in zip(X_train, y_train):
        templist.append(list(tup))

    testList = []
    for tup in zip(X_test, y_test):
        #if(tup[1]=='8' or tup[1]=='3'):
        testList.append(list(tup))

    logging.info("Training set size = "+str(len(templist)))
    logging.info("Test set size = "+str(len(testList)))
    
    X_test=[]
    y_test=[]

    for tup in testList:
        X_test.append(tup[0])
        y_test.append(tup[1])
    
    if(seed_size==0.1):
        with open("data/seed_10_MNIST.pkl", "rb") as fp:
            seed_list = pickle.load(fp)

        with open("data/unlabelled_10_MNIST.pkl", "rb") as fp:
            unlabelled_list = pickle.load(fp)

    if(seed_size==0.05):
        with open("data/seed_5_MNIST.pkl", "rb") as fp:
            seed_list = pickle.load(fp)

        with open("data/unlabelled_5_MNIST.pkl", "rb") as fp:
            unlabelled_list = pickle.load(fp)

    if(seed_size==0.01):
        with open("data/seed_1_MNIST.pkl", "rb") as fp:
            seed_list = pickle.load(fp)

        with open("data/unlabelled_1_MNIST.pkl", "rb") as fp:
            unlabelled_list = pickle.load(fp)



    final_seed_list=[]
    zero=[]
    one=[]
    two=[]
    three=[]
    four=[]
    five=[]
    six=[]
    seven=[]
    eight=[]
    nine=[]
    for i in seed_list[0:500]:
        if(i[1]=='0'):
            zero.append(i)
        if(i[1]=='1'):
            one.append(i)
        if(i[1]=='2'):
            two.append(i)
        if(i[1]=='3'):
            three.append(i)
        if(i[1]=='4'):
            four.append(i)
        if(i[1]=='5'):
            five.append(i)
        if(i[1]=='6'):
            six.append(i)
        if(i[1]=='7'):
            seven.append(i)
        if(i[1]=='8'):
            eight.append(i)
        if(i[1]=='9'):
            nine.append(i)

    k=5
    final_seed_list=zero[:k]+one[:k]+two[:k]+three[:k]+four[:k]+five[:k]+six[:k]+seven[:k]+eight[:k]+nine[:k]
    print(len(final_seed_list))
    return final_seed_list, unlabelled_list, X_test, y_test

def train(x_seed, y_seed, x_test, y_test, printFlag=True):
    input1 = Input(shape=(28,28,1))
    input2 = Input(shape=(28,28,1))
    input2c = Conv2D(32, kernel_size=(3, 3), activation='relu')(input2)
    joint = tf.keras.layers.concatenate([Flatten()(input1), Flatten()(input2c)])
    out = Dense(10, activation='softmax')(Dense(128, activation='relu')(joint))
    model = tf.keras.models.Model(inputs = [input1, input2], outputs=out)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    model.fit([x_seed, x_seed], y_seed, epochs=1)
    logging.info("=======Evaluation========")
    scores = model.evaluate([x_test,x_test], y_test, verbose=0)
    logging.info(scores[0])
    if(printFlag==True):
        logging.info("Accuracy on Test Set: "+str(scores[1]*100)+" %")
    return model, scores[1]


def select_random_from_unlabeled(u, batch_size):
    np.random.seed(0)
    selected = set(npr.choice(len(u), batch_size, replace=False)) #these will no longer be unlabeled
    
    modified_u = [] # modified unlabeled
    del_s = [] # new points to add to s
    
    i=0
    for x in u:
        if i in selected:
            del_s.append(x)
        else:
            modified_u.append(x)
            
        i=i+1

    return del_s, modified_u

def select_based_on_uncertainity_from_unlabeled(unlabelled_list, batch_size, clf):
    ulabelled_X=[]
    ulabelled_y=[]
    
    for tup in unlabelled_list:
        ulabelled_X.append(list(tup[0]))
        ulabelled_y.append(list(tup[1]))
        
    modified_u = [] # modified unlabeled
    del_s = [] # new points to add to s
    
    ulabelled_X = np.array((np.array(ulabelled_X).reshape(len(ulabelled_X),28,28,1)))
    predictions = clf.predict([ulabelled_X,ulabelled_X])
    #print(predictions[0][3])
    for i in range(batch_size):
        predictions_label_wise=np.array(predictions)
        uncertainity_list = list(1-predictions_label_wise.max(axis=1))
        max_index = uncertainity_list.index(max(uncertainity_list))
        del_s.append(unlabelled_list[max_index])
        del unlabelled_list[max_index]
        modified_u = unlabelled_list

    return del_s, modified_u


def select_based_on_entropy_uncertainity_from_unlabeled(unlabelled_list, batch_size, clf):
    ulabelled_X=[]
    ulabelled_y=[]
    
    for tup in unlabelled_list:
        ulabelled_X.append(list(tup[0]))
        ulabelled_y.append(list(str(tup[1])))
        
    modified_u = [] # modified unlabeled
    del_s = [] # new points to add to s
    
    ulabelled_X = np.array((np.array(ulabelled_X).reshape(len(ulabelled_X),28,28,1)))
    predictions = clf.predict([ulabelled_X,ulabelled_X])
    np_predictions=np.array(predictions)
    unsorted_entropy_list=list(entropy(np_predictions, base=10,axis=1))
    entropy_list=list(sorted(unsorted_entropy_list, reverse=True))
    max_elements=entropy_list[:batch_size]
    
    indices=[]
    for elem in max_elements:
        index = unsorted_entropy_list.index(elem)
        indices.append(index)
        
    for index in indices:
        del_s.append(list(unlabelled_list[index]))
        
    
    for index in list(sorted(indices, reverse=True)):
        del unlabelled_list[index]
        
    return del_s, unlabelled_list



def explanation_variant_1_1(seed_list, unlabelled_list, batch_size, clf, X_test, y_test):
    seed_X=[]
    seed_y=[]

    for tup in seed_list:
        seed_X.append(list(tup[0]))
        seed_y.append(list(str(tup[1])))
    
    seed_X = np.array(seed_X).reshape(len(seed_X),28,28,1)
    seed_y = np.array(seed_y).reshape(len(seed_y),)
    
    ## Generate SHAP based explanations using clf trained on S on the strong labelled S
    predictions_seed = np.argmax(clf.predict([seed_X, seed_X]), axis=1)
    explainer = shap.GradientExplainer(clf, [seed_X, seed_X])
    shap_values_seed_EV1 = explainer.shap_values([seed_X, seed_X])
    
    ## Sample K*B points from unlabelled data using uncertainity sampling
    del_s, u = select_based_on_entropy_uncertainity_from_unlabeled(unlabelled_list, 3*batch_size, clf)
    
    ## New set of unlabelled points
    new_unlabelled_KBS = del_s
    
    #Generate weak labels for the seed set
    new_points_X=[]
    new_points_y=[]

    for tup in new_unlabelled_KBS:
        new_points_X.append(list(tup[0]))
        new_points_y.append(list(tup[1]))
    
    new_points_X = np.array(new_points_X).reshape(len(new_points_X),28,28,1)
    new_points_y = np.array(new_points_y).reshape(len(new_points_y),)
    
    predicted_y = np.argmax(clf.predict([new_points_X, new_points_X]), axis=1)
    
    explainer = shap.GradientExplainer(clf, [new_points_X, new_points_X])
    shap_values_seed_EV2 = explainer.shap_values([new_points_X, new_points_X])
    
    seed_vectors_1 = []
    for i in range(len(predictions_seed)):
        pred_label = predictions_seed[i]
        seed_vectors_1.append(shap_values_seed_EV1[int(pred_label)][1][i].reshape(1,784)[0])
        
    seed_vectors_2 = []
    for i in range(len(predicted_y)):
        pred_label = predicted_y[i]
        seed_vectors_2.append(shap_values_seed_EV2[int(pred_label)][1][i].reshape(1,784)[0])
        
    
    from scipy import spatial
    distance_dictionary={}
    c=0
    
    for elem2 in seed_vectors_2:
        distance=0
        for elem1 in seed_vectors_1:
            distance=distance+spatial.distance.cosine(list(elem1), list(elem2))
        distance_dictionary[c]=distance/len(seed_vectors_1)
        c=c+1
        
    import operator
    sorted_d = dict(sorted(distance_dictionary.items(), key=operator.itemgetter(1),reverse=True))
    indices = list(sorted_d.keys())[0:batch_size]
    
    seed_X_tobeadded=[]
    seed_y_tobeadded=[]
    
    del_s_final = []
    for index in indices:
        del_s_final.append(new_unlabelled_KBS[index])
        
    return del_s_final, u

def get_random_evaluation(X_train, y_train, X_test, y_test,seed, batch_size):
    print ("Getting MNIST classification data...")
    s, u, X_test, y_test = make_Subset(X_train, y_train, X_test, y_test, seed_size=seed)

    batch_size = batch_size
    niters = 2

    random_accuracies = []
    seed_set_size=[]

    for i in range(niters):
        
        print("|S|_{} = {}, |U|_{} = {}".format(i, len(s), i, len(u)))
        x_seed = []
        y_seed = []

        for i in range(len(s)):
            x_seed.append(s[i][0])
            y_seed.append(s[i][1])
        
        x_seed = np.array(x_seed).reshape(len(x_seed),28,28,1)
        x_seed = x_seed.astype(float)
        
        y_seed = np.array(y_seed).reshape(len(y_seed),)
        y_seed = y_seed.astype(float)
        
        X_test = np.array(X_test).reshape(len(X_test),28,28,1)
        X_test = X_test.astype(float)
        
        y_test = np.array(y_test).reshape(len(y_test),)
        y_test = y_test.astype(float)
        
        clf, accuracy = train(x_seed, y_seed, X_test, y_test, printFlag=True)
        #lf, accuracy = train_mnist(np.array(x_seed), np.array(y_seed), np.array(X_test), np.array(y_test))
        #clf, accuracy = train(x_seed, y_seed, X_test, y_test)
        random_accuracies.append(accuracy)
        seed_set_size.append(len(s))    
        #batch_size=len(s)
        del_s, u = select_random_from_unlabeled(u, batch_size)
        s = s + del_s
        
        final_s = s
    
    logging.info("Writing Random Results......")
    f=open('MNIST_Accuracy_random_'+str(seed)+'_'+str(batch_size),'w')
    
    for i in range(len(seed_set_size)):
        f.write(str(seed_set_size[i])+'\t'+str(random_accuracies[i])+'\n')

    f.close()

    X_seed=[]
    y_seed=[]


    for tup in final_s:
        X_seed.append(tup[0])
        y_seed.append(tup[1])

    #print(y_seed)
    print('Writing Final Seed of Random..........')
    f=open('Final_seed_of_random.txt','w')
    for i in range(len(y_seed)):
        #print(y_seed[i])
        f.write(str(i)+"\t"+str(y_seed[i])+"\n")

    X_seed = np.array(X_seed).reshape(len(X_seed),28,28,1)
    y_seed = np.array(y_seed).reshape(len(y_seed),)
            

    explainer = shap.GradientExplainer(clf, [X_seed, X_seed])

def select_based_on_entropy_uncertainity_from_unlabeled(unlabelled_list, batch_size, clf):
    ulabelled_X=[]
    ulabelled_y=[]
    
    for tup in unlabelled_list:
        ulabelled_X.append(list(tup[0]))
        ulabelled_y.append(list(str(tup[1])))
        
    modified_u = [] # modified unlabeled
    del_s = [] # new points to add to s
    
    ulabelled_X = np.array((np.array(ulabelled_X).reshape(len(ulabelled_X),28,28,1)))
    predictions = clf.predict([ulabelled_X,ulabelled_X])
    #print(predictions)
    np_predictions=np.array(predictions)
    unsorted_entropy_list=list(entropy(np_predictions, base=10,axis=1))
    #print(unsorted_entropy_list)
    entropy_list=list(sorted(unsorted_entropy_list, reverse=True))
    max_elements=entropy_list[:batch_size]
    for elem in max_elements:
        index = unsorted_entropy_list.index(elem)
        print(index)
        try:
            del_s.append(list(unlabelled_list[index]))
            del unlabelled_list[index]
        except:
            pass
        modified_u = unlabelled_list
        
    return del_s, modified_u


def get_uncertainity_evaluation_1(X_train, y_train, X_test, y_test,seed, batch_size):
    print ("Getting MNIST classification data...")
    s, u, X_test, y_test = make_Subset(X_train, y_train, X_test, y_test, seed)

    batch_size = batch_size
    niters = 10

    max_uncertainity_accuracies = []
    seed_set_size = []

    for i in range(niters):
        print("|S|_{} = {}, |U|_{} = {}".format(i, len(s), i, len(u)))
        x_seed = []
        y_seed = []

        for i in range(len(s)):
            x_seed.append(s[i][0])
            y_seed.append(s[i][1])

        x_seed = np.array(x_seed).reshape(len(x_seed),28,28,1)
        x_seed = x_seed.astype(float)
        
        y_seed = np.array(y_seed).reshape(len(y_seed),)
        y_seed = y_seed.astype(float)
        
        X_test = np.array(X_test).reshape(len(X_test),28,28,1)
        X_test = X_test.astype(float)
        y_test = np.array(y_test).reshape(len(y_test),)
        y_test = y_test.astype(float)
        clf, accuracy = train(x_seed, y_seed, X_test, y_test, printFlag=True)
        predictions = clf.predict([X_test, X_test])
        #print(np.argmax(predictions, axis=1))
        max_uncertainity_accuracies.append(accuracy)
        seed_set_size.append(len(s))    
        #batch_size=len(s)
        del_s, u = select_based_on_uncertainity_from_unlabeled(u, batch_size, clf)
        s = s + del_s
            
        final_s = s

    logging.info("Writing Uncertain Results......")
    f=open('MNIST_Accuracy_uncertain_'+str(seed)+'_'+str(batch_size),'w')
    
    for i in range(len(seed_set_size)):
        f.write(str(seed_set_size[i])+'\t'+str(max_uncertainity_accuracies[i])+'\n')


    #print(len(final_s))

    X_seed=[]
    y_seed=[]

    for tup in final_s:
        #print(tup[0].shape)
        X_seed.append(tup[0])
        y_seed.append(tup[1])

    print('Writing Final Seed of Uncertain..........')
    f=open('Final_seed_of_k1_uncertain.txt','w')
    for i in range(len(y_seed)):
        f.write(str(i)+"\t"+str(y_seed[i])+"\n")

    X_seed = np.array(X_seed).reshape(len(X_seed),28,28,1)
    y_seed = np.array(y_seed).reshape(len(y_seed),)
            
    print(len(X_seed))  
    X_test = np.array(X_test).reshape(len(X_test),28,28,1)
    y_test = np.array(y_test).reshape(len(y_test),)

    import shap

    # since we have two inputs we pass a list of inputs to the explainer
    explainer = shap.GradientExplainer(clf, [X_seed, X_seed])

    # we explain the model's predictions on the first three samples of the test set
    shap_values = explainer.shap_values([X_test[:3], X_test[:3]])
    pl = image.image_plot([shap_values[i][0] for i in range(10)], X_test[:3])
    pl.savefig('MNIST_shap_plot_uncertain_'+str(seed)+'_'+str(batch_size)+'.png')
    #shap.image_plot([shap_values[i][0] for i in range(10)], X_test[:3])



def get_uncertainity_evaluation_2(X_train, y_train, X_test, y_test,seed, batch_size):
    print ("Getting MNIST classification data...")
    s, u, X_test, y_test = make_Subset(X_train, y_train, X_test, y_test, seed)

    batch_size = batch_size
    niters = 10

    max_uncertainity_accuracies = []
    seed_set_size = []

    for i in range(niters):
        print("|S|_{} = {}, |U|_{} = {}".format(i, len(s), i, len(u)))
        x_seed = []
        y_seed = []

        for i in range(len(s)):
            x_seed.append(s[i][0])
            y_seed.append(s[i][1])

        x_seed = np.array(x_seed).reshape(len(x_seed),28,28,1)
        x_seed = x_seed.astype(float)
        
        y_seed = np.array(y_seed).reshape(len(y_seed),)
        y_seed = y_seed.astype(float)
        
        X_test = np.array(X_test).reshape(len(X_test),28,28,1)
        X_test = X_test.astype(float)
        y_test = np.array(y_test).reshape(len(y_test),)
        y_test = y_test.astype(float)
        clf, accuracy = train(x_seed, y_seed, X_test, y_test, printFlag=True)
        predictions = clf.predict([X_test, X_test])
        #print(np.argmax(predictions, axis=1))
        max_uncertainity_accuracies.append(accuracy)
        seed_set_size.append(len(s))    
        #batch_size=len(s)
        del_s, u = select_based_on_entropy_uncertainity_from_unlabeled(u, batch_size, clf)
        s = s + del_s
            
        final_s = s

    logging.info("Writing Uncertain Results 2......")
    f=open('MNIST_Accuracy_uncertain_2'+str(seed)+'_'+str(batch_size),'w')
    
    for i in range(len(seed_set_size)):
        f.write(str(seed_set_size[i])+'\t'+str(max_uncertainity_accuracies[i])+'\n')


    #print(len(final_s))

    X_seed=[]
    y_seed=[]

    for tup in final_s:
        #print(tup[0].shape)
        X_seed.append(tup[0])
        y_seed.append(tup[1])

    print('Writing Final Seed of Uncertain2.........')
    f=open('Final_seed_of_k2_uncertain.txt','w')
    for i in range(len(y_seed)):
        f.write(str(i)+"\t"+str(y_seed[i])+"\n")

    X_seed = np.array(X_seed).reshape(len(X_seed),28,28,1)
    y_seed = np.array(y_seed).reshape(len(y_seed),)
            
    print(len(X_seed))  
    X_test = np.array(X_test).reshape(len(X_test),28,28,1)
    y_test = np.array(y_test).reshape(len(y_test),)

    import shap

    # since we have two inputs we pass a list of inputs to the explainer
    explainer = shap.GradientExplainer(clf, [X_seed, X_seed])

    # we explain the model's predictions on the first three samples of the test set
    shap_values = explainer.shap_values([X_test[:3], X_test[:3]])
    pl = image.image_plot([shap_values[i][0] for i in range(10)], X_test[:3])
    pl.savefig('MNIST_shap_plot_uncertain_'+str(seed)+'_'+str(batch_size)+'.png')
    #shap.image_plot([shap_values[i][0] for i in range(10)], X_test[:3])




def get_evaluation_based_explanation_1_1(X_train, y_train, X_test, y_test,seed_size, batch_size):
    s, u, X_test, y_test = make_Subset(X_train, y_train, X_test, y_test, seed_size)

    niters = 10
    batch_size=batch_size
    expl2_accuracies = []
    seed_set_size = []

    for i in range(niters):
        print("|S|_{} = {}, |U|_{} = {}".format(i, len(s), i, len(u)))
        x_seed = []
        y_seed = []

        for i in range(len(s)):
            x_seed.append(s[i][0])
            y_seed.append(s[i][1])

        x_seed = np.array(x_seed).reshape(len(x_seed),28,28,1)
        x_seed = x_seed.astype(float)

        y_seed = np.array(y_seed).reshape(len(y_seed),)
        y_seed = y_seed.astype(float)

        X_test = np.array(X_test).reshape(len(X_test),28,28,1)
        X_test = X_test.astype(float)

        y_test = np.array(y_test).reshape(len(y_test),)
        y_test = y_test.astype(float)

        clf, accuracy = train(x_seed, y_seed, X_test, y_test, printFlag=True)
        expl2_accuracies.append(accuracy)
        seed_set_size.append(len(s))
        #batch_size=len(s)
        del_s, u = explanation_variant_1_1(s, u, batch_size, clf, X_test, y_test)
        
        s = s + del_s
            
        final_s = s

    logging.info("Writing Explanation 1.1 Results......")
    f=open('MNIST_Accuracy_explanation1.1_'+str(seed_size)+'_'+str(batch_size),'w')
    
    for i in range(len(seed_set_size)):
        f.write(str(seed_set_size[i])+'\t'+str(expl2_accuracies[i])+'\n')



    print(len(final_s))

    X_seed=[]
    y_seed=[]

    for tup in final_s:
        #print(tup[0].shape)
        X_seed.append(tup[0])
        y_seed.append(tup[1])

    print('Writing Final Seed of Explanation 1..........')
    f=open('Final_seed_of_exp1.txt','w')
    for i in range(len(y_seed)):
        f.write(str(i)+"\t"+str(y_seed[i])+"\n")

    X_seed = np.array(X_seed).reshape(len(X_seed),28,28,1)
    y_seed = np.array(y_seed).reshape(len(y_seed),)
            
    print(len(X_seed))  
    X_test = np.array(X_test).reshape(len(X_test),28,28,1)
    y_test = np.array(y_test).reshape(len(y_test),)

    import shap

    # since we have two inputs we pass a list of inputs to the explainer
    explainer = shap.GradientExplainer(clf, [X_seed, X_seed])

    # we explain the model's predictions on the first three samples of the test set
    shap_values = explainer.shap_values([X_test[:3],X_test[:3]])
    pl = image.image_plot([shap_values[i][0] for i in range(10)], X_test[:3])
    pl.savefig('MNIST_shap_plot_explanation1.1_'+str(seed_size)+'_'+str(batch_size)+'.png')



if __name__ == "__main__":  
    X_train, y_train, X_test, y_test=getMNIST()

    batch_sizes=[10, 50, 100]
    seed_size = [0.01, 0.05, 0.1]
    
    for i in batch_sizes:
        for j in seed_size:
            print('Random Based', j)
            get_random_evaluation(X_train, y_train, X_test, y_test, j, i)
            print('Uncetainity Based 1', j)
            get_uncertainity_evaluation_1(X_train, y_train, X_test, y_test, j, i)
            print('Uncetainity Based 2', j)
            get_uncertainity_evaluation_2(X_train, y_train, X_test, y_test, j, i)
            print('Explanation based Variant 1', j)
            get_explanation_based_evaluation_1(X_train, y_train, X_test, y_test, j, i)
