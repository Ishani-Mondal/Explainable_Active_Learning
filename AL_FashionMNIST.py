#!/usr/bin/env python
# coding: utf-8

# In[113]:
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
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
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D
import numpy as np
import numpy.random as npr
import argparse
import shap
import matplotlib.pyplot as plt
import image
from keras.datasets import fashion_mnist


def fashionMNIST():
    t0 = time.time()
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train.reshape((X_train.shape[0], -1))
    X_test = X_test.reshape((X_test.shape[0], -1))
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
        with open("data/seed_10_FMNIST.pkl", "rb") as fp:
            seed_list = pickle.load(fp)

        with open("data/unlabelled_10_FMNIST.pkl", "rb") as fp:
            unlabelled_list = pickle.load(fp)

    if(seed_size==0.05):
        with open("data/seed_5_FMNIST.pkl", "rb") as fp:
            seed_list = pickle.load(fp)

        with open("data/unlabelled_5_FMNIST.pkl", "rb") as fp:
            unlabelled_list = pickle.load(fp)

    if(seed_size==0.01):
        with open("data/seed_1_FMNIST.pkl", "rb") as fp:
            seed_list = pickle.load(fp)

        with open("data/unlabelled_1_FMNIST.pkl", "rb") as fp:
            unlabelled_list = pickle.load(fp)

    
    #print(seed_list)

    return seed_list, unlabelled_list, X_test, y_test
    

def train(x_seed, y_seed, x_test, y_test, printFlag=True):
    tf.random.set_seed(0)
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
        ulabelled_y.append(str(list(tup[1])))
        
    modified_u = [] # modified unlabeled
    del_s = [] # new points to add to s
    
    ulabelled_X = np.array((np.array(ulabelled_X).reshape(len(ulabelled_X),28,28,1)))
    predictions = clf.predict([ulabelled_X,ulabelled_X])
    
    for i in range(batch_size):
        predictions_label_wise=np.array(predictions)
        uncertainity_list = list(1-predictions_label_wise.max(axis=1))
        max_index = uncertainity_list.index(max(uncertainity_list))
        del_s.append(unlabelled_list[max_index])
        del unlabelled_list[max_index]
        modified_u = unlabelled_list

    return del_s, modified_u


def select_based_on_uncertainity_from_unlabeled(unlabelled_list, batch_size, clf):
    ulabelled_X=[]
    ulabelled_y=[]
    
    for tup in unlabelled_list:
        ulabelled_X.append(list(tup[0]))
        ulabelled_y.append(list(str(tup[1])))
        
    modified_u = [] # modified unlabeled
    del_s = [] # new points to add to s
    
    ulabelled_X = np.array((np.array(ulabelled_X).reshape(len(ulabelled_X),28,28,1)))
    predictions = clf.predict([ulabelled_X,ulabelled_X])
    
    for i in range(batch_size):
        predictions_label_wise=np.array(predictions)
        uncertainity_list = list(1-predictions_label_wise.max(axis=1))
        max_index = uncertainity_list.index(max(uncertainity_list))
        del_s.append(unlabelled_list[max_index])
        del unlabelled_list[max_index]
        modified_u = unlabelled_list
    return del_s, modified_u

def select_based_on_explanation_from_unlabeled(seed_list, unlabelled_list, batch_size, clf):

    ulabelled_X=[]
    ulabelled_y=[]
    
    for tup in unlabelled_list:
        ulabelled_X.append(list(tup[0]))
        ulabelled_y.append(list(str(tup[1])))
       
    ulabelled_X = np.array(ulabelled_X).reshape(len(ulabelled_X),28,28,1)
    ulabelled_y = np.array(ulabelled_y).reshape(len(ulabelled_y),)
    
    
    seed_X=[]
    seed_y=[]

    for tup in seed_list:
        seed_X.append(list(tup[0]))
        seed_y.append(list(str(tup[1])))
    
    seed_X = np.array(seed_X).reshape(len(seed_X),28,28,1)
    seed_y = np.array(seed_y).reshape(len(seed_y),)
    
    explainer = shap.GradientExplainer(clf, [seed_X, seed_X])
    shap_values_seed = explainer.shap_values([seed_X, seed_X])
    
    #print(list(seed_y)[0])
    from matplotlib import pyplot as plt
    #plt.imshow(list(seed_X.reshape(len(seed_X), 28, 28))[0], cmap='gray')
    #plt.imshow()
    
    #predictions = clf.predict([X_test, X_test])
    predictions_seed = np.argmax(clf.predict([seed_X, seed_X]), axis=1)
    predictions_unlabelled = np.argmax(clf.predict([ulabelled_X, ulabelled_X]), axis=1)
    
    
    explainer = shap.GradientExplainer(clf, [ulabelled_X, ulabelled_X])
    shap_values_unlabelled = explainer.shap_values([ulabelled_X, ulabelled_X])
    
    vectors = np.empty([1, 784])
    for i in range(len(predictions_seed)):
        pred_label = predictions_seed[i]
        print(pred_label)
        np.append(vectors, shap_values_seed[int(pred_label)][1][i].reshape(1,784)[0], axis=None)
        #print(shap_values_seed[int(pred_label)][1][i].reshape(1,784))
        #plt.imshow(shap_values_seed[int(pred_label)][1][i].reshape(28,28), cmap='gray')
        #plt.show()
        
    seed_mean = np.mean(vectors, axis=0).reshape(1,784)[0]
    print("Seed_mean="+str(seed_mean.shape))
    
    sums=[]
    vectors = np.empty([1, 784])
    for i in range(len(predictions_unlabelled)):
        #print("==unlabelled==")
        pred_label = predictions_unlabelled[i]
        #print(pred_label)
        shap_value = shap_values_unlabelled[int(pred_label)][1][i].reshape(1,784)[0]
        #print(shap_value.shape)
        sums.append(np.sum(np.subtract(seed_mean, shap_value)))
        
    print(sums)
    
    del_s = []
    for i in range(batch_size):
        max_index=sums.index(max(sums))
        del_s.append(unlabelled_list[max_index])
        del unlabelled_list[max_index]
        del sums[max_index]
        
    return del_s, unlabelled_list


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


def density_based_selection(s, u, batch_size):
    x_seed = []
    for i in range(len(s)):
        x_seed.append(s[i][0])

    #X=np.array(s)
    kmeans = KMeans(n_clusters = 1)
    kmeans.fit(x_seed)
    mean_s = np.array(kmeans.cluster_centers_)

    x_u = []
    for i in range(len(u)):
        x_u.append(u[i][0])

    distance_from_s=[]
    for elem in x_u:
        distance_from_s.append(distance.cosine(elem, mean_s))


    modified_u = [] # modified unlabeled
    del_s = [] # new points to add to s
    
    for i in range(batch_size):
        index=distance_from_s.index(max(distance_from_s))
        del_s.append(u[distance_from_s.index(max(distance_from_s))])
        del u[index]


    return del_s, u

def get_density_based_evaluation(X_train, y_train, X_test, y_test,seed, batch_size):
    print ("Getting MNIST classification data...")
    s, u, X_test, y_test = make_Subset(X_train, y_train, X_test, y_test, seed_size=seed)
    
    batch_size = batch_size
    niters = 10

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
        random_accuracies.append(accuracy)
        seed_set_size.append(len(s))    
        #batch_size=len(s)
        del_s, u = density_based_selection(s, u, batch_size)
        s = s + del_s
        
        final_s = s
    logging.info("Writing Density Results......")
    f=open('MNIST_Accuracy_density_'+str(seed)+'_'+str(batch_size),'w')
    
    for i in range(len(seed_set_size)):
        f.write(str(seed_set_size[i])+'\t'+str(random_accuracies[i])+'\n')

    f.close()

    X_seed=[]
    y_seed=[]


    for tup in final_s:
        X_seed.append(tup[0])
        y_seed.append(tup[1])

    #print(y_seed)
    print('Writing Final Seed of Density..........')
    f=open('Final_seed_of_Density.txt','w')
    for i in range(len(y_seed)):
        #print(y_seed[i])
        f.write(str(i)+"\t"+str(y_seed[i])+"\n")

    X_seed = np.array(X_seed).reshape(len(X_seed),28,28,1)
    y_seed = np.array(y_seed).reshape(len(y_seed),)
            
    import shap

    # since we have two inputs we pass a list of inputs to the explainer
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

def get_random_evaluation(X_train, y_train, X_test, y_test,seed, batch_size):
    print ("Getting MNIST classification data...")
    s, u, X_test, y_test = make_Subset(X_train, y_train, X_test, y_test, seed_size=seed)

    batch_size = batch_size
    niters = 5

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
        
        del_s, u = select_random_from_unlabeled(u, batch_size)
        s = s + del_s
        
        final_s = s
    
    logging.info("Writing Random Results......")
    f=open('Fashion_MNIST_Accuracy_random_'+str(seed)+'_'+str(batch_size),'w')
    
    for i in range(len(seed_set_size)):
        f.write(str(seed_set_size[i])+'\t'+str(random_accuracies[i])+'\n')

    f.close()

    X_seed=[]
    y_seed=[]

    for tup in final_s:
        #print(tup[0].shape)
        X_seed.append(tup[0])
        y_seed.append(tup[1])

    X_seed = np.array(X_seed).reshape(len(X_seed),28,28,1)
    y_seed = np.array(y_seed).reshape(len(y_seed),)
            
    import shap

    # since we have two inputs we pass a list of inputs to the explainer
    explainer = shap.GradientExplainer(clf, [X_seed, X_seed])

    # we explain the model's predictions on the first three samples of the test set
    shap_values = explainer.shap_values([X_test[:3], X_test[:3]])
    #fig = shap.summary_plot(np.array(shap_values), [X_test[:3], X_test[:3]], show=False)
    pl = image.image_plot([shap_values[i][0] for i in range(10)], X_test[:3])
    pl.savefig('MNIST_shap_plot_random_'+str(seed)+'_'+str(batch_size)+'.png')
    #plt.savefig('books_read.png')
    #plt.savefig('shap_random_'+str(seed)+"_"+str(batch_size)+'.png')

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
    f=open('MNIST_Accuracy_uncertain_2_'+str(seed)+'_'+str(batch_size),'w')
    
    for i in range(len(seed_set_size)):
        f.write(str(seed_set_size[i])+'\t'+str(max_uncertainity_accuracies[i])+'\n')


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


def get_uncertainity_evaluation(X_train, y_train, X_test, y_test,seed, batch_size):
    print ("Getting MNIST classification data...")
    s, u, X_test, y_test = make_Subset(X_train, y_train, X_test, y_test, seed)

    batch_size = batch_size
    niters = 5

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

        del_s, u = select_based_on_entropy_uncertainity_from_unlabeled(u, batch_size, clf)
        s = s + del_s
            
        final_s = s

    logging.info("Writing Uncertain Results......")
    f=open('Fashion_MNIST_Accuracy_uncertain_'+str(seed)+'_'+str(batch_size),'w')
    
    for i in range(len(seed_set_size)):
        f.write(str(seed_set_size[i])+'\t'+str(max_uncertainity_accuracies[i])+'\n')


    #print(len(final_s))

    X_seed=[]
    y_seed=[]

    for tup in final_s:
        #print(tup[0].shape)
        X_seed.append(tup[0])
        y_seed.append(tup[1])

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


def get_explanation_based_evaluation_1(X_train, y_train, X_test, y_test,seed, batch_size):

    print ("Getting MNIST classification data...")
    s, u, X_test, y_test = make_Subset(X_train, y_train, X_test, y_test, seed)

    batch_size =  batch_size
    niters = 5

    expl1_accuracies = []
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
        expl1_accuracies.append(accuracy)
        seed_set_size.append(len(s))
        #print(np.argmax(predictions, axis=1))
        #select_based_on_explanation_from_unlabeled(s, u, batch_size, clf)
        del_s, u = select_based_on_explanation_from_unlabeled(s, u, batch_size, clf)
        s = s + del_s
            
        final_s = s


    logging.info("Writing Explanation 1 Results......")
    f=open('Fashion_MNIST_Accuracy_explanation1_'+str(seed)+'_'+str(batch_size),'w')
    
    for i in range(len(seed_set_size)):
        f.write(str(seed_set_size[i])+'\t'+str(expl1_accuracies[i])+'\n')



    print(len(final_s))

    X_seed=[]
    y_seed=[]

    for tup in final_s:
        #print(tup[0].shape)
        X_seed.append(tup[0])
        y_seed.append(tup[1])

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
    pl.savefig('MNIST_shap_plot_explanation1_'+str(seed)+'_'+str(batch_size)+'.png')

    #shap.image_plot([shap_values[i][0] for i in range(10)], X_test[:3])


def get_evaluation_based_explanation_2(X_train, y_train, X_test, y_test,seed_size, batch_size):
    s, u, X_test, y_test = make_Subset(X_train, y_train, X_test, y_test, seed_size)

    niters = 5
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
        del_s, u = explanation_variant_2(s, u, batch_size, clf, X_test, y_test)
        
        s = s + del_s
            
        final_s = s

    logging.info("Writing Explanation 2 Results......")
    f=open('Fashion_MNIST_Accuracy_explanation2_'+str(seed_size)+'_'+str(batch_size),'w')
    
    for i in range(len(seed_set_size)):
        f.write(str(seed_set_size[i])+'\t'+str(expl2_accuracies[i])+'\n')



    print(len(final_s))

    X_seed=[]
    y_seed=[]

    for tup in final_s:
        #print(tup[0].shape)
        X_seed.append(tup[0])
        y_seed.append(tup[1])

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
    pl.savefig('MNIST_shap_plot_explanation2_'+str(seed_size)+'_'+str(batch_size)+'.png')



if __name__ == "__main__":  

    X_train, y_train, X_test, y_test=fashionMNIST()

    batch_sizes=[10,50,100]
    seed_size = [0.01, 0.05, 0.1]

    for i in batch_sizes:
        for j in seed_size:
            print('Random Based', j, i)
            get_random_evaluation(X_train, y_train, X_test, y_test, j, i)
            print('Uncetainity Based 1', j, i)
            get_uncertainity_evaluation(X_train, y_train, X_test, y_test, j, i)
            print('Uncetainity Based 2', j, i)
            get_uncertainity_evaluation_2(X_train, y_train, X_test, y_test, j, i)
            print('Density weighting based', j, i)
            get_density_based_evaluation(X_train, y_train, X_test, y_test, j, i)
            print('Explanation based', j, i)
            get_explanation_based_evaluation_1(X_train, y_train, X_test, y_test, j, i)
            
