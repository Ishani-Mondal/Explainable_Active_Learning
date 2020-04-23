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
random.seed(0)

import numpy as np
np.random.seed(0)

import lime
import lime.lime_tabular
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
np.random.seed(0)
random.seed(0)
tf.random.set_seed(0)

def make_38_Subset(X_train, y_train, X_test, y_test, seed_size):
    '''
    templist = []
    for tup in zip(X_train, y_train):
        if(tup[1]=='8' or tup[1]=='3'):
            templist.append(list(tup))

    testList = []
    for tup in zip(X_test, y_test):
        if(tup[1]=='8' or tup[1]=='3'):
            testList.append(list(tup))

    print("3/8 Training set size = "+str(len(templist)))
    print("3/8 Test set size = "+str(len(testList)))
    
    X_test=[]
    y_test=[]

    for tup in testList:
        #print(tup[0].shape)
        X_test.append(tup[0])
        y_test.append(tup[1])
    
    #seed_size=int(0.1*len(templist))
    seed_s = int(seed_size*len(templist))
    #seed_s = 5
    seed_list=templist[0:seed_s]

    with open('seed_10.pkl','wb') as fp:
        pickle.dump(seed_list, fp)


    print("Initial Seed set size = "+str(len(seed_list)))
    unlabelled_list=templist[seed_s:]
    print("Unlabelled dataset size = "+str(len(unlabelled_list)))
    
    with open('unlabelled_10.pkl','wb') as fp:
        pickle.dump(unlabelled_list, fp)

    with open('X_test.pkl','wb') as fp:
        pickle.dump(X_test, fp)

    with open('y_test.pkl','wb') as fp:
        pickle.dump(y_test, fp)
    
    '''
    print(type(seed_size), seed_size)
    if(seed_size==0.1):
        with open("seed_10.pkl", "rb") as fp:
            seed_list = pickle.load(fp)

        with open("unlabelled_10.pkl", "rb") as fp:
            unlabelled_list = pickle.load(fp)

    if(seed_size==0.05):
        with open("seed_5.pkl", "rb") as fp:
            seed_list = pickle.load(fp)

        with open("unlabelled_5.pkl", "rb") as fp:
            unlabelled_list = pickle.load(fp)

    if(seed_size==0.01):
        with open("seed_1.pkl", "rb") as fp:
            seed_list = pickle.load(fp)

        with open("unlabelled_1.pkl", "rb") as fp:
            unlabelled_list = pickle.load(fp)

    with open("X_test.pkl", "rb") as fp:
        X_test = pickle.load(fp)

    with open("y_test.pkl", "rb") as fp:
        y_test = pickle.load(fp)

    return seed_list, unlabelled_list, X_test, y_test


# In[152]:


def train(x_seed, y_seed, x_test, y_test):
    tf.random.set_seed(0)
    input1 = Input(shape=(28,28,1))
    input2 = Input(shape=(28,28,1))
    tf.random.set_seed(0)
    input2c = Conv2D(32, kernel_size=(3, 3), activation='relu')(input2)
    tf.random.set_seed(0)
    joint = tf.keras.layers.concatenate([Flatten()(input1), Flatten()(input2c)])
    out = Dense(10, activation='softmax')(Dense(128, activation='relu')(joint))
    tf.random.set_seed(0)
    model = tf.keras.models.Model(inputs = [input1, input2], outputs=out)
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    print(y_seed[0:10])
    model.fit([x_seed, x_seed], y_seed, epochs=1)
    print("=======Evaluation========")
    scores = model.evaluate([x_test,x_test], y_test, verbose=0)
    #print(y_test[0:10])
    print(scores[0])
    print("Accuracy on Test Set: "+str(scores[1]*100)+" %")
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
    
    print(len(del_s))
    print(len(modified_u))
    return del_s, modified_u

def select_based_on_explanation_from_unlabeled(seed_list, unlabelled_list, batch_size, clf):
    #print(len(seed_list))
    #print(len(unlabelled_list))
    ulabelled_X=[]
    ulabelled_y=[]
    
    for tup in unlabelled_list:
        ulabelled_X.append(list(tup[0]))
        ulabelled_y.append(list(tup[1]))
       
    ulabelled_X = np.array(ulabelled_X).reshape(len(ulabelled_X),28,28,1)
    ulabelled_y = np.array(ulabelled_y).reshape(len(ulabelled_y),)
    
    
    seed_X=[]
    seed_y=[]

    for tup in seed_list:
        seed_X.append(list(tup[0]))
        seed_y.append(list(tup[1]))
    
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


def select_based_on_explanation_from_unlabeled_2(seed_list, unlabelled_list, batch_size, clf, X_test, y_test):
    #print(len(seed_list))
    #print(len(unlabelled_list))
    ulabelled_X=[]
    ulabelled_y=[]
    
    for tup in unlabelled_list:
        ulabelled_X.append(list(tup[0]))
        ulabelled_y.append(list(tup[1]))
       
    ulabelled_X = np.array(ulabelled_X).reshape(len(ulabelled_X),28,28,1)
    ulabelled_y = np.array(ulabelled_y).reshape(len(ulabelled_y),)
    
    n_seed_add= int(5 * batch_size)
    
    for i in range(n_seed_add):
        del_s, unlabelled_list = select_based_on_uncertainity_from_unlabeled(unlabelled_list, 1, clf)
        seed_list = seed_list + del_s
    
    
    new_x_seed = []
    new_y_seed = []

    for i in range(len(seed_list)):
        new_x_seed.append(seed_list[i][0])
        new_y_seed.append(seed_list[i][1])

    new_x_seed = np.array(new_x_seed).reshape(len(new_x_seed),28,28,1)
    new_x_seed = new_x_seed.astype(float)
    
    new_y_seed = np.array(new_y_seed).reshape(len(new_y_seed),)
    new_y_seed = new_y_seed.astype(float)
    
    X_test = np.array(X_test).reshape(len(X_test),28,28,1)
    X_test = X_test.astype(float)
    y_test = np.array(y_test).reshape(len(y_test),)
    y_test = y_test.astype(float)
    clf2, accuracy = train(new_x_seed, new_y_seed, X_test, y_test)
    new_predictions_seed = np.argmax(clf2.predict([new_x_seed, new_x_seed]), axis=1)
    
    explainer = shap.GradientExplainer(clf2, [new_x_seed, new_x_seed])
    shap_values_seed = explainer.shap_values([new_x_seed, new_x_seed])
    
    vectors = np.empty([1, 784])
    for i in range(len(new_predictions_seed)):
        pred_label = new_predictions_seed[i]
        print(pred_label)
        np.append(vectors, shap_values_seed[int(pred_label)][1][i].reshape(1,784)[0], axis=None)
        
        
    new_seed_mean = np.mean(vectors, axis=0).reshape(1,784)[0]
    print("Seed_mean="+str(new_seed_mean.shape))
    
    
    
    new_ulabelled_X=[]
    new_ulabelled_y=[]
    
    for tup in unlabelled_list:
        new_ulabelled_X.append(list(tup[0]))
        new_ulabelled_y.append(list(tup[1]))
     
    print(len(unlabelled_list))
        
    new_ulabelled_X = np.array(new_ulabelled_X).reshape(len(new_ulabelled_X),28,28,1)
    new_ulabelled_y = np.array(new_ulabelled_y).reshape(len(new_ulabelled_y),)
    
    new_predictions_unlabelled = np.argmax(clf2.predict([new_ulabelled_X, new_ulabelled_X]), axis=1)
    explainer = shap.GradientExplainer(clf2, [new_ulabelled_X, new_ulabelled_X])
    shap_values_unlabelled = explainer.shap_values([new_ulabelled_X, new_ulabelled_X])
    
    sums=[]
    
    vectors = np.empty([1, 784])
    for i in range(len(new_predictions_unlabelled)):
        pred_label = new_predictions_unlabelled[i]
        shap_value = shap_values_unlabelled[int(pred_label)][1][i].reshape(1,784)[0]
        sums.append(np.sum(np.subtract(new_seed_mean, shap_value)))
        
    print(sums)
    
    del_s = []
    for i in range(batch_size):
        max_index=sums.index(max(sums))
        del_s.append(unlabelled_list[max_index])
        del unlabelled_list[max_index]
        del sums[max_index]
  
    return del_s, unlabelled_list



def get_random_evaluation(X_train, y_train, X_test, y_test,seed, batch_size):
    print ("Getting MNIST 3/8 classification data...")
    s, u, X_test, y_test = make_38_Subset(X_train, y_train, X_test, y_test, seed_size=seed)

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
        
        clf, accuracy = train(x_seed, y_seed, X_test, y_test)
        #lf, accuracy = train_mnist(np.array(x_seed), np.array(y_seed), np.array(X_test), np.array(y_test))
        #clf, accuracy = train(x_seed, y_seed, X_test, y_test)
        random_accuracies.append(accuracy)
        seed_set_size.append(len(s))    
        
        del_s, u = select_random_from_unlabeled(u, batch_size)
        s = s + del_s
        
        final_s = s
    
    f=open('Accuracy_random_'+str(seed)+'_'+str(batch_size),'w')
    
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
    pl.savefig('shap_plot_random_'+str(seed)+'_'+str(batch_size)+'.png')
    #plt.savefig('books_read.png')
    #plt.savefig('shap_random_'+str(seed)+"_"+str(batch_size)+'.png')


def get_uncertainity_evaluation(X_train, y_train, X_test, y_test,seed, batch_size):
    print ("Getting MNIST 3/8 classification data...")
    s, u, X_test, y_test = make_38_Subset(X_train, y_train, X_test, y_test, seed)

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
        clf, accuracy = train(x_seed, y_seed, X_test, y_test)
        predictions = clf.predict([X_test, X_test])
        #print(np.argmax(predictions, axis=1))
        max_uncertainity_accuracies.append(accuracy)
        seed_set_size.append(len(s))    

        del_s, u = select_based_on_uncertainity_from_unlabeled(u, batch_size, clf)
        s = s + del_s
            
        final_s = s


    f=open('Accuracy_uncertain_'+str(seed)+'_'+str(batch_size),'w')
    
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
    pl.savefig('shap_plot_uncertain_'+str(seed)+'_'+str(batch_size)+'.png')
    #shap.image_plot([shap_values[i][0] for i in range(10)], X_test[:3])


def get_explanation_based_evaluation_1(X_train, y_train, X_test, y_test,seed, batch_size):

    print ("Getting MNIST 3/8 classification data...")
    s, u, X_test, y_test = make_38_Subset(X_train, y_train, X_test, y_test, seed)

    batch_size =  batch_size
    niters = 4

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
        clf, accuracy = train(x_seed, y_seed, X_test, y_test)
        expl1_accuracies.append(accuracy)
        seed_set_size.append(len(s))
        #print(np.argmax(predictions, axis=1))
        #select_based_on_explanation_from_unlabeled(s, u, batch_size, clf)
        del_s, u = select_based_on_explanation_from_unlabeled(s, u, batch_size, clf)
        s = s + del_s
            
        final_s = s


    f=open('Accuracy_explanation1_'+str(seed)+'_'+str(batch_size),'w')
    
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
    pl.savefig('shap_plot_explanation1_'+str(seed)+'_'+str(batch_size)+'.png')

    #shap.image_plot([shap_values[i][0] for i in range(10)], X_test[:3])


def get_evaluation_based_explanation_2(X_train, y_train, X_test, y_test,seed_size, batch_size):
    s, u, X_test, y_test = make_38_Subset(X_train, y_train, X_test, y_test, seed_size)

    niters = 5

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

        clf, accuracy = train(x_seed, y_seed, X_test, y_test)

        del_s, u = select_based_on_explanation_from_unlabeled_2(s, u, 1, clf, X_test, y_test)
        
        s = s + del_s
            
        final_s = s


    # In[16]:


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
    explainer = shap.GradientExplainer(clf, [X_test, X_test])

    # we explain the model's predictions on the first three samples of the test set
    shap_values = explainer.shap_values([X_test[:3],X_test[:3]])


    shap.image_plot([shap_values[i][0] for i in range(10)], X_test[:3])



if __name__ == "__main__":  
    with open("Train_X.pkl", "rb") as fp:
        X_train = pickle.load(fp)
        
    with open("Train_y.pkl", "rb") as fp:
        y_train = pickle.load(fp)
        
    with open("Test_X.pkl", "rb") as fp:
        X_test = pickle.load(fp)
        
    with open("Test_y.pkl", "rb") as fp:
        y_test = pickle.load(fp) 

    #parser = argparse.ArgumentParser()
    #parser.add_argument('seed_size', required=False)
    #parser.add_argument('batch_size', required=False)
    #parser.add_argument('method', required=False)
    #args = parser.parse_args()

    #if(args.method=='random'):
        #get_random_evaluation(X_train, y_train, X_test, y_test, float(args.seed_size), int(args.batch_size))

    #if(args.method=='uncertain'):
        #get_uncertainity_evaluation(X_train, y_train, X_test, y_test, float(args.seed_size), int(args.batch_size))

    #if(args.method=='exp1'):
        #get_explanation_based_evaluation_1(X_train, y_train, X_test, y_test, float(args.seed_size), int(args.batch_size))   


    batch_sizes=[5,10,20]
    seed_size = [0.01, 0.05, 0.1]

    for i in batch_sizes:
        for j in seed_size:
            print('Random Based', j, i)
            get_random_evaluation(X_train, y_train, X_test, y_test, j, i)
            print('Uncetainity Based', j, i)
            get_uncertainity_evaluation(X_train, y_train, X_test, y_test, j, i)
            print('Explanation based', j, i)
            get_explanation_based_evaluation_1(X_train, y_train, X_test, y_test, j, i)

