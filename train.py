import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import tensorflow as tf
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')
import math


#loading the previously generated folds
data_dir = os.getcwd()
data_dir=data_dir+"/data"
print("data_dir:",data_dir)

for k in range(1,11):
    fold_name = 'fold' + str(k)
    print ("\nAdding " + fold_name)
    feature_file = data_dir + "/" + fold_name + "_x.npy"
    labels_file = data_dir + "/" + fold_name + "_y.npy"
    loaded_features = np.load(feature_file)
    loaded_labels = np.load(labels_file)
    print ("New Features: ", loaded_features.shape)
    
    if k==1:
        train_x = loaded_features
        train_y = loaded_labels
        subsequent_fold = True

    else:
        train_x = np.concatenate((train_x, loaded_features))
        train_y = np.concatenate((train_y, loaded_labels))


#dividing the data into train,test and validation
m=train_x.shape[0]

c=6
a=np.count_nonzero(train_y[:,c])
d=5
a=a+np.count_nonzero(train_y[:,d])
e=4
a=a+np.count_nonzero(train_y[:,e])
f=7
a=a+np.count_nonzero(train_y[:,f])
g=8
a=a+np.count_nonzero(train_y[:,g])
#h=2
#a=a+np.count_nonzero(train_y[:,h])
z=3
a=a+np.count_nonzero(train_y[:,z])


x1=np.zeros((m-a,train_x.shape[1]))
y1=np.zeros((m-a,4))
k=0
for i in range(0,m):
    if(np.argmax(train_y[i,:])==c or np.argmax(train_y[i,:])==z or np.argmax(train_y[i,:])==d or np.argmax(train_y[i,:])==g or np.argmax(train_y[i,:])==f or np.argmax(train_y[i,:])==e):
        continue
    else:
        x1[k,:]=train_x[i,:]
        
        b=np.argmax(train_y[i,:])
        if (b<z):
            y1[k,b]=1
        elif(b<e):
            y1[k,b-1]=1
        else:
            y1[k,b-6]=1
        k=k+1
print (x1.shape)
print(y1.shape)

x1_train=x1[:int(math.ceil((m-a))*0.7),:]
x1_cv=x1[int(math.ceil((m-a))*0.7):int(math.ceil((m-a))*0.9),:]
x1_test=x1[int(math.ceil((m-a))*0.9):,:]
y1_train=y1[:int(math.ceil((m-a))*0.7),:]
y1_cv=y1[int(math.ceil((m-a))*0.7):int(math.ceil((m-a))*0.9),:]
y1_test=y1[int(math.ceil((m-a))*0.9):,:]

n_dim = x1_train.shape[1]
n_classes = y1_train.shape[1]
n_hidden_units_1 = n_dim
n_hidden_units_2 = 350 # approx n_dim * 2
n_hidden_units_3 = 175 # half of layer 2


#training the model
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

tf.set_random_seed(0)
np.random.seed(0)

def create_model(activation_function='relu', init_type='normal', optimiser='Adamax', dropout_rate=0.5):
    model = Sequential()
    # layer 1
    model.add(Dense(n_hidden_units_1, input_dim=n_dim, kernel_initializer=init_type, activation=activation_function))
    # layer 2
    model.add(Dense(n_hidden_units_2, kernel_initializer=init_type, activation=activation_function))
    model.add(Dropout(dropout_rate))
    # layer 3
    model.add(Dense(n_hidden_units_3, kernel_initializer=init_type, activation=activation_function))
    model.add(Dropout(dropout_rate))
    # output layer
    model.add(Dense(n_classes, kernel_initializer=init_type, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer=optimiser, metrics=['accuracy'])
    return model

# a stopping function to stop training before we excessively overfit to the training set
earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

model = create_model()

history=model.fit(x1_train, y1_train, validation_data=(x1_cv, y1_cv), callbacks=[earlystop], epochs=20, batch_size=24)


#predicting and evaluating on test data
from sklearn import metrics 
from keras.utils import np_utils

# obtain the prediction probabilities
y_prob = model.predict_proba(x1_test, verbose=0)
y_pred=np.zeros((x1_test.shape[0],1))
for i in range(0,x1_test.shape[0]):
    y_pred[i]=np.argmax(y_prob[i,:])

y_true = np.argmax(y1_test, 1)


roc = metrics.roc_auc_score(y1_test, y_prob)
print ("ROC:",  round(roc,2))

# evaluate the model
score, accuracy = model.evaluate(x1_test, y1_test, batch_size=32)
print("\nAccuracy = {:.2f}".format(accuracy))

p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='micro')
print ("F-Score:", round(f,2))


#generating confusion matrix
from sklearn.metrics import confusion_matrix

labels = ["aircon","horn","child","music"]
print ("Confusion_matrix")
cm = confusion_matrix(y_true, y_pred)

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print ("    " + empty_cell,)
    for label in labels: 
        print ("%{0}s".format(columnwidth) % label,)
    print ()
    # Print rows
    for i, label1 in enumerate(labels):
        print ("    %{0}s".format(columnwidth) % label1,)
        for j in range(len(labels)): 
            cell = "%{0}s".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print (cell,)
        print()
    
print_cm(cm, labels)


#plotting convergence
fig = plt.figure(figsize=(16,8))

print ("History keys:", (history.history.keys()))
# summarise history for training and validation set accuracy
plt.subplot(1,2,1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

# summarise history for training and validation set loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

