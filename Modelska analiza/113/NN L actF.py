from scipy import signal, linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten
import seaborn as sn


matplotlib.style.use('seaborn')
matplotlib.rcParams['savefig.format'] = 'pdf'
matplotlib.rcParams["savefig.directory"] = r"E:\Faks\4.letnik\Modelska_analiza_I\113\slike"

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255


# train_data = np.zeros(10)
# for i in y_test:
#   train_data[i] +=1      

L_list = [0,1,2,3,4,5]
actF_list = ['swish', 'sigmoid', 'tanh']
for actF in actF_list:
  acc_list = []
  acc_train_list = []
  for L in L_list:

    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    for i in range(L):
      model.add(Dense(128, activation=actF))
    model.add(Dense(10))
    
    # model = tf.keras.models.Sequential([
    #   tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    #   tf.keras.layers.Dense(64, activation=actF),
    #   tf.keras.layers.Dense(10)
    # ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )

    b = model.fit(
        x = x_train, 
        y = y_train,
        epochs=3,
    )

    acc = model.evaluate(x_test, y_test)[1]

    acc_list.append(acc)
    acc_train_list.append(b.history['sparse_categorical_accuracy'][-1])
  plt.plot(L_list, acc_list,label=actF + ' - train', marker='o')
  plt.plot(L_list, acc_train_list,label=actF + ' - test', ls='dotted', color= 'C' + str(actF_list.index(actF)))
  print('*************************************************************************************************************')
  print(actF)

plt.legend(frameon = True, fancybox = True, facecolor='white', framealpha=1)
plt.title('Mreža (input -> ... -> $10$; $N_L = 128 \\forall l$), optimizer.: adam($\eta = 0.01$), število epoh: 3')
plt.xlabel('število skritih plastih $L$')
plt.ylabel('natančnost')
plt.show()








# # Plot confusion matrix

# cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
# cm_norm = cm / train_data
# plt.figure(figsize = (10,7))
# sn.heatmap(cm_norm, annot=True, cmap='magma', vmax=0.05)
# plt.xlabel('Predicted')
# plt.ylabel('Truth')
# plt.show()

# # Plot some missclassified
# index = np.where((y_test - y_predicted_labels) != 0)[0] 
  
# for i in index[:10] :
#     loc = index.tolist().index(i)
#     plt.subplot(2,5,loc+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.title('Napoved : ' + '$\\bf{}$'.format((y_predicted_labels[i])) + '\n Resnica : $\\bf{}$'.format(y_test[i]))
#     plt.imshow(x_test[i], cmap='magma')
# plt.show()



