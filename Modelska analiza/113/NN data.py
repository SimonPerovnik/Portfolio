from scipy import signal, linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import matplotlib.gridspec as gridspec
import tensorflow as tf
import tensorflow_datasets as tfds
import seaborn as sn
import matplotlib as mpl


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



fig, (ax1,ax2) = plt.subplots(1,2)
cmap = plt.get_cmap('viridis')
N_test, bins, patches = ax1.hist(y_test, bins= range(0,11), rwidth=0.7, align='left', density = True)
for i in range(10):
  patches[i].set_facecolor(cmap( (N_test[i] - N_test.min()) / (N_test.max() - N_test.min()) ))
ax1.set_xticks(range(10),[str(i) for i in range(10)])
ax1.set_ylabel('delež', rotation=0)
ax1.set_title('Testni podatki')
N, bins, patches = ax2.hist(y_train, bins= range(0,11), rwidth=0.7, align='left', density = True)
for i in range(10):
  patches[i].set_facecolor(cmap( (N[i] - N.min()) / (N.max() - N.min()) ))
ax2.set_xticks(range(10),[str(i) for i in range(10)])
ax2.set_ylabel('delež', rotation=0)
ax2.set_title('Učni podatki')
plt.show()

print(abs(N_test-N))

plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap='magma')
plt.tight_layout()
plt.show()










# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   #tf.keras.layers.Dense(128, activation='softmax'),
#   tf.keras.layers.Dense(10)
# ])

# model.summary()

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(0.001),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     #loss='mse',
#     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
#     #metrics= ['mae', 'accuracy']
# )

# model.fit(
#     x = x_train, 
#     y = y_train,
#     epochs=3,
#     #validation_data=(x_test, y_test)
# )

# model.evaluate(x_test, y_test)

# y_predicted = model.predict(x_test)
# y_predicted_labels = [np.argmax(i) for i in y_predicted]

# cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)

# plt.figure(figsize = (10,7))
# sn.heatmap(cm, annot=True, fmt='d')
# plt.xlabel('Predicted')
# plt.ylabel('Truth')
# plt.show()

# print(np.sum(cm))
