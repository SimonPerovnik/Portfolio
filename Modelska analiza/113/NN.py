from scipy import signal, linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import matplotlib.gridspec as gridspec
import tensorflow as tf
import tensorflow_datasets as tfds
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


train_data = np.zeros(10)
for i in y_test:
  train_data[i] +=1      

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(1024, activation='swish'),
  tf.keras.layers.Dense(1024, activation='swish'),
  #tf.keras.layers.Dense(128, activation='softmax'),
  tf.keras.layers.Dense(10)
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #loss='mse',
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    #metrics= ['mae', 'accuracy']
)

b = model.fit(
    x = x_train, 
    y = y_train,
    epochs=10,
    validation_split = 0.2,
    shuffle = 1,
    batch_size = 32
    #validation_data=(x_test, y_test)
)

a = model.evaluate(x_test, y_test)

y_predicted = model.predict(x_test)
y_predicted_labels = [np.argmax(i) for i in y_predicted]

# Plot confusion matrix

cm = tf.math.confusion_matrix(labels=y_test, predictions=y_predicted_labels)
cm_norm = cm / train_data
plt.figure(figsize = (10,7))
sn.heatmap(cm_norm, annot=True, cmap='magma', vmax=0.05)
plt.xlabel('Napoved')
plt.ylabel('Resnica')
plt.show()

# Plot some missclassified
index = np.where((y_test - y_predicted_labels) != 0)[0] 
  
for i in index[:10] :
    loc = index.tolist().index(i)
    plt.subplot(2,5,loc+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.title('Napoved : ' + '$\\bf{}$'.format((y_predicted_labels[i])) + '\n Resnica : $\\bf{}$'.format(y_test[i]))
    plt.imshow(x_test[i], cmap='magma')
plt.tight_layout()
plt.show()

# Plot histogram of missclassfied rate for each number

fig, (ax1,ax2) = plt.subplots(1,2, sharey=True)
cmap = plt.get_cmap('viridis')
N, bins, patches = ax1.hist(y_test[index], bins= range(0,11), rwidth=0.7, align='left', density = True)
for i in range(10):
  patches[i].set_facecolor(cmap( (N[i] - N.min()) / (N.max() - N.min()) ))
ax1.set_xticks(range(10),[str(i) for i in range(10)])
ax1.set_ylabel('$\\frac{N_{napačne}^i}{N_{napačne}}$', rotation=0)
ax1.set_title('Pri katerih številkah se zmotimo')
N, bins, patches = ax2.hist(y_train[index], bins= range(0,11), rwidth=0.7, align='left', density = True)
for i in range(10):
  patches[i].set_facecolor(cmap( (N[i] - N.min()) / (N.max() - N.min()) ))
ax2.set_xticks(range(10),[str(i) for i in range(10)])
ax2.set_title('Za katero številko zamešamo')
plt.show()


