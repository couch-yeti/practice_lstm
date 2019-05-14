import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data() # load data

#  normalize the data 
x_train = tf.keras.utils.normalize(x_train, axis = 1)
x_test = tf.keras.utils.normalize(x_test, axis = 1)


model = tf.keras.models.Sequential()

# if the data is two-dimensional then flatten otherwise look for a 
# different starting layer than flatten

# input layer will accept two dimensional array and flatten to 
# 1xm

model.add(tf.keras.layers.Flatten()) # input layer

model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) # dense layer 

model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu)) # dense layer 


# the output layer has units matching the number of possible predictions
# in this case there are 10 hand drawn digits (0 - 9) so 10 is the number 
# of units given 
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax)) # output layer


# compile the model 
model.compile(optimizer = "adam",
				loss = "sparse_categorical_crossentropy",
				metrics = ["accuracy"])

model.fit(x_train, y_train, epochs=3) # train the model

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)