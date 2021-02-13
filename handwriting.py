import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data() # 10-20% / 80-90% split


x_train = tf.keras.utils.normalize(x_train, axis=1) # normalising data
x_test = tf.keras.utils.normalize(x_test, axis=1)

#take input layer of 2 layers, 2 hidden layers one outpt - Basic Neural Network

model = tf.keras.models.Sequential()

#Flatten Layer
model.add(tf.keras.layers.Flatten(input_shape=(28,28))) #flattens image layer to be 28x28 pixels

# 2 Hidden Layers
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu)) #all neurons are connected to previous and next layer
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))

# Dense output layer
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)) #probability of that number being the result

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Fitting the model - 3 epochs
model.fit(x_train, y_train, epochs=3)

#Accuracy

loss,accuracy = model.evaluate(x_test, y_test)
print(accuracy)
print(loss)

model.save('digits.model')



# Scanning in my own images

for x in range(1, 7):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'The result is most likely: {np.argmax(prediction)}')
    plt.imshow(img[0], cmap=plt.cm.binary) #black on white images
    plt.show()


