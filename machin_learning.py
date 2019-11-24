from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import keras
import matplotlib.pyplot as plt


print("Beginning to Load the training dataset")
print("...")
# load the training dataset
dataset = loadtxt('/Users/yannick/Desktop/training_data.txt', delimiter='\t')
# split into input (x_train) and output (y_train) variables
x_train = dataset[:,0:7]
y_train = dataset[:,7]
print("Training dataset loaded")

print("Beginning to Load the testing dataset")
print("...")
# load the testing dataset
dataset = loadtxt('//Users/yannick/Desktop/testData.txt', delimiter='\t')
# split into input (x_test) and output (y_test) variables
x_test = dataset[:,0:7].astype(float)
y_test = dataset[:,7]
print(y_test)
print("Testing dataset loaded")


print("Starting to make the model")
#define a Keras model
model = Sequential()
model.add(Dense(12, input_dim=7, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='softmax'))

# make the output classses matrices
y_train = keras.utils.to_categorical(y_train, 8)
y_test = keras.utils.to_categorical(y_test, 8)

print("Compiling the model")
# compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# train the model
# note that we are passing a list of Numpy arrays as training data
# since the model has 2 inputs
print("Training the model")
history = model.fit(x_train, y_train,
                    validation_split=0.25,
                    epochs=5, batch_size=16,
                    verbose=1,
                    validation_data=(x_test, y_test))

print("\nTesting the model")
score = model.evaluate(x_test, y_test, batch_size=16)
print(score)
#keras.utils.plot_model(model, to_file='model.png')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()