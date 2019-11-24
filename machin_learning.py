from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
import keras
import matplotlib.pyplot as plt


print("Beginning to Load the training dataset")
print("...")
# load the training dataset
# delimiter='\t' => split when a tab arrises
dataset = loadtxt('/Users/yannick/Desktop/training_data.txt', delimiter='\t')
# split into input (x_train) and output (y_train) variables
# this splits the first 8 floats of the line, and puts it in the x_train array
x_train = dataset[:,0:7].astype(float)
# this splits the last of the line, and puts it in the x_train array
y_train = dataset[:,7]
print("Training dataset loaded")

print("Beginning to Load the testing dataset")
print("...")
# load the testing dataset
# delimiter='\t' => split when a tab arrises
dataset = loadtxt('//Users/yannick/Desktop/testData.txt', delimiter='\t')
# split into input (x_test) and output (y_test) variables
x_test = dataset[:,0:7].astype(float)
y_test = dataset[:,7]
print(y_test)
print("Testing dataset loaded")


print("Starting to make the model")
#define a Keras model
model = Sequential()
# first layer has 12 nodes, input layer has dimension 8 (0->7)
model.add(Dense(12, input_dim=7, activation='relu'))
# next layer has 16 nodes
model.add(Dense(16, activation='relu'))
# next layer has 8 nodes
model.add(Dense(8, activation='relu'))
# Output layer has 8 nodes
model.add(Dense(8, activation='softmax'))

# make the output classses matrices
# this makes matrices out of the classes e.g. 1 = [0, 1, 0, 0, ...]
# 8 is the size of the matrix (0->7)
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
# the inputs are your training inputs
print("Training the model")
history = model.fit(x_train, y_train,
                    validation_split=0.25,
                    epochs=5, batch_size=16,
                    verbose=1,
                    validation_data=(x_test, y_test))

# this tests the model
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