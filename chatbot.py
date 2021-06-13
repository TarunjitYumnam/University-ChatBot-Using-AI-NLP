import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import re

import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout, Input
from keras.optimizers import Adam, RMSprop
from keras.optimizers import SGD
import random
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, mean_squared_error
from sklearn.model_selection import train_test_split

words = []
classes = []
documents = []
ignore_words = ['?', '!', '&', '(', ')', ',', '.', '/', '-', '`']
data_file = open('intents.json', encoding='utf-8').read()
intents = json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        w = nltk.word_tokenize(pattern)
        words.extend(w)

        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print (len(documents), "documents", documents)

print (len(classes), "classes", classes)

print (len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# training data initialization
training = []
output_empty = [0] * len(classes)
for doc in documents:

    bag = []

    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

print(pattern_words)
print(bag)
print(output_row)
#print(training)
random.shuffle(training)
training = np.array(training)
#print(training)
# create train and test lists. X - patterns, Y - intents
x = list(training[:,0])
y = list(training[:,1])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=5)
print("Training data created")
print(len(x_train))
print(len(x_train[0]))
print(len(y_train))
print(len(y_train[0]))
print(len(y_test))
print(len(x_test))

model = Sequential()
model.add(Dense(512, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(y_train[0]), activation='softmax'))

print(model.summary())

# Functional API

# layer1 = Input(shape=(len(train_x[0]),))
# layer2 = Dense(128, activation='relu')(layer1)
# layer3 = Dense(64, activation='relu')(layer2)
# output = Dense(len(train_y[0], activation='softmax')(layer3)
# func_model = Model(inputs = layer1, outputs = output).+


# Stochastic gradient descent with Nesterov accelerated gradient model categorical_crossentropy, validation_data=(test_x, test_y)
# sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
test_x = []
test_y = []

test_x = np.array(x_test)
test_y = np.array(y_test)

adam = Adam(lr=0.001)
rmsprop = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# fitting and saving the model
hist = model.fit(np.array(x_train), np.array(y_train), epochs=200, batch_size=8, verbose=1, validation_data=(test_x, test_y))

model.save('chatbot_model.h5', hist)

print("model created")

y_predict = model.predict(x_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_predict)))

print("Performance of training data")
print('RMSE is {}'.format(rmse))
print("\n")

y_predict_test = model.predict(x_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_predict_test)))
y_predict_test = (y_predict_test > 0.5)

print("Performance of testing data")
print('RMSE is {}'.format(rmse))
print("\n")

y_predict = np.argmax(model.predict(x,batch_size=8),axis=1)
# test_x_output = np.argmax(test_x_output, axis=1)
test_y = np.argmax(y, axis=1)


predictedValues = confusion_matrix(test_y, y_predict)
sns.heatmap(predictedValues, data = True)

print(predictedValues)

loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training Acc')
plt.plot(epochs, val_acc, 'r', label='Validation Acc')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# plt.figure(figsize=(5,5))
# plt.scatter(y_test, y_predict_test)
# plt.plot([min(y_predict_test.all()), max(y_predict_test.all())], [min(y_predict_test.all()), max(y_predict_test.all())])
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
