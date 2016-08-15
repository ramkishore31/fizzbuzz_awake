import numpy as np
from sklearn.metrics import confusion_matrix
from keras.layers import LSTM,Dense, Dropout, Activation
from keras.models import Sequential
import matplotlib.pyplot as plt
import random
from keras.models import model_from_json

def generate_output_data(low,high,data):
    '''
    Generates one hot encoding for each number to represent its output.
    :param low: starting number
    :param high: ending number
    :param data: array of numbers
    :return: array of one-hot encodings for each number to represent its output
    '''
    output_data = []
    for i in range(low, high):
        if data[i] % 15 == 0:
            output_data.append(np.array([0, 0, 0, 1]))
        elif data[i] % 5 == 0:
            output_data.append(np.array([0, 0, 1, 0]))
        elif data[i] % 3 == 0:
            output_data.append(np.array([0, 1, 0, 0]))
        else:
            output_data.append(np.array([1, 0, 0, 0]))
    return np.array(output_data)

def get_binary_encoding(number,digits=10):
    '''
    Generates binary representation for each number.
    :param number: input number
    :param digits: maximum number of digits in binary representation
    :return: binary of the number
    '''
    binary_number = np.binary_repr(number)
    binary_number =  binary_number.zfill(digits)
    binary_number = ','.join(binary_number[i:i+1] for i in range(0, len(binary_number)))
    binary_number = binary_number.split(',')
    binary_number = [int(i) for i in binary_number]
    return np.array(binary_number)

def generate_input_data(low,high,data):
    '''
    Generates binary representation for each number to represent its input.
    :param low: start number
    :param high: ending number
    :param data: array of numbers
    :return: array of binary representations for each number to represent its output
    '''
    input_data = []
    for i in range(low,high):
        input_data.append([get_binary_encoding(data[i])])
    input_data = np.array(input_data)
    return input_data

def build_MLP_model(input_train_data,output_train_data,hidden_layers,input_nodes,hidden_nodes,output_nodes,
                activation_function,optimizer_function,dropout,batch_size,epochs):
    '''
    Build the Multi Layer Perceptron model
    :param input_train_data: Training examples
    :param output_train_data: output for each training example
    :param hidden_layers: Number of hidden layers
    :param input_nodes: Number of nodes in the input layer
    :param hidden_nodes: Number of nodes in the hidden layer
    :param output_nodes: Number of nodes in the output layer
    :param activation_function: Activation function used for hidden layers
    :param optimizer_function: Optimizer function used to find the best weights
    :param dropout: Dropouts used to reduce overfitting
    :param batch_size: Number of batches of training examples to be trained
    :param epochs: Number of total iterations
    :return: Multilayer perceptron Model built
    '''
    model = Sequential()
    model.add(Dense(input_nodes, input_dim=10, init='uniform'))
    model.add(Activation(activation_function))
    if dropout > 0:
        model.add(Dropout(dropout))

    while hidden_layers > 0:
        model.add(Dense(hidden_nodes, init='uniform'))
        model.add(Activation(activation_function))
        if dropout > 0:
            model.add(Dropout(dropout))
        hidden_layers -= 1

    model.add(Dense(output_nodes, init='uniform'))
    model.add(Activation(activation_function))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer_function, metrics=['accuracy'])
    model.fit(input_train_data, output_train_data, batch_size=batch_size, nb_epoch=epochs)
    return model

def build_LSTM_model(input_train_data, output_train_data, hidden_layers, input_nodes, hidden_nodes, output_nodes,
                        activation_function, optimizer_function, dropout, batch_size, epochs):
    '''
    Build the LSTM model
    :param input_train_data: Training examples
    :param output_train_data: output for each training example
    :param hidden_layers: Number of hidden layers
    :param input_nodes: Number of nodes in the input layer
    :param hidden_nodes: Number of nodes in the hidden layer
    :param output_nodes: Number of nodes in the output layer
    :param activation_function: Activation function used for hidden layers
    :param optimizer_function: Optimizer function used to find the best weights
    :param dropout: Dropouts used to reduce overfitting
    :param batch_size: Number of batches of training examples to be trained
    :param epochs: Number of total iterations
    :return: LSTM Model built
    '''
    model = Sequential()
    model.add(LSTM(input_nodes, return_sequences=True, input_shape=(1, 10)))
    model.add(Activation(activation_function))
    if dropout > 0:
        model.add(Dropout(dropout))

    while hidden_layers > 1:
        model.add(LSTM(hidden_nodes, return_sequences=True))
        model.add(Activation(activation_function))
        if dropout > 0:
            model.add(Dropout(dropout))
        hidden_layers -= 1
    model.add(LSTM(hidden_nodes, return_sequences=False))
    model.add(Activation(activation_function))

    model.add(Dense(output_nodes, init='uniform'))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizer_function, metrics=['accuracy'])
    model.fit(input_train_data, output_train_data, batch_size=batch_size, nb_epoch=epochs)
    return model

def get_prediction_values(predicted_output):
    '''
    Converts the softmax output to one hot encoding
    :param predicted_output: Softmax outputs of all predicted numbers
    :return: One-hot encoding of all predicted numbers
    '''
    for i in range(len(predicted_output)):
        cur_max = 0
        cur_max_index = -1
        for j in range(len(predicted_output[i])):
            if cur_max < predicted_output[i][j]:
                cur_max = predicted_output[i][j]
                cur_max_index = j
        for j in range(len(predicted_output[i])):
            if j == cur_max_index:
                predicted_output[i][j] = 1
            else:
                predicted_output[i][j] = 0
    return predicted_output

def get_prediction_accuracy(predicted_output,output_test_data):
    '''
    Computes the % of correct instances
    :param predicted_output: The predicted output values by model
    :param output_test_data: Original output values
    :return:
    '''
    cur_sum = 0
    for i in range(len(predicted_output)):
        if np.array_equal(predicted_output[i], output_test_data[i]):
            cur_sum += 1
    return cur_sum / float(len(predicted_output))

def get_output_label(output_data):
    '''
    Generate output label based on one hot encoding
    :param output_data: Set of outputs
    :return:labels for output
    '''
    output_labels = []
    for i in range(len(output_data)):
        if np.array_equal(output_data[i],np.array([0, 0, 0, 1])):
            output_labels.append("FizzBuzz")
        elif np.array_equal(output_data[i], np.array([0, 0, 1, 0])):
            output_labels.append("Buzz")
        elif np.array_equal(output_data[i], np.array([0, 1, 0, 0])):
            output_labels.append("Fizz")
        else:
            output_labels.append("Others")
    return output_labels

def print_output(output_data):
    '''
    Print labels based on one-hot encoding of numbers
    :param output_data: Set of output numbers
    :return: None
    '''
    for i in range(len(output_data)):
        if np.array_equal(output_data[i], np.array([0, 0, 0, 1])):
            print("FizzBuzz")
        elif np.array_equal(output_data[i], np.array([0, 0, 1, 0])):
            print("Buzz")
        elif np.array_equal(output_data[i], np.array([0, 1, 0, 0])):
            print("Fizz")
        else:
            print i+1

def plot_confusion_matrix(output_test_labels,predicted_output_labels,title='Confusion matrix',name='Confusion matrix'):
    '''

    :param output_test_labels: Original test output
    :param predicted_output_labels: Predicted outputs by the odel
    :param title: title of the confusion matrix
    :param name: name of the confusion matrix
    :return: None
    '''
    labels = ["FizzBuzz","Fizz","Buzz","Others"]
    cm = confusion_matrix(output_test_labels, predicted_output_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plt.imshow(cm_normalized, interpolation='nearest',cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(4)
    plt.xticks(tick_marks,labels, rotation=45)
    plt.yticks(tick_marks,labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(name + '.png')

def save_model(model,name):
    '''
    Save the trained model
    :param model: input model
    :param name: model name
    :return: None
    '''
    json_string = model.to_json()
    open(name + '.json', 'w').write(json_string)
    model.save_weights(name + '.h5')


data = [x for x in range(1,1001)]

input_train_data = generate_input_data(0,1000,data)
output_train_data = generate_output_data(0,1000,data)

model = model_from_json(open('final_model.json').read())
model.load_weights('final_model.h5')

model.compile(optimizer='RMSprop', loss='mse')

predicted_output = model.predict(input_train_data)
predicted_output = get_prediction_values(predicted_output)
print_output(predicted_output)