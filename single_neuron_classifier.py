import argparse
import numpy as np
import math
import matplotlib.pyplot as plt


# Command Line Arguments
# These commands essentially allow running of the file from command line by specifying the key parameters 

parser = argparse.ArgumentParser(description='generate training and validation data for assignment 2')
parser.add_argument('trainingfile', help='name stub for training data and label output in csv format', default="train")
parser.add_argument('validationfile', help='name stub for validation data and label output in csv format',
                    default="valid")
parser.add_argument('numtrain', help='number of training samples', type=int, default=200)
parser.add_argument('numvalid', help='number of validation samples', type=int, default=20)
parser.add_argument('-seed', help='random seed', type=int, default=1)
parser.add_argument('-learningrate', help='learning rate', type=float, default=0.1)
parser.add_argument('-actfunction', help='activation functions', choices=['sigmoid', 'relu', 'linear'],
                    default='linear')
parser.add_argument('-numepoch', help='number of epochs', type=int, default=50)

args = parser.parse_args()

traindataname = args.trainingfile + "data.csv"
trainlabelname = args.trainingfile + "label.csv"

print("training data file name: ", traindataname)
print("training label file name: ", trainlabelname)

validdataname = args.validationfile + "data.csv"
validlabelname = args.validationfile + "label.csv"

print("validation data file name: ", validdataname)
print("validation label file name: ", validlabelname)

print("number of training samples = ", args.numtrain)
print("number of validation samples = ", args.numvalid)

print("learning rate = ", args.learningrate)
print("number of epoch = ", args.numepoch)

print("activation function is ", args.actfunction)

# Set some global variables
seed = args.seed  # This is the random seed generator value we set so as to allow reproducible results
lr = args.learningrate
epochs = args.numepoch
activation = str(args.actfunction)

# Get training data and validation data
train_data = np.array(np.loadtxt(traindataname, delimiter=","))
train_label = np.array(np.loadtxt(trainlabelname, delimiter=","))

valid_data = np.array(np.loadtxt(validdataname, delimiter=","))
valid_label = np.array(np.loadtxt(validlabelname, delimiter=","))

# Randomize instantiation of weights and bias
np.random.seed(seed)
weights = np.random.rand(9, 1)
np.random.seed(seed)
bias = np.random.random()


# Define Activation functions for ReLU and Sigmoid as well as our accuracy function
def sigmoid(x):
    x = np.array(x)
    return 1 / (1 + np.power(math.e, -1 * x))


def relu(x):
    x = np.array(x)
    return np.where(x > 0, x, 0)


def accuracy(prediction, label):  #Strictly speaking, using such a naive accuracy measure isn't the best call, but it serves for this simple project
    num_correct = 0
    prediction = np.array(prediction)
    prediction = np.where(prediction >= 0.5, 1, 0)
    for k in range(0,len(label),1):
        if prediction[k] == label[k]:
            num_correct += 1.0
    acc = float(num_correct/len(label))
    return acc


# Now we create some arrays to store data from training so we can plot them
loss_array = [0]*epochs
accuracy_array = [0]*epochs
validation_loss = [0]*epochs
validation_accuracy = [0]*epochs

predictions = np.zeros_like(train_label)  # This array stores the output of our model every training epoch
validation_prediction = np.zeros_like(valid_label)  # Similar as above, but for the validation data 
gradient_tensor = np.zeros((predictions.shape[0], 10))  # So a 200 row, 10 column matrix.
                                                        # Each row is a training point,and each column is a parameter (w_0 --- w_8 and b)

for i in range(epochs):
    if activation == 'linear':
        predictions += np.squeeze(train_data.dot(weights) + bias)
        validation_prediction += np.squeeze(valid_data.dot(weights) + bias)
    if activation == 'sigmoid':
        predictions += np.squeeze(sigmoid(train_data.dot(weights) + bias))
        validation_prediction += np.squeeze(sigmoid(valid_data.dot(weights) + bias))
    if activation == 'relu':
        predictions += np.squeeze(relu(train_data.dot(weights) + bias))
        validation_prediction += np.squeeze(relu(valid_data.dot(weights) + bias))

    loss = np.power((predictions - train_label), 2)  # This is an array for whole training set. Utilizing mean squared error

    # Collect information for graphing later on
    loss_array[i] += np.mean(loss)
    accuracy_array[i] += accuracy(predictions, train_label)
    validation_loss[i] += np.mean((validation_prediction - valid_label) * (validation_prediction - valid_label))
    validation_accuracy[i] += accuracy(validation_prediction, valid_label)

    # Now have to do gradient descent step. Find gradient with respect to weights and bias
    # Note, we are only tracking the validation accuracy and loss for plotting, we do not apply backpropogation to it
    
    if args.actfunction == 'linear':
        for row in range(1, len(predictions), 1):
            for column in range(0, 10, 1):
                if column == 9:  # For the bias 
                    gradient_tensor[row][column] += 2*(predictions[row] - train_label[row])
                else:  # For the weights
                    gradient_tensor[row][column] += 2*(predictions[row] - train_label[row]) * train_data[row][column]
        # Now, we average each column
        average_mse = np.mean(gradient_tensor, axis=0)
        for m in range(0, 9, 1):
            weights[m] = weights[m] - average_mse[m] * lr
        bias = bias - average_mse[9] * lr

    if args.actfunction == 'sigmoid':
        for row in range(1, len(predictions), 1):
            for column in range(0, 10, 1):
                if column == 9:  # For the bias
                    gradient_tensor[row][column] += 2*(predictions[row] - train_label[row]) * predictions[row] * (1 - predictions[row])
                else:  # For the weights
                    gradient_tensor[row][column] += 2*(predictions[row] - train_label[row]) * train_data[row][column] * predictions[row] * (1 - predictions[row])
        # Now, we average each column
        average_mse = np.mean(gradient_tensor, axis=0)
        for m in range(0, 9, 1):
            weights[m] = weights[m] - average_mse[m] * lr
        bias = bias - average_mse[9] * lr

    if args.actfunction == 'relu':
        for row in range(1, len(predictions), 1):
            for column in range(0, 10, 1):
                if column == 9:  # For the bias
                    if predictions[row] > 0:
                        gradient_tensor[row][column] += 2*(predictions[row] - train_label[row])
                    else: 
                        gradient_tensor[row][column] += 0
                else:  # For the weights
                    if predictions[row] > 0:
                        gradient_tensor[row][column] += 2*(predictions[row] - train_label[row]) * train_data[row][column]
                    else:
                        gradient_tensor[row][column] += 0
        # Now, we average each column
        average_mse = np.mean(gradient_tensor, axis=0)
        for m in range(0, 9, 1):
            weights[m] = weights[m] - average_mse[m] * lr
        bias = bias - average_mse[9] * lr

    predictions = np.zeros_like(train_label)
    validation_prediction = np.zeros_like(valid_label)
    gradient_tensor = np.zeros((predictions.shape[0], 10))

# So we now have our weights and bias that have been trained and tuned via back-propagation
# We can now plot loss vs epoch and accuracy vs epoch

# Need an epoch array:
epoch_array = [0]*epochs
for i in range(0, epochs, 1):
    epoch_array[i] += i

font = {'size': 6}

print("epoch: " + str(epochs) + " | learning rate: " + str(lr) + " | training acc: " + str(accuracy_array[epochs -1]) + " | validation acc: " + str(validation_accuracy[epochs - 1]))
print(weights)
print(bias)

plt.figure(1)  # Figure 1 is for training and validation loss + accuracy
plt.subplot(211)
plt.plot(epoch_array, loss_array, 'r', epoch_array, validation_loss, 'g')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('Training and Validation Loss by Epoch. learning rate: ' + str(lr) + ' activation function: ' + str(activation), fontdict= font)
plt.legend(['training loss', 'validation loss'])
plt.subplot(212)
plt.plot(epoch_array, accuracy_array, 'r', epoch_array, validation_accuracy, 'g')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Training and Validation Accuracy by Epoch. learning rate: ' + str(lr) + ' activation function: ' + str(activation), fontdict= font)
plt.legend(['training accuracy', 'validation accuracy'])
plt.tight_layout()
plt.show()

