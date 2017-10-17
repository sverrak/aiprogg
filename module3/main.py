# from module3 import tflowtools as TFT
from module3.tutor3 import *
from module3 import mnist_basics as MB
import numpy as np
import time


# Reads data from files and partitions into training, validation and testing data
# case_fraction: fraction of data set to be used, TeF = testing fraction, VaF) = validation fraction
def load_data(file_name, delimiter=',', case_fraction=1.0):
    # Reads data set into numpy array
    cases = np.genfromtxt('./mnist/' + file_name, delimiter=delimiter)
    print(len(cases))
    features, labels = cases[:,:-1], cases[:,-1]
    separator = round(case_fraction * len(features))
    np.random.shuffle(features)
    np.random.shuffle(labels)
    features = features[:separator]
    labels = labels[:separator]
    new_labels = []
    n_labels = number_of_labels(labels)
    for l in labels:
        new_labels.append(TFT.int_to_one_hot(int(l)-1, size=n_labels))
    cases = [[data, label] for data, label in zip(features, new_labels)]
    return cases, n_labels


def load_mnist(case_fraction):
    cases = MB.load_all_flat_cases()
    features, labels = cases
    separator = round(case_fraction * len(features))
    np.random.shuffle(features)
    np.random.shuffle(labels)
    features = features[:separator]
    labels = labels[:separator]

    new_labels = []
    n_labels = number_of_labels(labels)
    for l in range(len(labels)):
        new_labels.append(TFT.int_to_one_hot(int(labels[l]), size=n_labels))
    cases = [[data, label] for data, label in zip(features, new_labels)]
    return cases, n_labels


def number_of_labels(labels):
    # labels_new = []
    # for l in labels:
    #     if l not in labels_new:
    #         labels_new.append(l)
    # return len(labels_new)
    return int(max(labels))


# ------------------------------------------


def gann_runner(dataset, lrate, hidden_layers, hidden_act_f, output_act_f, cost_f, case_fraction, vfrac,
                tfrac, init_weight_range, mbs, epochs, bestk, softmax, vint):
    n_labels = 0
    if dataset == 'mnist':
        loaded = load_mnist(case_fraction)
        cases = (lambda: loaded[0])
        n_labels = loaded[1]

    elif dataset == 'wine':
        loaded = load_data(dataset + '.txt', delimiter=';', case_fraction=1)
        cases = (lambda: loaded[0])
        n_labels = loaded[1]

    elif dataset in ['glass', 'yeast']:
        loaded = load_data(dataset + '.txt', delimiter=',', case_fraction=1)
        cases = (lambda: loaded[0])
        n_labels = loaded[1]
    else:
        cases = (lambda: dataset)  # TODO: spør studass om vi tolker dette riktig

    cman = CaseManager(cfunc=cases, vfrac=vfrac, tfrac=tfrac)
    dims = [len(cman.training_cases[0][0])] + hidden_layers + [n_labels]  # TODO: endre til DYNamisk
    # print(dims)

    # Run ANN with all input functions
    ann = GANN(dims=dims, cman=cman, lrate=lrate, showint=None, mbs=mbs, vint=vint, softmax=softmax,
               hidden_act_f=hidden_act_f, output_act_f=output_act_f, init_w_range=init_weight_range, cost_f=cost_f)
    ann.run(epochs=epochs, bestk=bestk)


def get_input():
    # Until told, the algorithm should run infinitely
    # while True:  # TODO: uncomment
    mode = ''
    while mode != 'no':
        mode = 'no'
        # mode = input("Do you want to type all parameters (enter '.' to quit): ")  # TODO: uncomment
        start_time = time.time()
        if mode == "yes":

            # Choose dataset
            print("Candidate datasets: 'mnist', 'wine', 'glass', 'yeast' ")
            dataset = input("Choose dataset: ")

            # Get input values
            # method = input("What method do you want to use (GD, ... ): ")
            lr = float(input("Learning rate: "))
            n_hidden_layers = int(input("Hidden layers: "))
            hidden_layers = []
            epochs = int(input("Epochs: "))
            bestk = bool(input("bestk: "))
            activation_functions = []
            for i in range(n_hidden_layers):
                print("\nParameters for hidden layer " + str(i) + ".")

                # Collecting the inputs
                sizeX = float(input("Layer size: "))
                hidden_layers.append([sizeX])
            # activation_functions.append(input("Input layer activation function:" ))
            activation_functions.append(input("Hidden layer activation function (relu, softmax, sigmoid, tanh):"))
            activation_functions.append(input("Output layer activation function (relu, softmax, sigmoid, tanh):"))
            cost_function = input("Cost function (ce, mse, ..): ")
            softmax = bool(input("Softmax? "))
            vint = int(input("Vint: "))

            case_fraction = float(input("Case fraction: "))
            vfrac = float(input("Validation fraction: "))
            tfrac = float(input("Test fraction: "))
            wr0 = int(input("Lower weight range: "))
            wr1 = int(input("Upper weight range: "))
            mbs = int(input("MBS: "))
            wrange = [wr0, wr1]
        elif mode == '.':
            break
        else:
            dataset = "wine"
            case_fraction = 0.05
            cost_function = "MSE"
            epochs = 100
            bestk=1
            lr = 0.2

            hidden_layers = [150, 120]
            activation_functions = ["sigmoid", "softmax"]
            softmax = True      # TODO: hvorfor får vi 100% når softmax = False???

            vfrac, tfrac = 0.1, 0.15
            wr0, wr1 = -0.1, 0.1
            wrange = [wr0, wr1]
            mbs = 50
            vint = 50

        # Run the GANN
        print("Computing optimal weights....")

        gann_runner(dataset, lr, hidden_layers, activation_functions[0], activation_functions[1],
                    cost_function, case_fraction, vfrac, tfrac, wrange, mbs, epochs, bestk, softmax, vint)

        print("Done computing weights!\n")
        print('\nRun time:', time.time() - start_time, 's')


if __name__ == '__main__':
    # countex()
    # autoex()
    get_input()
