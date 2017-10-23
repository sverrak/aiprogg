# from module3 import tflowtools as TFT
from module3.tutor3 import *
from module3 import mnist_basics as MB
import numpy as np
import time

PRINT_MODE = True


# Reads data from files and partitions into training, validation and testing data
# case_fraction: fraction of data set to be used, TeF = testing fraction, VaF) = validation fraction
def load_data(file_name, case_fraction=1, delimiter=','):
    # Reads data set into numpy array

    is_one_hot = True  # Indicates if the input data labels are already one hot vectors
    labels = []

    # Generate cases (191017)
    if file_name == 'mnist':
        is_one_hot = False
        cases = MB.load_all_flat_cases()
        features, labels = cases

    elif file_name == "parity":     # 10
        size = 10
        # size = int(input("Parity size: "))
        cases = TFT.gen_all_parity_cases(size, True)

    elif file_name == "count":
        a = 500
        b = 15
        # a = int(input("First parameter:"))      # 500
        # b = int(input("Second parameter:"))     # 15
        cases = TFT.gen_vector_count_cases(a, b)

    elif file_name == "onehot":
        size = 8
        # size = int(input("One hot size: "))     # 8
        cases = TFT.gen_all_one_hot_cases(size)

    elif file_name == "auto":
        # x = input("Do you want to type all parameters: ")
        x = ''
        if x == "yes":
            a = int(input("First parameter:"))
            b = int(input("Second parameter:"))
            c = int(input("Third parameter:"))
            d = int(input("Forth parameter:"))
        else:
            a, b, c, d = 500, 15, 0.4, 0.7
        cases = TFT.gen_dense_autoencoder_cases(a, b, (c, d))

    elif file_name == "segmented":
        # x = input("Do you want to type all parameters: ")
        x = ''
        if x == "yes":
            a = int(input("First parameter:"))
            b = int(input("Second parameter:"))
            c = int(input("Third parameter:"))
            d = int(input("Forth parameter:"))
            e = bool(input("Fifth parameter:"))
        else:
            a, b, c, d, e = 25, 1000, 0, 5, True
        cases = TFT.gen_segmented_vector_cases(a, b, c, d, e)

    else:
        is_one_hot = False
        cases = np.genfromtxt('./mnist/' + file_name, delimiter=delimiter, dtype=float)
        np.random.shuffle(cases)
        features, labels = cases[:, :-1], cases[:, -1]

    # Separate features and labels (191017)
    if is_one_hot:
        np.random.shuffle(cases)
        features, labels = [case[0] for case in cases], [TFT.one_hot_to_int(case[1]) for case in cases]

    else:
        if len(labels) == 0:    # if labels (and features) are not yet set, then:
            np.random.shuffle(cases)
            features, labels = [case[0] for case in cases], [case[1] for case in cases]

    # Separate & shuffle cases
    separator = round(case_fraction * len(features))

    features = features[:separator]
    labels = labels[:separator]
    # print(features)

    len_of_cases = len(labels)
    n_labels = max(number_of_labels(labels), 2)

    new_labels = []
    for l in labels:
        new_labels.append(TFT.int_to_one_hot(int(l)-1, size=n_labels))
    cases = [[data, label] for data, label in zip(features, new_labels)]
    # print()
    # print(labels)
    # print(new_labels)

    return cases, n_labels, len_of_cases


def number_of_labels(labels):
    return int(max(labels))


# ------------------------------------------


def gann_runner(dataset, lrate, hidden_layers, hidden_act_f, output_act_f, cost_f, case_fraction, vfrac,
                tfrac, init_weight_range, mbs, epochs, bestk, softmax, vint):
    loaded = []
    if dataset in ['mnist', 'wine', 'glass', 'yeast', 'iris']:
        if dataset == 'mnist':
            loaded = load_data(dataset, case_fraction)
        elif dataset == 'wine':
            loaded = load_data(dataset + '.txt', delimiter=';', case_fraction=1)
        elif dataset in ['glass', 'yeast', 'iris']:
            loaded = load_data(dataset + '.txt', delimiter=',', case_fraction=1)
        cases = (lambda: loaded[0])
        n_labels, len_of_cases = loaded[1], loaded[2]

    else:
        loaded = load_data(dataset, case_fraction=1)
        n_labels, len_of_cases = loaded[1], loaded[2]
        cases = (lambda: loaded[0])

    cman = CaseManager(cfunc=cases, vfrac=vfrac, tfrac=tfrac)
    dims = [len(cman.training_cases[0][0])] + hidden_layers + [n_labels]
    if PRINT_MODE:
        print('Length of data set:', len_of_cases)
        print('Number of labels:', n_labels)

    # Run ANN with all input functions
    ann = GANN(dims=dims, cman=cman, lrate=lrate, showint=None, mbs=mbs, vint=vint, softmax=softmax,
               hidden_act_f=hidden_act_f, output_act_f=output_act_f, init_w_range=init_weight_range, cost_f=cost_f)
    errors = ann.run(epochs=epochs, bestk=bestk)
    return errors[0] / (round(len_of_cases * (1 - vfrac - tfrac))), errors[1] / (round(len_of_cases * tfrac))


def get_input():
    # Until told to stop, the algorithm should run infinitely
    # while True:
    mode = ''
    while mode != 'no':
        mode = 'no'
        # mode = input("Do you want to type all parameters (enter '.' to quit): ")
        start_time = time.time()
        if mode == "yes":

            # Choose dataset
            print("Candidate datasets: 'mnist', 'wine', 'glass', 'yeast' ")
            dataset = input("Choose dataset: ")

            # Get input values
            # method = input("What method do you want to use (GD, ... ): ")
            lr = float(input("Learning rate: "))
            n_hidden_layers = int(input("Number of hidden layers: "))
            hidden_layers = []
            epochs = int(input("Epochs: "))
            bestk = bool(input("bestk: "))
            for i in range(n_hidden_layers):
                print("\nParameters for hidden layer " + str(i) + ".")

                # Collecting the inputs
                hidden_layers.append(float(input("Layer size: ")))
            h_act_f = input("Hidden layer activation function (relu, softmax, sigmoid, tanh):")
            output_act_f = input("Output layer activation function (relu, softmax, sigmoid, tanh):")
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
            dataset = 'parity'
            case_fraction = 0.03  # only for MNIST-dataset - the others are always 1
            epochs = 10
            lr = 0.1
            mbs = 1

            hidden_layers = []
            h_act_f = "relu"
            output_act_f = 'softmax'
            softmax = True  # TODO: hvorfor får vi 100% når softmax = False???
            cost_function = "MSE"   # TODO: implement other cost functions
            bestk = 1

            vfrac, tfrac = 0.1, 0.1
            wr0, wr1 = -0.1, 0.1
            wrange = [wr0, wr1]
            vint = math.ceil(epochs / 5)

        # Run the GANN
        print("\nComputing optimal weights....")
        gann_runner(dataset, lr, hidden_layers, h_act_f, output_act_f, cost_function, case_fraction, vfrac, tfrac,
                    wrange, mbs, epochs, bestk, softmax, vint)

        print("Done computing weights!")
        print('\nRun time:', time.time() - start_time, 's')


def test_input_combinations():
    start_time = time.time()
    with open('results_of_testing.txt', 'w') as file:
        inputs = ['dataset', 'h_act_f', 'output_act_f', 'hidden_layers', 'cost_function', 'epochs', 'lr', 'mbs']
        file.write('\t'.join(['train_score', 'test_score'] + inputs + ['\n']))
        case_fraction = 0.05
        dataset = ['count']
        h_act_f = ['relu']
        output_act_f = ['softmax']  # , 'sigmoid', 'tanh']
        hidden_layers = [[35, 20], [45, 20]] #, [40, 25], [40, 20], [45, 25], [45, 20]]   # [64, 32], [128, 64], [512, 256], [1024, 512], [128, 64, 32]]
        cost_function = ['MSE']  # TODO: implementere den andre
        epochs = [2000]
        lr = [0.1]
        mbs = [1]
        iterations = 2
        counter = 0
        for d in dataset:
            for h in h_act_f:
                for o in output_act_f:
                    for l in hidden_layers:
                        for c in cost_function:
                            for e in epochs:
                                for r in lr:
                                    for m in mbs:
                                        for _ in range(iterations):
                                            res = gann_runner(d, r, l, h, o, c, case_fraction, 0.1, 0.1, [-0.1, 0.1],
                                                              m, e, softmax=True, vint=None, bestk=1)
                                            res = [round(r, 2) * 100 for r in res]
                                            file.write('\t'.join([str(i) for i in
                                                                  [res[0], res[1], d, h, o, l, c, e, r, m]] + ['\n']))
                                            counter += 1
                                            print(counter)
    print("Run time:", time.time() - start_time, 's')


if __name__ == '__main__':
    # global PRINT_MODE
    # countex()
    # autoex()
    # print(TFT.gen_segmented_vector_cases(4, 50, 1, 4, True))
    # PRINT_MODE = True
    # get_input()
    PRINT_MODE = False
    test_input_combinations()
