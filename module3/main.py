# from module3 import tflowtools as TFT
from module3.tutor3 import *
from module3 import mnist_basics as MB
import numpy as np
import time


# Reads data from files and partitions into training, validation and testing data
# case_fraction: fraction of data set to be used, TeF = testing fraction, VaF) = validation fraction
def load_data(file_name, case_fraction, delimiter=','):
    # Reads data set into numpy array
    if file_name == 'mnist':
        cases = MB.load_all_flat_cases()
        features, labels = cases
    else:
        cases = np.genfromtxt('./mnist/' + file_name, delimiter=delimiter, dtype=float)
        features, labels = cases[:, :-1], cases[:, -1]
    separator = round(case_fraction * len(features))
    np.random.shuffle(features)
    np.random.shuffle(labels)
    features = features[:separator]
    labels = labels[:separator]
    len_of_cases = len(labels)
    n_labels = number_of_labels(labels)

    new_labels = []
    for l in labels:
        new_labels.append(TFT.int_to_one_hot(int(l)-1, size=n_labels))
    cases = [[data, label] for data, label in zip(features, new_labels)]
    return cases, n_labels, len_of_cases


def number_of_labels(labels):
    return int(max(labels))


# ------------------------------------------


def gann_runner(dataset, lrate, hidden_layers, hidden_act_f, output_act_f, cost_f, case_fraction, vfrac,
                tfrac, init_weight_range, mbs, epochs, bestk, softmax, vint):
    if dataset in ['mnist', 'wine', 'glass', 'yeast', 'iris']:
        loaded = []
        if dataset == 'mnist':
            loaded = load_data(dataset, case_fraction)
        elif dataset == 'wine':
            loaded = load_data(dataset + '.txt', delimiter=';', case_fraction=1)
        elif dataset in ['glass', 'yeast', 'iris']:
            loaded = load_data(dataset + '.txt', delimiter=',', case_fraction=1)
        cases = (lambda: loaded[0])
        n_labels, len_of_cases = loaded[1], loaded[2]
    else:
        len_of_cases = len(dataset)
        print(dataset)
        labels = [l[-1] for l in dataset]
        n_labels = number_of_labels(np.array(labels).flatten().tolist())
        cases = (lambda: dataset)

    cman = CaseManager(cfunc=cases, vfrac=vfrac, tfrac=tfrac)
    dims = [len(cman.training_cases[0][0])] + hidden_layers + [n_labels]
    print('Length of data set:', len_of_cases)
    print('Number of labels:', n_labels)

    # Run ANN with all input functions
    ann = GANN(dims=dims, cman=cman, lrate=lrate, showint=None, mbs=mbs, vint=vint, softmax=softmax,
               hidden_act_f=hidden_act_f, output_act_f=output_act_f, init_w_range=init_weight_range, cost_f=cost_f)
    errors = ann.run(epochs=epochs, bestk=bestk)
    return errors[0]/(round(len_of_cases*(1-vfrac-tfrac))), errors[1]/(round(len_of_cases*tfrac))


def get_input():
    # Until told to stop, the algorithm should run infinitely
    # while True:
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
            dataset = 'iris'
            functions = [TFT.gen_all_parity_cases(10), TFT.gen_all_one_hot_cases(500),
                         TFT.gen_dense_autoencoder_cases(500, 15, (0.4, 0.7)), TFT.gen_vector_count_cases(500, 15),
                         TFT.gen_segmented_vector_cases(4, 50, 1, 4, True), ]
            # TODO: Parity med (double=False) må ha bestk=None og n_labels -1,
            # TODO: autoencoder kjører ikke (kommer ikke på demoen) (ingen av dem)

            case_fraction = 0.03    # only for MNIST-dataset - the others are always 1
            cost_function = "MSE"
            epochs = 100
            lr = 0.05

            hidden_layers = [1024]
            h_act_f = "relu"      # TODO: evt prøve softplus i stedet for relu (ligner på)
            output_act_f = 'softmax'
                                # TODO: teste med ulike act.f. på de ulike hidden layers
            softmax = True      # TODO: hvorfor får vi 100% når softmax = False???
            bestk=1

            vfrac, tfrac = 0.1, 0.1
            wr0, wr1 = -0.1, 0.1
            wrange = [wr0, wr1]
            mbs = 10
            vint = math.ceil(epochs/10)

        # Run the GANN
        print("Computing optimal weights....")
        gann_runner(dataset, lr, hidden_layers, h_act_f, output_act_f, cost_function, case_fraction, vfrac, tfrac,
                    wrange, mbs, epochs, bestk, softmax, vint)

        print("Done computing weights!\n")
        print('\nRun time:', time.time() - start_time, 's')


def test_input_combinations(new_file):
    start_time = time.time()
    if new_file:    # deletes content of old file
        f = open('results_of_testing.txt', 'w')
        inputs = ['dataset', 'h_act_f', 'output_act_f', 'hidden_layers', 'cost_function', 'epochs', 'lr', 'mbs']
        f.write('\t'.join(['train_score', 'test_score'] + inputs + ['\n']))
        f.close()

    with open('results_of_testing.txt', 'a') as file:
        case_fraction = 0.05
        dataset = ['iris']
        h_act_f = ['relu'] #, 'sigmoid', 'tanh']
        output_act_f = ['softmax'] #, 'relu']  # , 'sigmoid', 'tanh']
        hidden_layers = [[], [32], [128], [512], [1024], [128, 64], [256, 128]]
        cost_function = ['MSE']     # TODO: implementere den andre
        epochs = [100]  # TODO: teste med mange flere epochs
        lr = [0.003, 0.01, 0.05, 0.1]
        mbs = [10, 50, 150]
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
    # countex()
    # autoex()
    get_input()
    # test_input_combinations(new_file=True)
