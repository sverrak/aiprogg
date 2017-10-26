# from module3 import tflowtools as TFT
from module3.tutor3 import *
from module3 import mnist_basics as MB
import numpy as np
import time

PRINT_MODE = True


def scale_features(features, mode=1):
    # Max & min scaling
    if mode == 1:
        for c in range(len(features[0])):
            col_max = 0
            col_min = 9999999

            # Get the right min and max value for the column
            for f in features:
                if f[c] > col_max:
                    col_max = f[c]
                elif f[c] < col_min:
                    col_min = f[c]

            # Scale each feature value
            for f in features:
                f[c] = (f[c] - col_min) / (col_max - col_min)
    # Mean & stdev scaling
    else:
        for c in range(len(features[0])):
            col_my = sum([f[c] for f in features]) / len(features)
            col_sigma = np.std([f[c] for f in features])

            for f in features:
                f[c] = (f[c] - col_my) / col_sigma

    return features


# Reads data from files and partitions into training, validation and testing data
# case_fraction: fraction of data set to be used, TeF = testing fraction, VaF) = validation fraction
def load_data(file_name, normalize=0, case_fraction=1, delimiter=','):
    # Reads data set into numpy array
    is_one_hot = True  # Indicates if the input data labels are already one hot vectors
    labels = []
    features = []

    # Generate cases
    if file_name == 'mnist':
        is_one_hot = False
        cases = MB.load_all_flat_cases()
        features, labels = cases

    elif file_name == "parity":     # 10
        # size = 10
        size = int(input("\nParity size: ") or 10)    # TODO: FJERN alle disse - må kunne velge selv!
        cases = TFT.gen_all_parity_cases(size, True)

    elif file_name == "count":
        # a = 500
        # b = 15
        print("\n----> Bit counter:")
        a = int(input("First parameter:") or 500)
        b = int(input("Second parameter:") or 15)
        cases = TFT.gen_vector_count_cases(a, b)

    elif file_name == "onehot":
        # size = 8
        size = int(input("\nOne hot size: ") or 8)
        cases = TFT.gen_all_one_hot_cases(size)

    elif file_name == "auto":
        # x = ''
        print("\n----> Auto encoder:")
        x = input("Do you want to type all parameters: ")
        if x == "y":
            a = int(input("First parameter:"))
            b = int(input("Second parameter:"))
            c = int(input("Third parameter:"))
            d = int(input("Forth parameter:"))
        else:
            a, b, c, d = 500, 15, 0.4, 0.7
        cases = TFT.gen_dense_autoencoder_cases(a, b, (c, d))

    elif file_name == "segmented":
        # x = ''
        print("\n----> Segment counter:")
        x = input("Do you want to type all parameters: ")
        if x == "y":
            a = int(input("First parameter:"))
            b = int(input("Second parameter:"))
            c = int(input("Third parameter:"))
            d = int(input("Forth parameter:"))
            e = bool(input("Fifth parameter:"))
        else:
            a, b, c, d, e = 25, 1000, 0, 5, True
        cases = TFT.gen_segmented_vector_cases(a, b, c, d, e)

    else:   # if glass.txt or other dataset from UC Irvine ML repository
        is_one_hot = False
        cases = np.genfromtxt('./mnist/' + file_name + '.txt', delimiter=delimiter, dtype=float)
        np.random.shuffle(cases)
        features, labels = cases[:, :-1], cases[:, -1]

    # Separate features and labels
    if is_one_hot:
        np.random.shuffle(cases)
        features, labels = [case[0] for case in cases], [TFT.one_hot_to_int(case[1]) for case in cases]
    # else:
    #     if len(labels) == 0:    # if labels (and features) are not yet set, then:
    #         np.random.shuffle(cases)
    #         features, labels = [case[0] for case in cases], [case[1] for case in cases]
    #         print("##########33\n##########3333\n###############33333\n##############3333\n###########3")

    if normalize:
        features = scale_features(features, 2)

    # Separate features and labels and adapt to chosen case fraction
    separator = round(case_fraction * len(features))

    features = features[:separator]
    labels = labels[:separator]

    len_of_cases = len(labels)
    n_labels = int(max(labels) + (1 if 0 in labels else 0))
    # if file_name in ['glass', 'wine']:
    #     n_labels += 1

    new_labels = []
    if file_name == 'mnist':
        for l in labels:
            new_labels.append(TFT.int_to_one_hot(int(l), size=n_labels))
    else:
        for l in labels:
            new_labels.append(TFT.int_to_one_hot(int(l)-1, size=n_labels))
    cases = [[data, label] for data, label in zip(features, new_labels)]
    # print(cases[0])

    return cases, n_labels, len_of_cases

# ------------------------------------------


def gann_runner(dataset, normalize, lrate, hidden_layers, hidden_act_f, output_act_f, cost_f, case_fraction, vfrac, tfrac, init_weight_range, mbs, epochs, bestk, softmax, vint, optim):
    loaded = []
    if dataset in ['mnist', 'wine', 'glass', 'yeast', 'iris']:
        if dataset == 'mnist':
            loaded = load_data(dataset, case_fraction=case_fraction)
        elif dataset == 'wine':
            loaded = load_data(dataset, normalize=normalize, delimiter=';', case_fraction=1)
        elif dataset in ['glass', 'yeast', 'iris']:
            loaded = load_data(dataset, normalize=normalize, delimiter=',', case_fraction=1)
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
    ann = GANN(dims=dims, cman=cman, lrate=lrate, showint=None, mbs=mbs, vint=vint, softmax=softmax, hidden_act_f=hidden_act_f,
                output_act_f=output_act_f, init_w_range=init_weight_range, cost_f=cost_f, optim=optim)
    errors = ann.run(epochs=epochs, bestk=bestk)
    return (errors[0] / (round(len_of_cases * (1 - vfrac - tfrac))), errors[1] / (round(len_of_cases * tfrac))), ann, cman


def get_post_training_cases(training_cases):
    # Get information about the cases to be used while mapping
    print("Map Batch Size: 0,1,2,...., n or x:n or x (<--number of random cases), where n = number of lines in cases")
    cases_to_be_examined = input("Which cases would you like to use in postprocessing: ")
    post_training_cases = []

    if cases_to_be_examined == '0':
        return 0

    # Collect the cases depending on input format
    if ":" in cases_to_be_examined:
        first = int(cases_to_be_examined[0:cases_to_be_examined.index(":")])
        last = int(cases_to_be_examined[cases_to_be_examined.index(":")+1:])
        post_training_cases = training_cases[first: last]

    elif "," in cases_to_be_examined:
        indices = cases_to_be_examined.split(",")
        for i in indices:
            post_training_cases.append(training_cases[i])
    else:
        # Add the first x cases from training_cases (after shuffled) to post_training_cases
        np.random.shuffle(training_cases)
        post_training_cases = training_cases[:int(cases_to_be_examined)]
    return post_training_cases


def init_and_run(mapping):
    while True:
        print('\n############################\n')
        # mode = ''
        mode = input("Do you want to use saved parameters (enter '.' to quit)? ")
        start_time = time.time()
        
        # *** 0 Setup the network ***
        # Get input parameters
        if mode != "y":

            # Choose dataset
            print("Candidate datasets: 'mnist', 'wine', 'glass', 'yeast' ")
            dataset = input("Choose dataset: ")
            normalize = input("Choose if one wants to normalize the input data (the features): ")

            # Get input values
            lr = float(input("Learning rate: "))
            n_hidden_layers = int(input("Number of hidden layers: "))
            hidden_layers = []
            epochs = int(input("Steps (epochs): "))
            bestk = bool(input("bestk: "))
            optimizer = input("Type of optimizer (GD or Adam): ")
            for i in range(n_hidden_layers):
                print("\nParameters for hidden layer " + str(i) + ".")

                # Collecting the inputs
                hidden_layers.append(float(input("Layer size: ")))
            h_act_f = input("Hidden layer activation function (relu, softmax, sigmoid, tanh): ")
            output_act_f = input("Output layer activation function (relu, softmax, sigmoid, tanh): ")
            cost_function = input("Cost function (cross-entropy, MSE, ..): ")
            softmax = bool(input("Softmax? "))
            vint = int(input("Vint: "))

            case_fraction = float(input("Case fraction: "))
            vfrac = float(input("Validation fraction: "))
            tfrac = float(input("Test fraction: "))
            wr0 = float(input("Lower weight range: "))
            wr1 = float(input("Upper weight range: "))
            mbs = int(input("MBS: "))
            wrange = [wr0, wr1]
        elif mode == '.':
            break
        else:
            dataset = 'count'
            case_fraction = 0.05  # only for MNIST-dataset - the others are always 1
            epochs = 200
            lr = 0.07
            mbs = 50

            hidden_layers = [24, 12]
            normalize = True
            h_act_f = "relu"
            output_act_f = 'softmax'
            softmax = True
            cost_function = "MSE"
            optimizer = 'adam'
            bestk = 1

            vfrac, tfrac = 0.1, 0.1
            wr0, wr1 = -0.1, 0.1
            wrange = [wr0, wr1]
            vint = math.ceil(epochs / 5)

        # *** 1 Train the network ***
        print("\nComputing optimal weights....")
        result, ann, cman = gann_runner(dataset, normalize, lr, hidden_layers, h_act_f, output_act_f, cost_function, case_fraction, vfrac, tfrac,wrange, mbs, epochs, bestk, softmax, vint, optimizer)
        print("Done computing weights!\n")
        PLT.show()

        if mapping:
            while True:
                # *** 2 Declare grab vars ***
                do_mapping = input("Would you like to explore the variables further? ")
                if do_mapping == "y":
                    print("\n------> Entering mapping mode...\n")
                    grabbed_vars = []
                    new_var1 = " "
                    while new_var1 != "":
                        new_var1 = input("Which variables would you like to explore ('Enter '' to exit): ")
                        if new_var1 == "":
                            break
                        new_var2 = input("wgt/out/bias/in: ")

                        # Check that the input is not already being examined
                        if (new_var1, new_var2) in grabbed_vars:
                            print("New variable " + str(new_var1) + ", " + str(new_var2) + " is already in grabbed_vars")
                        else:
                            grabbed_vars.append((new_var1, new_var2))
                            ann.add_grabvar(int(new_var1), new_var2)
                    print("Done grabbing variables.\n")

                    # *** 3 Determine cases for post-training phase ***
                    # Get user input
                    cases_to_show = input("Examine training, validation or testing cases? ")

                    # Retrieve cases
                    if cases_to_show == "training":
                        cases = cman.get_training_cases()
                    elif cases_to_show == "validation":
                        cases = cman.get_validation_cases()
                    else:
                        cases = cman.get_testing_cases()

                    # This list will contain the cases to be used in post_training
                    post_training_cases = get_post_training_cases(cases)

                    if str(post_training_cases) != '0':
                        # *** 4-5 Run the network in mapping mode ***
                        # Any mapping operation will require a session
                        sess = ann.reopen_current_session()
                        ann.do_mapping(sess, post_training_cases, msg='Mapping', bestk=bestk)

                run_more = input("Do you want to run more epochs? If yes, how many? ('.' for exit) ")
                if run_more == '.' or run_more == '':
                    break
                ann.runmore(int(run_more), bestk=bestk)
                PLT.show()

        print('\nRun time:', time.time() - start_time, 's')
        PLT.close()


def test_input_combinations():
    start_time = time.time()
    with open('results_of_testing.txt', 'w') as file:
        inputs = ['dataset', 'h_act_f', 'output_act_f', 'hidden_layers', 'cost_function', 'epochs', 'lr', 'mbs']
        file.write('\t'.join(['train_score', 'test_score'] + inputs + ['\n']))

        case_fraction = 0.05
        h_act_f = ['relu']
        output_act_f = ['softmax']  # , 'sigmoid', 'tanh']
        cost_function = ['MSE']
        normalize = True

        dataset = ['count']
        hidden_layers = [[24,12], [32], [64], [128], [1024], [48, 32], [128,64], [256,128]]
        lr = [0.01, 0.05, 0.1]
        mbs = [5, 16, 50]
        epochs = [200]
        iterations = 1
        counter = 0
        N = len(dataset)*len(hidden_layers)*len(mbs)*len(epochs)*len(lr)*iterations
        for d in dataset:
            for h in h_act_f:
                for o in output_act_f:
                    for l in hidden_layers:
                        for c in cost_function:
                            for r in lr:
                                for m in mbs:
                                    for e in epochs:
                                        for _ in range(iterations):
                                            res = gann_runner(d, normalize, r, l, h, o, c, case_fraction, 0.1, 0.1,
                                                              [-0.1, 0.1], m, e, softmax=True, vint=None, bestk=1, optim='adam')[0]
                                            res = [round(r, 2) * 100 for r in res]
                                            file.write('\t'.join([str(i) for i in
                                                                  [res[0], res[1], d, h, o, l, c, e, r, m]] + ['\n']))
                                            counter += 1
                                            print('.....', counter, 'of', N, 'in total\n')
                                file.flush()
    print("Run time:", time.time() - start_time, 's')
    # PLT.show()


if __name__ == '__main__':
    PRINT_MODE = True
    init_and_run(mapping=True)
    # PRINT_MODE = False
    # test_input_combinations()
