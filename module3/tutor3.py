import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
from module3 import tflowtools as TFT
from module3 import mnist_basics as MB
import time

# remove irritating warnings
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------


# ******* A General Artificial Neural Network ********
# This is the original GANN, which has been improved in the file gann.py
class GANN:
    def __init__(self, dims, cman, lrate=.1, showint=None, mbs=10, vint=None, softmax=False,
                 hidden_act_f='relu', output_act_f=None, init_w_range=(-0.1, 0.1), cost_f='MSE'):
        self.learning_rate = lrate
        self.layer_sizes = dims         # Sizes of each layer of neurons
        self.show_interval = showint    # Frequency of showing grabbed variables
        self.global_training_step = 0   # Enables coherent data-storage during extra training runs (see runmore).
        self.grabvars = []              # Variables to be monitored (by GANN code) during a run.
        self.grabvar_figures = []       # One matplotlib figure for each grabvar
        self.minibatch_size = mbs
        self.validation_interval = vint
        self.validation_history = []
        self.case_manager = cman
        self.softmax_outputs = softmax
        self.modules = []

        # Added parameters to original assignment code
        self.hidden_act_f = hidden_act_f
        self.output_act_f = output_act_f
        self.init_w_range = init_w_range
        self.cost_f = cost_f    # can be mse, cross-entropy

        self.build()


    # Probed variables are to be displayed in the Tensorboard.
    def gen_probe(self, module_index, type, spec):
        self.modules[module_index].gen_probe(type, spec)

    # Grabvars are displayed by my own code, so I have more control over the display format.  Each
    # grabvar gets its own matplotlib figure in which to display its value.
    def add_grabvar(self, module_index, type='wgt'):
        self.grabvars.append(self.modules[module_index].getvar(type))
        self.grabvar_figures.append(PLT.figure())

    def roundup_probes(self):
        self.probes = tf.summary.merge_all()

    def add_module(self, module):
        self.modules.append(module)

    def build(self):
        tf.reset_default_graph()  # This is essential for doing multiple runs!!
        num_inputs = self.layer_sizes[0]
        self.input = tf.placeholder(tf.float64, shape=(None, num_inputs), name='Input')
        invar = self.input
        insize = num_inputs

        # Build all of the modules
        for i, outsize in enumerate(self.layer_sizes[1:]):
            if i == len(self.layer_sizes[1:]):
                gmod = GANN_Module(self, i, invar, insize, outsize, self.output_act_f, self.init_w_range)
            else:
                gmod = GANN_Module(self, i, invar, insize, outsize, self.hidden_act_f, self.init_w_range)
            invar = gmod.output
            insize = gmod.outsize
        self.output = gmod.output  # Output of last module is output of whole network
        if self.softmax_outputs:
            self.output = tf.nn.softmax(self.output)
        self.target = tf.placeholder(tf.float64, shape=(None, gmod.outsize), name='Target')
        self.configure_learning()

    # The optimizer knows to gather up all "trainable" variables in the function graph and compute
    # derivatives of the error function with respect to each component of each variable, i.e. each weight
    # of the weight array.
    def configure_learning(self):
        if self.cost_f == 'MSE':
            self.error = tf.reduce_mean(tf.square(self.target - self.output), name='MSE')
        elif self.cost_f == 'cross-entropy':
            self.error = tf.reduce_mean(self.target * tf.log(self.output), name='cross-entropy')    # TODO: fix!

        self.predictor = self.output  # Simple prediction runs will request the value of output neurons
        # Defining the training operator
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error, name='Backprop')

    def do_training(self, sess, cases, epochs=100, continued=False):
        if not continued: self.error_history = []
        for i in range(epochs):
            error = 0
            step = self.global_training_step + i
            gvars = [self.error] + self.grabvars
            mbs = self.minibatch_size
            ncases = len(cases)
            nmb = math.ceil(ncases / mbs)
            for cstart in range(0, ncases, mbs):  # Loop through cases, one minibatch at a time.
                cend = min(ncases, cstart + mbs)
                minibatch = cases[cstart:cend]
                inputs = [c[0] for c in minibatch]
                targets = [c[1] for c in minibatch]
                feeder = {self.input: inputs, self.target: targets}
                _, grabvals, _ = self.run_one_step([self.trainer], gvars, self.probes, session=sess,
                                                   feed_dict=feeder, step=step, show_interval=self.show_interval)
                error += grabvals[0]
            self.error_history.append((step, error / nmb))
            self.consider_validation_testing(step, sess)
        self.global_training_step += epochs
        TFT.plot_training_history(self.error_history, self.validation_history, xtitle="Epoch", ytitle="Error",
                                  title="", fig=not (continued))

    # bestk = 1 when you're doing a classification task and the targets are one-hot vectors.  This will invoke the
    # gen_match_counter error function. Otherwise, when bestk=None, the standard MSE error function is used for testing.
    def do_testing(self, sess, cases, msg='Testing', bestk=None):
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.error
        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor, [TFT.one_hot_to_int(list(v)) for v in targets],
                                                    k=bestk)
        testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=sess,
                                                 feed_dict=feeder, show_interval=None)
        if bestk is None:
            print('%s Set Error = %f ' % (msg, testres))
        else:
            print('%s Set Correct Classifications = %f %%' % (msg, 100 * (testres / len(cases))))
        return testres  # self.error uses MSE, so this is a per-case value when bestk=None

    # Logits = tensor, float - [batch_size, NUM_CLASSES].
    # labels: Labels tensor, int32 - [batch_size], with values in range [0, NUM_CLASSES).
    # in_top_k checks whether correct val is in the top k logit outputs.  It returns a vector of shape [batch_size]
    # This returns a OPERATION object that still needs to be RUN to get a count.
    # tf.nn.top_k differs from tf.nn.in_top_k in the way they handle ties.  The former takes the lowest index, while
    # the latter includes them ALL in the "top_k", even if that means having more than k "winners".  This causes
    # problems when ALL outputs are the same value, such as 0, since in_top_k would then signal a match for any
    # target.  Unfortunately, top_k requires a different set of arguments...and is harder to use.

    def gen_match_counter(self, logits, labels, k=1):
        correct = tf.nn.in_top_k(tf.cast(logits, tf.float32), labels, k)  # Return number of correct outputs
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def training_session(self, epochs, sess=None, dir="probeview", continued=False):
        self.roundup_probes()
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.do_training(session, self.case_manager.get_training_cases(), epochs, continued=continued)

    def testing_session(self, sess, bestk=None):
        cases = self.case_manager.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(sess, cases, msg='Final Testing', bestk=bestk)

    def consider_validation_testing(self, epoch, sess):
        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.case_manager.get_validation_cases()
            if len(cases) > 0:
                error = self.do_testing(sess, cases, msg='Validation Testing')
                self.validation_history.append((epoch, error))

    # Do testing (i.e. calc error without learning) on the training set.
    def test_on_trains(self, sess, bestk=None):
        self.do_testing(sess, self.case_manager.get_training_cases(), msg='Total Training', bestk=bestk)

    # Similar to the "quickrun" functions used earlier.
    def run_one_step(self, operators, grabbed_vars=None, probed_vars=None, dir='probeview',
                     session=None, feed_dict=None, step=1, show_interval=1):
        sess = session if session else TFT.gen_initialized_session(dir=dir)
        if probed_vars is not None:
            results = sess.run([operators, grabbed_vars, probed_vars], feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        if show_interval and (step % show_interval == 0):
            self.display_grabvars(results[1], grabbed_vars, step=step)
        return results[0], results[1], sess

    def display_grabvars(self, grabbed_vals, grabbed_vars, step=1):
        names = [x.name for x in grabbed_vars]
        msg = "Grabbed Variables at Step " + str(step)
        print("\n" + msg, end="\n")
        fig_index = 0
        for i, v in enumerate(grabbed_vals):
            if names: print("   " + names[i] + " = ", end="\n")
            if type(v) == np.ndarray and len(v.shape) > 1:  # If v is a matrix, use hinton plotting
                TFT.hinton_plot(v, fig=self.grabvar_figures[fig_index], title=names[i] + ' at step ' + str(step))
                fig_index += 1
            else:
                print(v, end="\n\n")

    def run(self, epochs=100, sess=None, continued=False, bestk=None):
        PLT.ion()
        self.training_session(epochs, sess=sess, continued=continued)
        self.test_on_trains(sess=self.current_session, bestk=bestk)
        self.testing_session(sess=self.current_session, bestk=bestk)
        self.close_current_session(view=False)
        PLT.ioff()

    # After a run is complete, runmore allows us to do additional training on the network, picking up where we
    # left off after the last call to run (or runmore).  Use of the "continued" parameter (along with
    # global_training_step) allows easy updating of the error graph to account for the additional run(s).

    def runmore(self, epochs=100, bestk=None):
        self.reopen_current_session()
        self.run(epochs, sess=self.current_session, continued=True, bestk=bestk)

    #   ******* Saving GANN Parameters (weights and biases) *******************
    # This is useful when you want to use "runmore" to do additional training on a network.
    # spath should have at least one directory (e.g. netsaver), which you will need to create ahead of time.
    # This is also useful for situations where you want to first train the network, then save its parameters
    # (i.e. weights and biases), and then run the trained network on a set of test cases where you may choose to
    # monitor the network's activity (via grabvars, probes, etc) in a different way than you monitored during training.
    def save_session_params(self, spath='netsaver/my_saved_session', sess=None, step=0):
        session = sess if sess else self.current_session
        state_vars = []
        for m in self.modules:
            vars = [m.getvar('wgt'), m.getvar('bias')]
            state_vars = state_vars + vars
        self.state_saver = tf.train.Saver(state_vars)
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)

    def reopen_current_session(self):
        self.current_session = TFT.copy_session(self.current_session)  # Open a new session with same tensorboard stuff
        self.current_session.run(tf.global_variables_initializer())
        self.restore_session_params()  # Reload old weights and biases to continued from where we last left off

    def restore_session_params(self, path=None, sess=None):
        spath = path if path else self.saved_state_path
        session = sess if sess else self.current_session
        self.state_saver.restore(session, spath)

    def close_current_session(self, view=True):
        self.save_session_params(sess=self.current_session)
        TFT.close_session(self.current_session, view=view)

# ------------------------------------------


# A General ANN-module = a layer of neurons (the output) plus its incoming weights and biases.
class GANN_Module:
    def __init__(self, ann, index, invariable, insize, outsize, act_f, init_w_range=(-0.1, 0.1)):
        self.ann = ann
        self.insize = insize     # Number of neurons feeding into this module
        self.outsize = outsize   # Number of neurons in this module
        self.input = invariable  # Either the gann's input variable or the upstream module's output
        self.index = index
        self.name = "Module-" + str(self.index)
        self.build(act_f, init_w_range)

    def build(self, activation_f, init_w_range):
        model_name = self.name
        n = self.outsize

        self.weights = tf.Variable(np.random.uniform(init_w_range[0], init_w_range[1], size=(self.insize, n)),
                                   name=model_name + '-wgt', trainable=True)  # True = default for trainable anyway
        self.weights = tf.Variable(np.random.uniform(init_w_range[0], init_w_range[1], size=(self.insize, n)),
            name=model_name + '-wgt', trainable=True)  # True = default for trainable anyway
        self.biases = tf.Variable(np.random.uniform(init_w_range[0], init_w_range[1], size=n),
                                  name=model_name + '-bias', trainable=True)  # First bias vector

        if activation_f == 'relu':
            self.output = tf.nn.relu(tf.matmul(self.input, self.weights) + self.biases, name=model_name + '-out')
        elif activation_f == 'sigmoid':
            self.output = tf.nn.sigmoid(tf.matmul(self.input, self.weights) + self.biases, name=model_name + '-out')
        elif activation_f == 'tanh':
            self.output = tf.nn.tanh(tf.matmul(self.input, self.weights) + self.biases, name=model_name + '-out')

        self.ann.add_module(self)

    def getvar(self, type):  # type = (in,out,wgt,bias)
        return {'in': self.input, 'out': self.output, 'wgt': self.weights, 'bias': self.biases}[type]

    # spec, a list, can contain one or more of (avg,max,min,hist); type = (in, out, wgt, bias)
    def gen_probe(self, type, spec):
        var = self.getvar(type)
        base = self.name + '_' + type
        with tf.name_scope('probe_'):
            if ('avg' in spec) or ('stdev' in spec):
                avg = tf.reduce_mean(var)
            if 'avg' in spec:
                tf.summary.scalar(base + '/avg/', avg)
            if 'max' in spec:
                tf.summary.scalar(base + '/max/', tf.reduce_max(var))
            if 'min' in spec:
                tf.summary.scalar(base + '/min/', tf.reduce_min(var))
            if 'hist' in spec:
                tf.summary.histogram(base + '/hist/', var)

# ------------------------------------------


# *********** CASE MANAGER ********
# This is a simple class for organizing the cases (training, validation and test) for a
# a machine-learning system
class CaseManager:
    def __init__(self, cfunc, vfrac=0.0, tfrac=0.0):
        self.casefunc = cfunc
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        self.cases = np.array(self.casefunc())  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        ca = np.array(self.cases)
        np.random.shuffle(ca)  # Randomly shuffle all cases
        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases) * self.validation_fraction)
        self.training_cases = ca[0:separator1]
        self.validation_cases = ca[separator1:separator2]
        self.testing_cases = ca[separator2:]

    def get_training_cases(self): return self.training_cases

    def get_validation_cases(self): return self.validation_cases

    def get_testing_cases(self): return self.testing_cases

# ------------------------------------------


# ****  MAIN functions ****

# After running this, open a Tensorboard (Go to localhost:6006 in your Chrome Browser) and check the
# 'scalar', 'distribution' and 'histogram' menu options to view the probed variables.
def autoex(epochs=300, nbits=4, lrate=0.03, showint=100, mbs=None, vfrac=0.1, tfrac=0.1, vint=100, sm=False,
           bestk=None):
    size = 2 ** nbits
    mbs = mbs if mbs else size
    case_generator = (lambda: TFT.gen_all_one_hot_cases(2 ** nbits))
    cman = CaseManager(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac)
    ann = GANN(dims=[size, nbits, size], cman=cman, lrate=lrate, showint=showint, mbs=mbs, vint=vint, softmax=sm)
    ann.gen_probe(0,'wgt',('hist','avg'))  # Plot a histogram and avg of the incoming weights to module 0.
    ann.gen_probe(1,'out',('avg','max'))  # Plot average and max value of module 1's output vector
    ann.add_grabvar(0,'wgt') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.run(epochs, bestk=bestk)
    ann.runmore(epochs * 2, bestk=bestk)
    return ann


def countex(epochs=5000, nbits=10, ncases=500, lrate=0.5, showint=500, mbs=20, vfrac=0.1, tfrac=0.1, vint=200, sm=True,
            bestk=1):
    case_generator = (lambda: TFT.gen_vector_count_cases(ncases, nbits))
    # print(case_generator())
    cman = CaseManager(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac)
    ann = GANN(dims=[nbits, nbits * 3, nbits + 1], cman=cman, lrate=lrate, showint=showint, mbs=mbs, vint=vint,
               softmax=sm)
    ann.run(epochs, bestk=bestk)
    return ann


# Reads data from files and partitions into training, validation and testing data
# case_fraction: fraction of data set to be used, TeF = testing fraction, VaF) = validation fraction
def load_data(file_name, delimiter=',', case_fraction=1.0):
    # Reads data set into numpy array
    cases = np.genfromtxt('./mnist/' + file_name, delimiter=delimiter)
    np.random.shuffle(cases)     # Randomly shuffle all cases
    separator = round(len(cases) * case_fraction)
    cases = [[data, [int(label)]] for data, label in zip(cases[:, :-1].tolist(), cases[:, -1])]
    return cases[:separator]


def load_mnist(case_fraction):
    cases = MB.load_all_flat_cases()
    features, labels = cases
    separator = round(case_fraction*len(features))
    features = features[:separator]
    labels = labels[:separator]
    cases = [[data, [int(label)]] for data, label in zip(features, labels)]
    return cases

# ------------------------------------------


def gann_runner(dataset, method, lrate, hidden_layers, hidden_act_f, output_act_f, cost_f, case_fraction, vfrac,
                tfrac, init_weight_range, mbs):

    if dataset == 'mnist':
        cases = (lambda: load_mnist(case_fraction))

    elif dataset == 'yeast':
        cases = (lambda: load_data('yeast.txt', delimiter=';', case_fraction=case_fraction))

    elif dataset in ['glass', 'wine']:
        cases = (lambda: load_data(dataset + '.txt', delimiter=',', case_fraction=case_fraction))
    else:
        cases = (lambda: dataset)  # TODO: spør studass om vi tolker dette riktig

    cman = CaseManager(cfunc=cases, vfrac=vfrac, tfrac=tfrac)
    dims = [len(cman.training_cases[0][0])] + hidden_layers + [1]  # TODO: hva hvis output size != 1?
    print(dims)

    # Run ANN with all input functions
    ann = GANN(dims=dims, cman=cman, lrate=lrate, showint=None, mbs=mbs, vint=None, softmax=False,
                 hidden_act_f=hidden_act_f, output_act_f=output_act_f, init_w_range=init_weight_range, cost_f=cost_f)
    ann.run()


def get_input():
    # Until told, the algorithm should run infinitely
    while True:
        mode = input("Do you want to type all parameters (enter '.' to quit): ")
        if mode == "yes":

            # Choose dataset
            print("Candidate datasets: 'mnist', 'wine', 'glass', 'yeast' ")
            dataset = input("Choose dataset: ")

            # Get input values
            method = 0
            method = input("What method do you want to use (GD, ... ): ")
            lr = float(input("Learning rate: "))
            n_hidden_layers = int(input("Hidden layers: "))
            hidden_layers = []
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
            dataset = "mnist"

            # Get input values
            method = 'GD'
            lr = 0.05
            hidden_layers = [9, 10]
            activation_functions = ["relu", "softmax"]
            cost_function = "MSE"

            case_fraction = 0.1
            vfrac = 0.1
            tfrac = 0.1
            wr0 = -0.1
            wr1 = 0.1
            mbs = 10
            wrange = [wr0, wr1]

            epochs = 300
            nbits = 4
            showint = 100
            vint = 100
            sm = False

        # Run the GANN
        print("Computing optimal weights....")

        gann_runner(dataset, method, lr, hidden_layers, activation_functions[0], activation_functions[1],
                    cost_function, case_fraction, vfrac, tfrac, wrange, mbs)

        print("Done computing weights!\n")


if __name__ == '__main__':
    start_time = time.time()
    # autoex()
    # countex()

    get_input()

    # cases = (lambda: load_data('glass.txt', delimiter=',', case_fraction=1))
    # # cases = (lambda: MB.load_all_flat_cases())
    # print(MB.reconstruct_flat_cases(cases()))
    # # print(np.array(cases()))
    # cman = CaseManager(cfunc=cases, vfrac=0.1, tfrac=0.1)
    #
    # ann = GANN(dims=[9, 3, 3, 1], cman=cman)
    # ann.run()

    print('\nRun time:', time.time() - start_time, 's')
