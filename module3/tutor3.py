import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as PLT
# PLT.use("Qt5Agg")
from module3 import tflowtools as TFT

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
                 hidden_act_f='relu', output_act_f=None, init_w_range=(-0.1, 0.1), cost_f='MSE', optim='GD'):
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
        self.saved_state_path = "netsaver/my_saved_session"

        # Added parameters to original assignment code
        self.hidden_act_f = hidden_act_f
        self.output_act_f = output_act_f
        self.init_w_range = init_w_range
        self.cost_f = cost_f    # can be mse, cross-entropy
        self.optim = optim

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

    def act_error(self):
        res = 0
        training = self.case_manager.get_training_cases()
        num_records = len(training)
        for i in range(num_records):
            print(training[i][1], tf.sess(self.output[i]))
            if self.output[i] == training[i][1][0]:
                res += 1
                print('#', res)
        return float(res) / float(num_records)

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
            insize = int(gmod.outsize)
        self.output = gmod.output  # Output of last module is output of whole network
        if self.softmax_outputs:
            self.output = tf.nn.softmax(self.output)
        self.target = tf.placeholder(tf.float64, shape=(None, gmod.outsize), name='Target')
        self.configure_learning(self.optim)

    # The optimizer knows to gather up all "trainable" variables in the function graph and compute
    # derivatives of the error function with respect to each component of each variable, i.e. each weight
    # of the weight array.
    def configure_learning(self, optim):
        if self.cost_f == 'MSE':
            self.error = tf.reduce_mean(tf.square(self.target - self.output), name='MSE')
        elif self.cost_f == 'cross-entropy':
            self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.output))

        self.predictor = self.output  # Simple prediction runs will request the value of output neurons

        # Defining the training operator
        if optim == 'GD':
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error, name='Backprop')

    # Added early termination functionality
    def do_training(self, sess, cases, epochs=100, continued=False):
        if not continued: self.error_history = []
        # is_early_termination = 'no'
        is_early_termination = input("Do you want to terminate early given convergence? ") == "y" # TODO: uncomment
        print()

        for i in range(epochs):
            if len(self.error_history) > 1 and is_early_termination == 'y' and \
                    convergence_test(self.error_history[-1][1], self.error_history[-2][1]):
                print("Terminated run after " + str(i) + " epochs.")
                break

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
                                  title="", fig=not (continued))  # TODO: uncomment

    # bestk = 1 when you're doing a classification task and the targets are one-hot vectors.  This will invoke the
    # gen_match_counter error function. Otherwise, when bestk=None, the standard MSE error function is used for testing.
    def do_testing(self, sess, cases, msg='Testing', bestk=None):
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        self.test_func = self.error
        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor, targets, k=bestk)
        testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=sess,
                                                 feed_dict=feeder, show_interval=None)
        # if msg == "Total Training":
        #     print('#####', testres/len(cases))

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
        new_labels = []
        for l in labels:
            new_labels.append(TFT.one_hot_to_int(l))
        correct = tf.nn.in_top_k(tf.cast(logits, tf.float32), new_labels, k)  # Return number of correct outputs
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def training_session(self, epochs, sess=None, dir="probeview", continued=False):
        self.roundup_probes()
        session = sess if sess else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.do_training(session, self.case_manager.get_training_cases(), epochs, continued=continued)

    def testing_session(self, sess, bestk=None):
        cases = self.case_manager.get_testing_cases()
        if len(cases) > 0:
            return self.do_testing(sess, cases, msg='Final Testing', bestk=bestk)

    def consider_validation_testing(self, epoch, sess):
        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.case_manager.get_validation_cases()
            if len(cases) > 0:
                error = self.do_testing(sess, cases, msg='Validation Testing')
                self.validation_history.append((epoch, error))

    # Do testing (i.e. calc error without learning) on the training set.
    def test_on_trains(self, sess, bestk=None):
        return self.do_testing(sess, self.case_manager.get_training_cases(), msg='Total Training', bestk=bestk)

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
            if is_bias(v):    # If v is a bias
                TFT.hinton_plot(np.array([v]), fig=self.grabvar_figures[fig_index], title=names[i] + ' at step ' + str(step))
                fig_index += 1
            else:
                TFT.hinton_plot(v, fig=self.grabvar_figures[fig_index], title=names[i] + ' at step ' + str(step))
                fig_index += 1
        PLT.show()

    def run(self, epochs=100, sess=None, continued=False, bestk=None):
        PLT.ion()
        self.training_session(epochs, sess=sess, continued=continued)
        final_training_error = self.test_on_trains(sess=self.current_session, bestk=bestk)
        final_testing_error = self.testing_session(sess=self.current_session, bestk=bestk)
        self.close_current_session(view=False)
        PLT.ioff()
        return final_training_error, final_testing_error

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
        self.saved_state_path = self.state_saver.save(session, spath, global_step=step)   # TODO: uncomment

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

    # Will behave similarly to method do_testing in tutor3.py, although it need not have self.error as its main operator,
    # since self.predictor would suffice. It will also need code for gathering and storing the grabbed values. 
    # Be aware that the resulting dimensions of the grabbed variables could vary depending upon whether you 
    # run all the cases through as a single mini-batch or whether you perform N calls to session.run, where N is the number of cases.
    def do_mapping(self, sess, cases, msg='Mapping', bestk=None):
        # Code from do_testing (which should resemble the code of do_mapping)
        features = [c[0] for c in cases]
        labels = [c[1] for c in cases]

        feeder = {self.input: features, self.target: labels}
        self.test_func = self.error

        if bestk is not None:
            self.test_func = self.gen_match_counter(self.predictor, labels, k=bestk)
        
        # *** New code (not in do_testing) ***
        show_hinton = input("Display Hinton plot? ")
        show_interval = None
        if show_hinton == "y":
            show_interval = 1

        # With show_interval = 1, the Hinton plot is displayed through run_one_step --> display_mapping --> hinton_plot
        # Hopefully, this is enough to display the Hinton plot
        testres, grabvals, _ = self.run_one_step(self.test_func, self.grabvars, self.probes, session=sess,
                                                 feed_dict=feeder, show_interval=show_interval)
        if bestk is None:
            print('%s Set Error = %f ' % (msg, testres))
        else:
            print('%s Set Correct Classifications = %f %%' % (msg, 100 * (testres / len(cases))))

        # Dendrogram
        show_dendro = input("Display dendrogram? ")
        if show_dendro == "y":

            print("Creating dendrogram...")
            # Dendrograms require string labels. Dendrogram labels depend on the feature data types of the current dataset
            if all_feature_values_binary(features):
                string_labels = []
                for f in features:
                    string_labels.append(TFT.bits_to_str(f))
            else:
                string_labels = [str(TFT.one_hot_to_int(label)) for label in labels]

            for i, val in enumerate(grabvals):

                if is_bias(val):
                    print("Cannot print dendrogram for a bias vector.")
                    continue

                # Filter for unique feature-label pairs before dendrogram plotting
                val, string_labels = get_unique_values(val, string_labels)

                # Call dendrogram function
                TFT.dendrogram(val, string_labels, metric='euclidean', mode='average', ax=None, title='Dendrogram',
                               orient='top', lrot=90.0)
            PLT.show()
            print("Done creating dendrogram(s).\n")

        show_matrix = input("Display matrices: ")
        if show_matrix == "y":
            print("Creating matrices...")
            for i, val in enumerate(grabvals):
                if is_bias(val):
                    TFT.display_matrix(np.array([val]))
                else:
                    TFT.display_matrix(val)
            PLT.show()
        print("Done creating matrices!")

        return testres  # self.error uses MSE, so this is a per-case value when bestk=None


# ------------------------------------------


# A General ANN-module = a layer of neurons (the output) plus its incoming weights and biases.
class GANN_Module:
    def __init__(self, ann, index, invariable, insize, outsize, act_f, init_w_range=(-0.1, 0.1)):
        self.ann = ann
        self.insize = insize     # Number of neurons feeding into this module
        self.outsize = int(outsize)   # Number of neurons in this module
        self.input = invariable  # Either the gann's input variable or the upstream module's output
        self.index = index
        self.name = "Module-" + str(self.index)
        self.build(act_f, init_w_range)

    def build(self, activation_f, init_w_range):
        model_name = self.name
        n = int(self.outsize)
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
    # print(cman.training_cases[0:3])
    ann = GANN(dims=[size, nbits, size], cman=cman, lrate=lrate, showint=showint, mbs=mbs, vint=vint, softmax=sm)
    ann.gen_probe(0,'wgt',('hist','avg'))  # Plot a histogram and avg of the incoming weights to module 0.
    ann.gen_probe(1,'out',('avg','max'))  # Plot average and max value of module 1's output vector
    ann.add_grabvar(0,'wgt') # Add a grabvar (to be displayed in its own matplotlib window).
    ann.run(epochs, bestk=bestk)
    ann.runmore(epochs * 2, bestk=bestk)
    return ann


def countex(epochs=1000, nbits=10, ncases=500, lrate=0.5, showint=500, mbs=20, vfrac=0.1, tfrac=0.1, vint=200, sm=True,
            bestk=1):
    case_generator = (lambda: TFT.gen_vector_count_cases(ncases, nbits))
    # print(case_generator())
    cman = CaseManager(cfunc=case_generator, vfrac=vfrac, tfrac=tfrac)
    ann = GANN(dims=[nbits, nbits * 3, nbits + 1], cman=cman, lrate=lrate, showint=showint, mbs=mbs, vint=vint,
               softmax=sm)
    ann.run(epochs, bestk=bestk)
    PLT.show()
    return ann


def is_bias(v):
    return len(v.shape) < 2


# Help method for new do_training
def convergence_test(a, b):
    return abs(a - b) < 0.0001  # Insert code if this is requested


def get_unique_values(v, l):
    tuples = []
    for row in list(zip(v,l)):
        tuples.append((tuple(row[0]), row[1]))

    unique_tuples = set(tuples)
    unique_f, unique_l = [], []
    for row in unique_tuples:
        unique_f.append(list(row[0]))
        unique_l.append(row[1])

    return unique_f, unique_l


def all_feature_values_binary(features):
    return all(all(str(fi) in "01" for fi in f) for f in features)
