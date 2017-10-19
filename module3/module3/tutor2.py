import tensorflow as tf
import numpy as np
import matplotlib.pyplot as PLT
import module3.tflowtools as TFT


# **** Autoencoder ****
# We can extend the basic approach in tfex8 (tutor1.py) to handle a) a 3-layered neural network, and b) a collection
# of cases to be learned.  This is a specialized neural network designed to solve one type of classification
#  problem: converting an input string, through a single hidden layer, to a copy of itself on the output end.

class autoencoder():
    # nh = # hidden nodes (in the single hidden layer)
    # lr = learning rate

    def __init__(self, nh=3, lr=.1):
        self.cases = TFT.gen_all_one_hot_cases(2 ** nh)
        self.learning_rate = lr
        self.num_hiddens = nh
        self.build_neural_network(nh)

    def build_neural_network(self, nh):
        ios = 2 ** nh  # ios = input- and output-layer size
        self.w1 = tf.Variable(np.random.uniform(-.1, .1, size=(ios, nh)), name='Weights-1')  # first weight array
        self.w2 = tf.Variable(np.random.uniform(-.1, .1, size=(nh, ios)), name='Weights-2')  # second weight array
        self.b1 = tf.Variable(np.random.uniform(-.1, .1, size=nh), name='Bias-1')  # First bias vector
        self.b2 = tf.Variable(np.random.uniform(-.1, .1, size=ios), name='Bias-2')  # Second bias vector
        self.input = tf.placeholder(tf.float64, shape=(1, ios), name='Input')
        self.target = tf.placeholder(tf.float64, shape=(1, ios), name='Target')
        self.hidden = tf.sigmoid(tf.matmul(self.input, self.w1) + self.b1, name="Hiddens")
        self.output = tf.sigmoid(tf.matmul(self.hidden, self.w2) + self.b2, name="Outputs")
        self.error = tf.reduce_mean(tf.square(self.target - self.output), name='MSE')
        self.predictor = self.output  # Simple prediction runs will request the value of outputs
        # Defining the training operator
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error, name='Backprop')

    #  This is the same as quickrun in tutorial # 1, but now it's a method, not a function.

    def run_one_step(self, operators, grabbed_vars=None, dir='probeview',
                     session=None, feed_dict=None, step=1, show_interval=1):
        sess = session if session else TFT.gen_initialized_session(dir=dir)

        results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
        if show_interval and (step % show_interval == 0):
            TFT.show_results(results[1], grabbed_vars, dir)
        return results[0], results[1], sess

    def do_training(self, epochs=100, test_interval=10, show_interval=50):
        errors = []
        if test_interval: self.avg_vector_distances = []
        self.current_session = sess = TFT.gen_initialized_session()
        step = 0
        for i in range(epochs):
            error = 0
            grabvars = [self.error]
            for c in self.cases:
                feeder = {self.input: [c[0]], self.target: [c[1]]}
                _, grabvals, _ = self.run_one_step([self.trainer], grabvars, step=step, show_interval=show_interval,
                                                   session=sess, feed_dict=feeder)
                error += grabvals[0]
                step += 1
            errors.append(error)
            if (test_interval and i % test_interval == 0):
                self.avg_vector_distances.append(calc_avg_vect_dist(self.do_testing(sess, scatter=False)))
        PLT.figure()
        TFT.simple_plot(errors, xtitle="Epoch", ytitle="Error", title="")
        if test_interval:
            PLT.figure()
            TFT.simple_plot(self.avg_vector_distances, xtitle='Epoch',
                            ytitle='Avg Hidden-Node Vector Distance', title='')

    # This particular testing is ONLY called during training, so it always receives an open session.
    def do_testing(self, session=None, scatter=True):
        sess = session if session else self.current_session
        hidden_activations = []
        grabvars = [self.hidden]
        for c in self.cases:
            feeder = {self.input: [c[0]]}
            _, grabvals, _ = self.run_one_step([self.predictor], grabvars, session=sess,
                                               feed_dict=feeder, show_interval=None)
            hidden_activations.append(grabvals[0][0])
        if scatter:
            PLT.figure()
            vs = hidden_activations if self.num_hiddens > 3 else TFT.pca(hidden_activations, 2)
            TFT.simple_scatter_plot(hidden_activations, radius=8)
        return hidden_activations


# ********  Auxiliary functions for the autoencoder example *******

def vector_distance(vect1, vect2):
    return (sum([(v1 - v2) ** 2 for v1, v2 in zip(vect1, vect2)])) ** 0.5


def calc_avg_vect_dist(vectors):
    n = len(vectors);
    sum = 0
    for i in range(n):
        for j in range(i + 1, n):
            sum += vector_distance(vectors[i], vectors[j])
    return 2 * sum / (n * (n - 1))


#  A test of the autoencoder
def autoex1(epochs=2000, num_bits=3, lrate=0.5, tint=25, showint=100):
    ann = autoencoder(nh=num_bits, lr=lrate)
    PLT.ion()
    ann.do_training(epochs, test_interval=tint, show_interval=showint)
    ann.do_testing(scatter=True)  # Do a final round of testing to plot the hidden-layer activation vectors.
    PLT.ioff()
    TFT.close_session(ann.current_session)
    return ann
