# A little Tensorflow tutorial

import tensorflow as tf
import numpy as np
import module3.tflowtools as TFT


def tfex1(a, b):
    x = tf.constant(a)  # Scalar constant variables
    y = tf.constant(b)
    z = x * y  # As the product of variables, z is automatically declared as a multiplication OPERATOR
    sess = tf.Session()  # open a new session
    result = sess.run(z)  # Run the operator z and return the value of x * y
    sess.close()  # Explicitly close the session to release memory resources.  No danger if omitted.
    return result


def tfex2(a, b):
    x = tf.Variable(a, name='x')  # Now it's a variable with an initial value (a) and name
    y = tf.Variable(b, name='y')
    z = x * y  # Create operator z
    sess = tf.Session()
    sess.run(x.initializer)  # All variables at the leaves of a function graph must be initialized
    sess.run(y.initializer)  # or fed values from outside (via the "init_dict" given to "run")
    result = sess.run(z)
    sess.close()
    return result


def tfex2b(a, b):
    x = tf.Variable(a, name='x')  # Now it's a variable with an initial value (a) and name
    y = tf.Variable(b, name='y')
    z = x * y  # Create operator z
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())  # quick way to init all variables
    result = sess.run(z)
    sess.close()
    return result


# Consolidate this session-running into a simple function.  'operators' can be a single operator or a list of ops.

def quickrun(operators):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    result = sess.run(operators)  # result = a list of output values, one from each operator.
    sess.close()
    TFT.showvars(result)


# Now we'll work with matrices and vectors.
def tfex3():
    x = tf.Variable(np.random.uniform(0, 1, size=(3, 5)), name='x')  # Init as a matrix
    y = tf.Variable(np.random.uniform(10, 20, size=(5, 1)), name='y')  # Init as a column vector
    z = tf.matmul(x, y)  # z = matrix multiplication, yielding a (3,1) column vector
    return quickrun(z)


# This sends several operators to quickrun, each of which returns a value from sess.run
def tfex3b():
    w = tf.Variable(np.random.uniform(0, 1, size=(3, 5)), name='w')  # Init as a matrix
    x = tf.Variable(np.random.uniform(10, 20, size=(5, 1)), name='x')  # Init as a 5x1 column vector
    y = tf.Variable(np.random.uniform(100, 110, size=(3, 1)), name='y')  # Init as a 3x1 column vector
    z1 = tf.matmul(w, x)  # z1 = matrix multiplication, yielding a (3,1) column vector
    z2 = z1 + y
    return quickrun([w, z1, z2])


# Variable assignment as an explicit operation.
def tfex4():
    x = tf.Variable(np.random.uniform(1, 2, size=(5, 1)), name='x')
    x2 = tf.Variable(x.initialized_value())  # Only doing x2 = x would NOT do the copying.
    x2 = x2.assign(x + np.random.uniform(100, 200, size=(5, 1)))
    return quickrun([x, x2])


def quickrun2(operators, grabbed_vars=None, dir='probeview'):
    sess = tf.Session()
    probe_stream = TFT.viewprep(sess, dir=dir)
    sess.run(tf.global_variables_initializer())
    results = sess.run([operators, grabbed_vars])  # result = a list of output values, one from each operator.
    sess.close()
    TFT.show_results(results[1], grabbed_vars, dir)
    return results


def tfex5():
    w = tf.Variable(np.random.uniform(0, 1, size=(3, 5)), name='w')  # Init as a matrix
    x = tf.Variable(np.random.uniform(10, 20, size=(5, 1)), name='x')  # Init as a 5x1 column vector
    y = tf.Variable(np.random.uniform(100, 110, size=(3, 1)), name='y')  # Init as a 3x1 column vector
    z = tf.matmul(w, x) + y
    return quickrun2([z], [w, x, y])


def tfex5b():
    w = tf.Variable(np.random.uniform(0, 1, size=(3, 5)), name='w')  # Init as a matrix
    x = tf.Variable(np.random.uniform(10, 20, size=(5, 1)), name='x')  # Init as a 5x1 column vector
    y = tf.Variable(np.random.uniform(100, 110, size=(3, 1)), name='y')  # Init as a 3x1 column vector
    z = tf.add(tf.matmul(w, x), y, name="mult-and-add")
    return quickrun2([z], [w, x, y])


# Feeding the run with external inputs via a placeholder and feed_dict argument to session.run.  This comes in
# handy a little later, when we initialize some parameters once but then feed in values to placeholders
# many times during the course of a run.

def quickrun3(operators, grabbed_vars=None, dir='probeview', feed_dict=None):
    sess = tf.Session()
    probe_stream = TFT.viewprep(sess, dir=dir)
    sess.run(tf.global_variables_initializer())
    results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
    sess.close()
    TFT.show_results(results[1], grabbed_vars, dir)
    return results


def tfex6():
    x = tf.placeholder(tf.float64, shape=(5, 1), name='x')  # shape = None => accepts any shaped tensor
    y = tf.placeholder(tf.float64, shape=(3, 1), name='y')
    w = tf.Variable(np.random.uniform(0, 1, size=(3, 5)), name='w')  # Same matrix as before
    z = tf.matmul(w, x) + y
    # feed dictionary specifies the input values for the placeholders (x and y) to be used during session.run
    feeder = {x: np.random.uniform(10, 20, size=(5, 1)), y: np.random.uniform(100, 110, size=(3, 1))}
    # Now, only w gets initialized via its initializer; x and y get set by the feed dictionary.
    return quickrun3([z], [w, x, y], feed_dict=feeder)


# Enhancing quickrun to handle a pre-opened session as an argument and to return the session.

def quickrun4(operators, grabbed_vars=None, dir='probeview',
              session=None, feed_dict=None, step=1, show_interval=1):
    sess = session if session else TFT.gen_initialized_session(dir=dir)

    results = sess.run([operators, grabbed_vars], feed_dict=feed_dict)
    if (show_interval and step % show_interval) == 0:
        TFT.show_results(results[1], grabbed_vars, dir)
    return results[0], results[1], sess


# Use quickrun4 to call session.run several times in a loop, each time updating var x, while var y, a placeholder
# is set externally (via feed_dict) but to the same value on each call to session.run.

def tfex7(n=5):
    w = tf.Variable(np.random.uniform(-1, 1, size=(5, 5)), name='w')  # Same matrix as before
    x = tf.Variable(np.zeros((1, 5)), name='x')
    y = tf.placeholder(tf.float64, shape=(1, 5), name='y')
    feeder = {y: np.random.uniform(-1, 1, size=(1, 5))}
    update_x = x.assign(tf.matmul(x, w) + y)
    _, _, sess = quickrun4([update_x], [w, x, y], feed_dict=feeder)
    for step in range(n):
        quickrun4([update_x], [x], session=sess, feed_dict=feeder)
    TFT.close_session(sess)


# Now we have the tools to run a gradient-descent optimizer, the heart of supervised neural networks.

def tfex8(size=5, steps=50, tvect=None, learning_rate=0.5, showint=10):
    target = tvect if tvect else np.ones((1, size))
    w = tf.Variable(np.random.uniform(-.1, .1, size=(size, size)), name='weights')  # weights applied to x.
    b = tf.Variable(np.zeros((1, size)), name='bias')  # bias terms
    x = tf.placeholder(tf.float64, shape=(1, size), name='input')
    y = tf.sigmoid(tf.matmul(x, w) + b,
                   name='out-sigmoid')  # Gather all weighted inputs, then apply activation function
    error = tf.reduce_mean(tf.square(target - y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_operator = optimizer.minimize(error)
    feeder = {x: np.random.uniform(-1, 1, size=(1, size))}
    sess = TFT.gen_initialized_session()
    for step in range(steps):
        quickrun4([training_operator], [w, b, y], session=sess, feed_dict=feeder, step=step, show_interval=showint)
    TFT.close_session(sess)
