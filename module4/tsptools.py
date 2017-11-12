import numpy as np
import random
from matplotlib import pyplot as plt
import math

# ------------------------------------------

# *** CLASSES ***


class SOM(object):
    def __init__(self, problem, learning_rate, decay_rate, printing_frequency, n_output_neurons=None):
        self.connection_weights = self.init_weights(len(self.input_neurons), len(self.output_neurons))
        self.input_neurons = self.init_input_neurons()
        self.output_neurons = self.init_output_neurons()
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.problem = problem
        self.priting_frequency = printing_frequency
        self.n_output_neurons = n_output_neurons

    def init_input_neurons(self):
        return problem.get_elements()

    def init_output_neurons(self, n_output_neurons):
        # Targeted data structures
        output_neurons = []
        
        
        
        # Distribute points over circle circumference 
        xs, ys = PointsInCircum(0.2, n=n_output_neurons)
        
        # Create output neurons
        for i in range(n_output_neurons):
            output_neurons.append(OutputNeuron(xs[i], ys[i]))

        # Set output neuron neighbors in OutputNeuron class
        for i, n in enumerate(output_neurons):
            if(i==0):
                n.set_neighbors([output_neurons[-1], output_neurons[1]])
            else:
                n.set_neighbors([output_neurons[i-1], output_neurons[i+1]])

        # Create neighborhood matrix
        self.neighbor_matrix = self.create_neighborhood_matrix(output_neurons)

        # Create lateral distance matrix
        self.lateral_distances = self.compute_lateral_distances(output_neurons)

        


    
    # *** WEIGHTS ***

    def init_weights(self, len_input, len_output):
        weights = [[random.uniform(0,1) for i in range(len_input)] for i in range(len_output)]
        return weights

    def update_weights(self, time_step):
        # Set iteration dependent variables
        lr = self.compute_learning_rate(time_step)
        weight_decay = self.compute_weight_decay(time_step)
        topologies = self.update_topologies(weight_decay)

        # Update the weights according to slide L16-10
        for i in range(len(self.input_neurons)):
            for j in range(len(self.output_neurons)):
                delta_w_ij = lr * topologies[i][j] * (self.input_neurons[i] - self.connection_weights[i][j])
                self.connection_weights[i][j] += delta_w_ij

    def update_topologies(self, time_step):
        n_output_neurons = self.n_output_neurons
        topology_matrix = [[0 for j in range(n_output_neurons)] for i in range(len(self.input_neurons))]
        sigma = compute_sigma_t(time_step)

        for i in range(len(topology_matrix)):
            for j in range(len(topology_matrix[i])):
                topology_matrix[i][j] = math.exp(- self.lateral_distances[i][j] ** 2 / (2 * sigma ** 2))


        return 0

    def compute_sigma_t(time_step):
        return self.sigma_0 * math.exp(- time_step / self.tau_sigma)
    # Assuming highest weight value decides which output neuron is winning
    def compute_winning_neurons(self):
        # Winners is a dictionary mapping input vector x to its winning neuron
        winners = {}
        

        for i,x in enumerate(self.input_neurons):
            # Something is wrong
            winners[x] = argmax(self.discriminants(x))
        return winners

    def compute_total_cost(self):
        return 0

    def argmax(list):
        argmax, maxvalue = 0, list[0]
        for i,val in enumerate(list):
            if val > maxvalue:
                argmax, maxvalue = i, val
        return argmax

    def create_neighborhood_matrix(self, output_neurons):
        n_output_neurons = self.n_output_neurons
        neighbor_matrix = [[0 for i in range(n_output_neurons)] for j in range(n_output_neurons)]
        # Depending on output neuron structure, create the lateral distance matrix
        if(self.problem.get_output_neuron_structure() == "2D_lattice"):
            # To do
            return neighbor_matrix

        elif(self.problem.get_output_neuron_structure() == "ring"):
            for i in range(len(neighbor_matrix)):
                try:
                    neighbor_matrix[i][i+1] = 1
                except:
                    # Do nothing
                    a = 0
                
                try:
                    neighbor_matrix[i][i-1] = 1
                except:
                    # Do nothing
                    a = 0

        return neighbor_matrix

    def compute_lateral_distances(self, output_neurons):
        n_output_neurons = self.n_output_neurons
        lateral_distances = [[0 for i in range(n_output_neurons)] for j in range(n_output_neurons)]

        # Depending on output neuron structure, create the lateral distance matrix
        if(self.problem.get_output_neuron_structure() == "2D_lattice"):
            # To do
            return lateral_distances


        elif(self.problem.get_output_neuron_structure() == "ring"):
            for i in range(n_output_neurons):
                for j in range(i, n_output_neurons):
                    # To do: describe logic: 
                    lateral_distances[i][j] = min(abs(i - j),abs(n_output_neurons - j + i), abs(n_output_neurons - i + j))
                    lateral_distances[j][i] = lateral_distances[i][j]
        
        return lateral_distances

    def manhattan_distance(x,y):
        return sum([abs(x[i] - y[i]) for i in range(len(x))])

    def discriminants(self, x):
        # This array is supposed to be equal to the d_j(x) of slide L16-8
        d_js = [0 for j in range(len(self.output_neurons))]
        D = len(self.input_neurons)

        # For each output neuron, compute the MSE between the input vector and each corresponding weight vector
        for j in range(len(self.output_neurons)):
            temp_sum = 0
            for i in range(D):
                temp_sum += (x[i] - self.connection_weights[i][j])

            d_js[j] = temp_sum

        return d_js

    def convergence_reached(self):
        return False

    def compute_input_output_distance(self):
        return []

    def run(self):
        self.init()

        while not self.convergence_reached():
            self.sampling()
            self.matching()
            self.updating()

# ------------------------------------------


class InputNeuron(object):
    pass


class OutputNeuron(object):
    def __init__(self, x, y):
        super(OutputNeuron, self).__init__()
        self.x = x
        self.y = y
        self.neighbors = []
        self.attached_input_vectors = []

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def attach_input_vector(self, input_vector):
        if(input_vector not in self.attached_input_vectors):
            self.attached_input_vectors.append(input_vector)

    def remove_input_vector(self, input_vector):
        if(input_vector in self.attached_input_vectors):
            self.attached_input_vectors.remove(input_vector)

# Sub-class for TSP-problems
class City(InputNeuron):
    def __init__(self, x, y):
        InputNeuron.__init__(self)
        self.x = x
        self.y = y

    # def set_closest_neuron(Neuron):
    #     self.Neuron = Neuron


# Sub-class for problems using images from MNIST
class Image(InputNeuron):
    def __init__(self, x, y):
        InputNeuron.__init__(self)
        # self.x = x    # TODO: Image skal muligens ikke ha x og y, men noe annet som input
        # self.y = y

# ------------------------------------------


class Problem(object):

    def get_elements(self):
        pass


class TSP(Problem):

    def __init__(self, file_name):
        Problem.__init__(self)
        self.data = file_reader(file_name)
        self.coordinates = scale_coordinates([[float(row[1]), float(row[2])] for row in self.data])
        self.cities = [City(city[0], city[1]) for city in self.coordinates]
        self.distances = []

    def compute_distances(self):
        distances = []
        for index1, city1 in enumerate(self.coordinates):
            distances.append([])
            for city2 in self.coordinates:
                dist = distance_between_cities(city1, city2)
                distances[index1].append(dist)
        return np.array(distances)

    def get_distances(self):
        self.distances = self.compute_distances()
        return self.distances

    def get_elements(self):
        return self.cities

    def plot_map(self):
        fig, ax = plt.subplots()

        ax.plot([c[0] for c in self.coordinates], [c[1] for c in self.coordinates], marker='*', c='gold',
                       markersize=15, linestyle='None')
        ax.set_xlim(0, 1.05)    # adjust figure axes to max x- and y-values, i.e. 0 and 1 (as they are normalized)
        ax.set_ylim(0, 1.05)    # use 1.05 to have some margin on the top and right side

        plt.pause(0.5)

        # TODO: iterate through the improving solution routes (dette er nå bare en test)
        for i, sol in enumerate([[[x[0]*0.9, x[1]] for x in self.coordinates], [[x[0]*0.95, x[1]*0.95] for x in self.coordinates]]):
            if i == 0:
                map, = ax.plot([c[0] for c in sol], [c[1] for c in sol], marker='o', markerfacecolor='None', c='green',
                       markersize=10, linestyle=':')
            else:
                map.set_data([c[0] for c in sol], [c[1] for c in sol])

            plt.pause(0.5)

        plt.show()

# ------------------------------------------

# *** GENERAL FUNCTIONS ***

def scale_coordinates(coordinates):
    for i in range(2):

        # Max & min scaling
        c_max = max([c[i] for c in coordinates])
        c_min = min([c[i] for c in coordinates])

        # Scale each feature value
        for c in range(len(coordinates)):
            coordinates[c][i] = (coordinates[c][i] - c_min) / (c_max - c_min)

    return coordinates


def distance_between_cities(i, j):
    return ((i[1] - j[1]) ** 2 + (i[0] - j[0]) ** 2) ** 0.5


def print_distances(distances):
    temp_string = ""
    for row in distances:
        temp_string += str(row[0]) + "\t" + row[1] + "\t" + row[2] + '\n'
    print(temp_string)


def file_reader(filename):
    with open(filename) as f:
        file = f.readlines()[5:]
        data = []
        for line in file:
            if line == 'EOF\n':
                break
            data.append(line.replace('\n', '').split())
    return data


def PointsInCircum(r, n=100):
        return [(math.cos(2*math.pi/n*x)*r,math.sin(2*math.pi/n*x)*r) for x in range(0, n+1)]


# ------------------------------------------

# ****  Parameters ****
printing_frequency = 25
L_RATE = 0.05
decay_rate = 0.05
RUN_MODE = "TSP"
FILE = 1

# ------------------------------------------

# ****  MAIN function ****

if __name__ == '__main__':

    if RUN_MODE == "TSP":
        # Instantiate TSP 
        test = TSP('./data/' + str(FILE) + '.txt')
    elif RUN_MODE == "MNIST":
        test = 0    # Todo

    # Create and run SOM
    # som = SOM(TSP_test, learning_rate, decay_rate, printing_frequency)
    # som.run()

    # Visualize solution
    # test.plot_map()
    # print_distances(TSP_test.data)

    for city in test.cities:
        print(city.x, city.y)

