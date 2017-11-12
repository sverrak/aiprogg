import numpy as np
import random
from matplotlib import pyplot as plt

class SOM(object):
    """docstring for SOM"""
    def __init__(self, learning_rate, decay_rate, printing_frequency, input_neurons, output_neurons, ):
        super(SOM, self).__init__()
        self.arg = arg
        self.connection_weights = self.init_weights(len(self.input_neurons), len(self.output_neurons))
        self.input_neurons = self.init_input_neurons()
        self.output_neurons = self.init_output_neurons()
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate

    def init_input_neurons():
        return 0        

    def init_output_neurons():
        # To do: Create a circle for TSP 
        return 0

    ### WEIGHTS
    def init_weights(self, len_input, len_output):
        weights = [[random.uniform(0,1) for i in range(len_input)] for i in range(len_output)]
        return weights

    def update_weights():
        return 0
    
    def update_topologies():
        return 0

    def compute_winning_neurons():
        return 0

    def compute_total_cost():
        return 0

    def discriminant():
        # MSE input vector and weight vector

    def convergence_reached():
        return False

    def compute_input_output_distance():
        return 0

    def run_som():
        self.init()

        while not convergence_reached():
            self.sampling()
            self.matching()
            self.updating()


class InputNeuron(Neuron):
    """docstring for InputNeuron"""
    def __init__(self, arg):
        super(InputNeuron, self).__init__()
        self.arg = arg
        
class OutputNeuron(Neuron):
    """docstring for OutputNeuron"""
    def __init__(self, x, y, neighbors):
        super(OutputNeuron, self).__init__()
        self.neighbors = neighbors
        self.x = x
        self.y = y



class City(InputNeuron):
    """docstring for City"""
    def __init__(self, x, y, Neuron):
        
        self.x = x
        self.y = y
        self.Neuron = None

    def set_closest_neuron(Neuron):
        self.Neuron = Neuron

        

def scale_coordinates(coordinates):
    for i in range(2):

        # Max & min scaling
        c_max = max([c[i] for c in coordinates])
        c_min = min([c[i] for c in coordinates])

        # Scale each feature value
        for c in range(len(coordinates)):
            coordinates[c][i] = (coordinates[c][i] - c_min) / (c_max - c_min)

    return coordinates



class TSP(object):

    def __init__(self, file_name):
        # super(TSPInstance, self).__init__()
        self.data = file_reader(file_name)
        self.cities = [row[0] for row in self.data]
        self.coordinates = scale_coordinates([[float(row[1]), float(row[2])] for row in self.data])
        self.distances = []

        # self.learning_rate = None
        # self.radius = None

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

    def get_route_distance(self):
        route_distance = 0
        # for city in range(1, len(self.co))
        return 0

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

# ------------------------------------------
# ****  Parameters ****
printing_frequency = 25
learning_rate = 0.05
decay_rate = 0.05
run_mode = "TSP"
FILE = 1

# ****  MAIN functions ****

if __name__ == '__main__':
    if run_mode == "TSP":
        # Instantiate TSP 
        test = TSP('./data/' + str(FILE) + '.txt')
    elif run_moe == "MNIST":
        test = 0 # To do


    # Create and run SOM
    som = SOM(test, learning_rate, decay_rate, printing_frequency)
    som.run()

    # Visualize solution
    test.plot_map()
    print_distances(test.data)











=======
    # print(test.cities)
    # print(test.coordinates)
    # print_distances(test.data)
    print(test.get_distances())
    # print(test.get_total_distance())
>>>>>>> 1bd5b3cd231f89df38feab010c1c1889c1c28d3f
