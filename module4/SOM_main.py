import numpy as np
import random
from matplotlib import pyplot as plt
from module4.SOM_tools import *

# ------------------------------------------


# *** CLASSES ***

class SOM(object):
    def __init__(self, problem, learning_rate0, learning_rate_tau, printing_frequency, sigma0, tau_sigma,
                 n_output_neurons=None):
        self.problem = problem
        self.learning_rate0 = learning_rate0
        self.learning_rate_tau = learning_rate_tau
        self.sigma0 = sigma0
        self.tau_sigma = tau_sigma
        self.printing_frequency = printing_frequency
        self.n_output_neurons = len(problem.get_elements()) if n_output_neurons is None else n_output_neurons

        self.input_neurons = self.init_input_neurons()
        self.output_neurons = self.init_output_neurons()
        self.connection_weights = self.init_weights(len(self.input_neurons), len(self.output_neurons))

        self.winners = {}
        self.sample_index = 0
        self.winner_index = 0
        self.sigma = 0
        self.winner = self.output_neurons[self.winner_index]
        self.topology_matrix = np.zeros((len(self.output_neurons), len(self.output_neurons)))
        self.init_topology_matrix()

    @staticmethod
    def init_input_neurons():
        return problem.get_elements()

    def init_output_neurons(self):
        # Targeted data structures
        output_neurons = []
        neighbor_matrix = [[]]
        lateral_distances = [[]]

        # Distribute points over circle circumference
        temp = PointsInCircum(0.2, n=self.n_output_neurons)
        xs, ys = [row[0] for row in temp], [row[1] for row in temp]

        # Create output neurons
        for i in range(self.n_output_neurons):
            output_neurons.append(OutputNeuron(xs[i], ys[i]))

        # Set output neuron neighbors in OutputNeuron class
        for i, n in enumerate(output_neurons):
            if i==0:
                n.set_neighbors([output_neurons[-1], output_neurons[1]])
            elif i == len(output_neurons) - 1:
                n.set_neighbors([output_neurons[i-1], output_neurons[0]])
            else:
                n.set_neighbors([output_neurons[i-1], output_neurons[i+1]])

        # Create neighborhood matrix
        self.neighbor_matrix = self.create_neighborhood_matrix(output_neurons)

        # Create lateral distance matrix
        self.lateral_distances = self.compute_lateral_distances(output_neurons)

        return output_neurons

    def init_topology_matrix(self):
        for i, x in enumerate(self.input_neurons):
            winner_index, winner = self.compute_winning_neuron(i, x)
            self.update_topologies(0, winner_index)

    # *** WEIGHTS ***

    def init_weights(self, len_input, len_output):
        weights = [[random.uniform(0,1) for i in range(len_input)] for i in range(len_output)]
        return weights

    def set_winner_index(self, index):
        self.winner_index = index
        self.winner = self.output_neurons[index]

    def set_sample_index(self, index):
        self.sample_index = index

    def update_weights(self, time_step):
        # Set iteration dependent variables
        lr = self.compute_learning_rate(time_step)
        #weight_decay = self.compute_weight_decay(time_step)
        
        # # Update the weights according to slide L16-10
        # for i in range(len(self.input_neurons)):
        #     for j in range(len(self.output_neurons)):
        #         # delta_w_ij = lr * self.topology_matrix[i][j] * (self.input_neurons[i] - self.connection_weights[i][j])
        #         delta_w_ij = lr * self.topology_matrix[i][j] * (euclidian_distance(self.input_neurons[i],
        #                                                                            self.output_neurons[j]))
        #         self.connection_weights[i][j] += delta_w_ij
        #         self.output_neurons[j].x += delta_w_ij
        #         self.output_neurons[j].y += delta_w_ij

        # New: Update the weights according to slide L16-10
        for j in range(len(self.output_neurons)):
            # Compute deltas
            delta_w_jx = lr * self.topology_matrix[self.winner_index][j] * (
            self.input_neurons[self.sample_index].x - self.output_neurons[j].x)
            delta_w_jy = lr * self.topology_matrix[self.winner_index][j] * (
            self.input_neurons[self.sample_index].y - self.output_neurons[j].y)

            # Update coordinates
            self.output_neurons[j].x += delta_w_jx
            self.output_neurons[j].y += delta_w_jy
        print('#')
        print(self.topology_matrix[self.winner_index][j])
        print()

        # print(self.topology_matrix[0])

    def compute_learning_rate(self, time_step):
        return self.learning_rate0 * math.exp(-time_step / self.learning_rate_tau)

    def update_topologies(self, time_step, winner_index):
        topology_matrix = [[0 for j in range(self.n_output_neurons)] for i in range(len(self.input_neurons))]
        self.sigma = self.compute_sigma_t(time_step)

        for j in range(len(topology_matrix[winner_index])):
            self.topology_matrix[winner_index][j] = math.exp(- self.lateral_distances[winner_index][j] ** 2 / (2 * self.sigma ** 2))

    def compute_sigma_t(self, time_step):
        return max(self.sigma0 * math.exp(- time_step / self.tau_sigma), 0.01)
    
    # Assuming highest weight value decides which output neuron is winning
    # def compute_winning_neurons_for_all(self):
    #     # Winners is a dictionary mapping input vector x to its winning neuron
    #
    #
    #     for i,x in enumerate(self.input_neurons):
    #         # Something is wrong
    #         previous_winner = x.get_output_neuron() # Previous winner is an output neuron. Want to remove City from this neuron
    #         self.winners[x] = argmin(self.discriminants(x))
    #
    #         # Connect City and Neuron
    #         x.set_output_neuron(self.winners[x])
    #         self.winners[x].attach_input_vector(x) # Attach City to new output neuron
    #
    #         # Remove previous connection
    #         previous_winner.remove_input_vector(x) # Remove City from previous output neuron


    # Assuming highest weight value decides which output neuron is winning
    def compute_winning_neuron(self, i, x):
        # Something is wrong
        previous_winner = x.get_output_neuron() # Previous winner is an output neuron. Want to remove City from this neuron
        discriminant_list = (self.discriminant_function()[i])
        arg_min, dist = argmin(discriminant_list)
        winner =  self.output_neurons[arg_min]
        self.winners[x] = winner

        # Connect City and Neuron
        x.set_output_neuron(self.winners[x])
        self.winners[x].attach_input_vector(x) # Attach City to new output neuron
        
        # Remove previous connection
        try:
            previous_winner.remove_input_vector(x) # Remove City from previous output neuron
        except:
            # Do nothing
            pass

        return arg_min, winner

    def compute_total_cost(self):
        if self.problem.get_output_neuron_structure() == "2D_lattice":
            pass     # Todo

        elif self.problem.get_output_neuron_structure() == "ring":
            # Cost is equal to dist(xN, x0) + sum(dist(xi, xi+1) for i in output_neurons)
            return euclidian_distance(self.output_neurons[0], self.output_neurons[-1]) + \
                   sum([euclidian_distance(x, self.output_neurons[i+1]) for i, x in enumerate(self.output_neurons[:-1])])
        return 0

    def create_neighborhood_matrix(self, output_neurons):
        n_output_neurons = self.n_output_neurons
        neighbor_matrix = [[0 for i in range(n_output_neurons)] for j in range(n_output_neurons)]
        # Depending on output neuron structure, create the lateral distance matrix
        if self.problem.get_output_neuron_structure() == "2D_lattice":
            # To do
            return neighbor_matrix

        elif(self.problem.get_output_neuron_structure() == "ring"):
            for i in range(len(neighbor_matrix)):
                try:
                    neighbor_matrix[i][i+1] = 1
                except:
                    # Do nothing
                    pass
                try:
                    neighbor_matrix[i][i-1] = 1
                except:
                    # Do nothing
                    pass

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
                    lateral_distances[i][j] = min(abs(i - j), abs(n_output_neurons - j + i), abs(n_output_neurons - i + j))
                    lateral_distances[j][i] = lateral_distances[i][j]
        
        return lateral_distances
    
    # def discriminants(self):
    #
    #     # This array is supposed to be equal to the d_j(x) of slide L16-8
    #     d_js = [0 for j in range(len(self.output_neurons))]
    #     D = len(self.input_neurons)
    #
    #     # For each output neuron, compute the MSE between the input vector and each corresponding weight vector
    #     for j in range(len(self.output_neurons)):
    #         temp_sum = 0
    #         for i in range(D):
    #             # temp_sum += (x[i] - self.connection_weights[i][j])
    #         d_js[j] = temp_sum
    #
    #     return d_js

    def convergence_reached(self, time_steps):
        if self.problem.get_output_neuron_structure() == "2D_lattice":
            pass  # Todo

        elif self.problem.get_output_neuron_structure() == "ring":

            for neuron in self.output_neurons:
                if len(neuron.get_attached_input_vectors()) == 0:  # if there is not a one-to-one relationship between input and output nodes
                    return False

            # Check distance between
            for neuron in self.input_neurons:
                if euclidian_distance(neuron, neuron.get_output_neuron) > self.legal_radius:
                    return False

            # Todo: legg til flere krav til convergence.
            return time_steps < MAX_ITERATIONS

    def discriminant_function(self):
        # Depending on output neuron structure
        if self.problem.get_output_neuron_structure() == "2D_lattice":
            pass     # Todo

        elif self.problem.get_output_neuron_structure() == "ring":
            distances = []
            for index1, n1 in enumerate(self.input_neurons):
                distances.append([])
                for n2 in self.output_neurons:
                    dist = euclidian_distance(n1, n2)
                    distances[index1].append(dist)
            return np.array(distances)

    def plot_map(self, plot_counter):

        # Depending on output neuron structure
        if self.problem.get_output_neuron_structure() == "2D_lattice":
            pass  # Todo

        elif self.problem.get_output_neuron_structure() == "ring":

            # if plot_counter == 0:
            if True:
                fig, ax = plt.subplots()

                ax.plot([c[0] for c in self.problem.coordinates], [c[1] for c in self.problem.coordinates], marker='*', c='gold',
                        markersize=15, linestyle='None')
                ax.set_xlim(0, 1.05)  # adjust figure axes to max x- and y-values, i.e. 0 and 1 (as they are normalized)
                ax.set_ylim(0, 1.05)  # use 1.05 to have some margin on the top and right side

                # plt.pause(0.5)

            neuron_plot, = ax.plot([n.x for n in self.output_neurons], [n.y for n in self.output_neurons],
                                   marker='o', markerfacecolor='None', c='green', markersize=10, linestyle=':')
            # if plot_counter == 0:
            #     neuron_plot, = ax.plot([n.x for n in self.output_neurons], [n.y for n in self.output_neurons],
            #                    marker='o', markerfacecolor='None', c='green', markersize=10, linestyle=':')
            # else:
            #     neuron_plot.set_data([neuron.x for neuron in self.output_neurons], [neuron.y for neuron in self.output_neurons])

            plt.pause(0.5)

    def run(self):
        time_counter = 0
        plot_counter = 0
        while not self.convergence_reached():
            # Sample input vector
            sample_index = random.randint(0, len(self.input_neurons)-1)
            x_sample = self.input_neurons[sample_index]

            # Match
            winner_index, winner = self.compute_winning_neuron(sample_index, x_sample)
            self.update_topologies(time_counter, winner_index)

            # Update
            self.update_weights(time_counter)
            time_counter += 1

            # New
            if PRINTING_MODE == True and time_counter % self.printing_frequency == 0:
                self.plot_map(plot_counter)
                plot_counter += 1
                print(time_counter)

                print(self.sigma)
                print(self.compute_learning_rate(time_counter))
                print(self.output_neurons[0].x, self.output_neurons[1].x, self.output_neurons[2].x)

        # New
        return self.compute_total_cost()


# ------------------------------------------


# Abstract class for input neurons
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

    def get_attached_input_vectors(self):
        return self.attached_input_vectors


# Sub-class for TSP-problems
class City(InputNeuron):
    def __init__(self, x, y):
        InputNeuron.__init__(self)
        self.x = x
        self.y = y
        self.output_neuron = None #To do: Necessary?

    def set_output_neuron(self, OutputNeuron):
        self.output_neuron = OutputNeuron
    
    def get_output_neuron(self):
        return self.output_neuron


# Sub-class for problems using images from MNIST
class Image_input(InputNeuron):
    def __init__(self, x, y):
        InputNeuron.__init__(self)
        # self.x = x    # TODO: Image skal muligens ikke ha x og y, men noe annet som input
        # self.y = y

# ------------------------------------------


class Problem(object):

    def __init__(self, output_neuron_structure):
        self.output_neuron_structure = output_neuron_structure

    def get_elements(self):
        pass


class TSP(Problem):

    def __init__(self, file_name):
        Problem.__init__(self, 'ring')
        self.data = file_reader(file_name)
        self.coordinates = scale_coordinates([[float(row[1]), float(row[2])] for row in self.data])
        self.cities = [City(city[0], city[1]) for city in self.coordinates]
        self.distances = []

    def get_elements(self):
        return self.cities

    def get_output_neuron_structure(self):
        return self.output_neuron_structure


class Image(Problem):

    def __init__(self, file_name):
        Problem.__init__(self, '2D_lattice')
        self.data = file_reader(file_name)

    def get_elements(self):
        return []

    def get_output_neuron_structure(self):
        return self.output_neuron_structure

# ------------------------------------------


# ****  Parameters ****
RUN_MODE = "TSP"
FILE = 1
L_RATE0 = 0.2
L_RATE_tau = 50000
printing_frequency = 1000
sigma0 = 0.05
tau_sigma = 50000
n_output_neurons = None

# New
PRINTING_MODE = False
MAX_ITERATIONS = 100000

# ------------------------------------------

# New
def multiple_runs():
    L_RATE0s = [float(x*math.log(x)+0.01)/10.0 for x in range(10)]
    L_RATE_taus = [100*x for x in range(10)]
    sigma0s =  [float(x*math.log(x)+0.01)/10.0 for x in range(10)]
    tau_sigams = [100*x for x in range(50)]

    res_array = [[]]

    for L_RATE0 in L_RATE0s:
        for L_RATE_tau in L_RATE_taus:
            for sigma0 in sigma0s:
                for tau_sigma in tau_sigmas:
                    som = SOM(problem, L_RATE0, L_RATE_tau, printing_frequency, sigma0, tau_sigma)
                    dist, cost = som.run()

                    res_array.append([L_RATE0, L_RATE_tau, sigma0, tau_sigma, dist, cost])

    write_results(res_array)

# New
def write_to_file(res_array):
    import xlsxwriter
    book = xlsxwriter.Workbook('module4results.xlsx')
    sheet = book.add_worksheet("Results")

    headers = ['Learning rate0', 'Learning rateTAU', 'Sigma0', 'SigmaTau']
    
    for h in range(len(headers)):
        sheet.write(0, h, headers[h])


    for r in range(len(res_array)):
        for i in range(len(res_array[r])):
            sheet.write(r+1, i, res_array[r][i])


    


# ****  MAIN function ****

if __name__ == '__main__':

    if RUN_MODE == "TSP":
        # Instantiate TSP 
        problem = TSP('./data/' + str(FILE) + '.txt')
    elif RUN_MODE == "MNIST":
        problem = 0    # Todo

    # Create and run SOM
    som = SOM(problem, L_RATE0, L_RATE_tau, printing_frequency, sigma0, tau_sigma)
    som.run()

    # Visualize solution

    plt.show()


