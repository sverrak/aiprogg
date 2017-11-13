import time
import numpy as np
import random
from matplotlib import pyplot as plt
USER = "Sverre1"
if USER == "Sverre":
    pass
    # from SOM_tools import *
else:
    from module4.SOM_tools import *
    from module4.mnist_basics import *
import scipy.spatial.distance as SSD

# ------------------------------------------


# *** CLASSES ***

class SOM(object):
    def __init__(self, problem, learning_rate0, learning_rate_tau, printing_frequency, sigma0, tau_sigma,
                 n_input_neurons=2, classification_frequency=1000, tfrac=0.2):
        self.problem = problem
        self.learning_rate0 = learning_rate0
        self.learning_rate_tau = learning_rate_tau
        self.sigma0 = sigma0
        self.tau_sigma = tau_sigma
        self.printing_frequency = printing_frequency
        self.n_output_neurons = len(problem.get_elements()) if problem.n_output_neurons is None else int(problem.n_output_neurons)
        self.n_input_neurons = n_input_neurons
        self.legal_radius = LEGAL_RADIUS

        self.classification_frequency = classification_frequency
        self.tfrac = tfrac

        self.problem_elements, self.testing_elements = self.init_problem_elements()
        self.input_neurons = self.init_input_neurons()
        self.output_neurons = self.init_output_neurons()
        self.lateral_distances = self.compute_lateral_distances()
        self.connection_weights = self.init_weights(len(self.input_neurons), len(self.output_neurons))

        self.winners = {}
        self.sample_index = 0
        self.winner_index = 0
        self.sigma = 0
        self.winner = self.output_neurons[self.winner_index]
        self.topology_matrix = np.zeros((len(self.output_neurons), len(self.output_neurons)))
        self.init_topology_matrix()
        self.solution_route = []

    def init_input_neurons(self):
        input_neurons = [0 for i in range(self.n_input_neurons)]
        for i in range(len(input_neurons)):
            input_neurons[i] = InputNeuron(self.n_output_neurons)

        return input_neurons

    def init_problem_elements(self):

        if CLASSIFICATION_MODE == True:
               random.shuffle(self.problem.get_elements())
               all_elements = self.problem.get_elements()
               training_elements = all_elements[0:int(self.tfrac*len(all_elements))]
               testing_elements = all_elements[int(self.tfrac*len(all_elements)):]
               return training_elements, testing_elements

        return self.problem.get_elements(), []

    def init_output_neurons(self):
        # Targeted data structures
        output_neurons = []

        if self.problem.get_neuron_structure() == 'ring':
            # Distribute points over circle circumference
            center_x = sum(c.x for c in self.problem_elements)/len(self.problem_elements)
            center_y = sum(c.y for c in self.problem_elements)/len(self.problem_elements)

            temp = PointsInCircum(center_y*0.5, center_x, center_y, n=self.n_output_neurons)
            xs, ys = [row[0] for row in temp], [row[1] for row in temp]

            # Create output neurons
            for i in range(self.n_output_neurons):
                output_neurons.append(OutputNeuron(xs[i], ys[i], 2))

            # Set output neuron neighbors in OutputNeuron class
            for i, n in enumerate(output_neurons):
                if i == 0:
                    n.set_neighbors([output_neurons[-1], output_neurons[1]])
                elif i == len(output_neurons) - 1:
                    n.set_neighbors([output_neurons[i-1], output_neurons[0]])
                else:
                    n.set_neighbors([output_neurons[i-1], output_neurons[i+1]])

            # Create neighborhood matrix
            # self.neighbor_matrix = self.create_neighborhood_matrix(output_neurons)

        else:   # if problem is MNIST
            # Create output neurons
            for j in range(int(self.n_output_neurons / 10)):
                for i in range(int(self.n_output_neurons / 10)):
                    output_neurons.append(OutputNeuron(j, i, 784))

            # Set output neuron neighbors in OutputNeuron class
            for i, n in enumerate(output_neurons):
                # Corner cases
                if i == 0:
                    n.set_neighbors([output_neurons[1+1], output_neurons[i+10]])
                elif i == 9:
                    n.set_neighbors([output_neurons[i-1], output_neurons[i+10]])
                elif i == 90:
                    n.set_neighbors([output_neurons[i+1], output_neurons[i-10]])
                elif i == 99:
                    n.set_neighbors([output_neurons[-2], output_neurons[-10]])

                elif i % 10 == 0:   # other left side cases
                    n.set_neighbors([output_neurons[i-10], output_neurons[i+1], output_neurons[i+10]])
                elif (i-9) % 10 == 0:   # other right side cases
                    n.set_neighbors([output_neurons[i-10], output_neurons[i-1], output_neurons[i+10]])
                elif i < 10:   # other top side cases
                    n.set_neighbors([output_neurons[i+1], output_neurons[i-1], output_neurons[i+10]])
                elif i > 90:   # other bottom side cases
                    n.set_neighbors([output_neurons[i+1], output_neurons[i-1], output_neurons[i-10]])

                # All other cases
                else:
                    n.set_neighbors([output_neurons[i+1], output_neurons[i-1], output_neurons[i-10], output_neurons[i+10]])

        return output_neurons

    def init_topology_matrix(self):
        for i, x in enumerate(self.problem_elements):
            self.winner_index, _ = self.compute_winning_neuron(i, x)
            self.update_topologies(0)

    # *** WEIGHTS ***

    @staticmethod
    def init_weights(len_input, len_output):
        weights = [[random.uniform(0, 1) for _ in range(len_output)] for _ in range(len_input)]
        return weights

    def set_winner_index(self, index):
        self.winner_index = index
        self.winner = self.output_neurons[index]

    def set_sample_index(self, index):
        self.sample_index = index

    def update_weights(self, time_step):
        # Set iteration dependent variables
        lr = self.compute_learning_rate(time_step)

        for j in range(len(self.output_neurons)):
            # Compute deltas
            # delta_w_j = lr * self.topology_matrix[self.winner_index][j] * (np.subtract(
            #     self.problem_elements[self.sample_index], self.output_neurons[j]))
            delta_w_jx = lr * self.topology_matrix[self.winner_index][j] * (self.problem_elements[self.sample_index].x - self.output_neurons[j].x)
            delta_w_jy = lr * self.topology_matrix[self.winner_index][j] * (self.problem_elements[self.sample_index].y - self.output_neurons[j].y)

            # Update coordinates
            # self.output_neurons[j].weights = np.add(self.output_neurons[j].weights, delta_w_j)
            self.output_neurons[j].x += delta_w_jx
            self.output_neurons[j].y += delta_w_jy

    def compute_learning_rate(self, time_step):
        return self.learning_rate0 * math.exp(-time_step / self.learning_rate_tau)

    def update_topologies(self, time_step):
        self.sigma = self.compute_sigma_t(time_step)
        for j in range(len(self.topology_matrix[self.winner_index])):
            self.topology_matrix[self.winner_index][j] = math.exp(- self.lateral_distances[self.winner_index][j] ** 2 / (2 * self.sigma ** 2))

    def compute_sigma_t(self, time_step):
        return max(self.sigma0 * math.exp(- time_step / self.tau_sigma), 0.01)

    # Assuming highest weight value decides which output neuron is winning
    def compute_winning_neuron(self, i, x):
        # Something is wrong
        previous_winner = x.get_output_neuron()    # Previous winner is an output neuron. Want to remove City from this neuron
        discriminant_list = (self.discriminant_function()[i]) # To do: Improve
        arg_min, dist = argmin(discriminant_list)
        winner = self.output_neurons[arg_min]
        self.winners[x] = winner

        # Connect City and Neuron
        x.set_output_neuron(self.winners[x])
        self.winners[x].attach_input_vector(x)  # Attach City to new output neuron
        
        if previous_winner != self.winners[x]:
            # Remove previous connection
            try:
                previous_winner.remove_input_vector(x)  # Remove City from previous output neuron
            except:
                pass

        return arg_min, winner

    def compute_cost_of_path(self, path):
        return sum(euclidian_distance(path[i], path[i+1]) for i in range(len(path[:-1])))

    def compute_optimal_order(self, previous, current_elements, next, guesses=5):
        best_path = [previous] + current_elements + [next]
        cost_of_best_path = self.compute_cost_of_path(best_path)

        for i in range(guesses):
            random.shuffle(current_elements)
            path = [previous] + current_elements + [next]
            cost_of_path = self.compute_cost_of_path(best_path)
            if cost_of_best_path > cost_of_path:
                best_path = path
                cost_of_best_path = cost_of_path

        return best_path[1:-1]

    @staticmethod
    def next_city_heuristic(c0, c_list):
        closest_n = None
        lowest_dist = 0
        for n in c_list:
            dist = euclidian_distance(c0, n)
            if dist > lowest_dist:
                closest_n = n
        return closest_n

    def compute_total_cost(self):
        # To find the total cost, we simply walk around the ring of output neurons and read off all the cities
        # ... in the order they appear. The resulting sequence constitutes a TSP solution
        solution_route = []
        for n in self.output_neurons:
            if len(n.get_attached_input_vectors())==1:
                solution_route.append(n.get_attached_input_vectors()[0])
            elif len(n.get_attached_input_vectors()) > 1:
                optimally_ordered_elements = self.compute_optimal_order(solution_route[-1],
                    n.get_attached_input_vectors(), self.next_city_heuristic(solution_route[-1], n.get_attached_input_vectors()))
                solution_route = solution_route + optimally_ordered_elements

        if len(solution_route) != len(self.problem_elements):
            raise RuntimeError("A non-correct number of cities have been added to the solution route.", len(solution_route),
                               'have been added, of the correct', len(self.problem_elements))

        self.solution_route = solution_route

        # Cost is equal to the euclidian distance between the cities in the order they appear
        return euclidian_distance(solution_route[0], solution_route[-1]) + \
               sum([euclidian_distance(solution_route[i], solution_route[i + 1]) for i in
                    range(len(solution_route)-1)])

    # def create_neighborhood_matrix(self, output_neurons):
    #     n_output_neurons = self.n_output_neurons
    #     neighbor_matrix = [[0 for i in range(n_output_neurons)] for j in range(n_output_neurons)]
    #     # Depending on output neuron structure, create the lateral distance matrix
    #     if self.problem.get_neuron_structure() == "2D_lattice":
    #         # To do
    #         return neighbor_matrix
    #
    #     elif self.problem.get_neuron_structure() == "ring":
    #         for i in range(len(neighbor_matrix)):
    #             try:
    #                 neighbor_matrix[i][i+1] = 1
    #             except:
    #                 pass
    #             try:
    #                 neighbor_matrix[i][i-1] = 1
    #             except:
    #                 pass
    #
    #     return neighbor_matrix

    def compute_lateral_distances(self):
        lateral_distances = [[0 for _ in range(self.n_output_neurons)] for _ in range(self.n_output_neurons)]

        # Depending on output neuron structure, create the lateral distance matrix
        if self.problem.get_neuron_structure() == "2D_lattice":
            neuron_coordinates = [[neuron.x, neuron.y] for neuron in self.output_neurons]
            return SSD.cdist(neuron_coordinates, neuron_coordinates, metric='cityblock')

        elif self.problem.get_neuron_structure() == "ring":
            for i in range(self.n_output_neurons):
                for j in range(i, self.n_output_neurons):
                    # To do: describe logic: 
                    lateral_distances[i][j] = min(abs(i - j), abs(self.n_output_neurons - j + i), abs(self.n_output_neurons - i + j))
                    lateral_distances[j][i] = lateral_distances[i][j]

        return lateral_distances
    
    def compute_input_output_distance(self):
        temp_sum = 0
        for neuron in self.problem_elements:
            temp_sum += euclidian_distance(neuron, neuron.get_output_neuron())

        return temp_sum

    def convergence_reached(self, time_steps):
        # Todo: legg til flere krav til convergence.

        if time_steps > MAX_ITERATIONS:
            return True

        # elif self.problem.get_neuron_structure() == "2D_lattice":
            # pass

        elif self.problem.get_neuron_structure() == "ring":

            # Check distance between
            for city in self.problem_elements:
                if euclidian_distance(city, city.get_output_neuron()) > self.legal_radius:
                    return False

            for city in self.output_neurons:
                if len(city.get_attached_input_vectors()) == 0:  # if there is not a one-to-one relationship between input and output nodes
                    return False

        return False

    def discriminant_function(self):
        if self.problem.get_neuron_structure() == "2D_lattice":
            inputs = [neuron.pixels for neuron in self.problem_elements]
            outputs = [[neuron.x, neuron.y] for neuron in self.output_neurons]
            return SSD.cdist(inputs, outputs, metric='euclidean')
        else:
            inputs = [[neuron.x, neuron.y] for neuron in self.problem_elements]
            outputs = [[neuron.x, neuron.y] for neuron in self.output_neurons]
            return SSD.cdist(inputs, outputs, metric='euclidean')

    # Animate how the TSP is solved
    def plot_TSP(self, first_run=False, has_found_solution=False):

        # Depending on output neuron structure
        if self.problem.get_neuron_structure() == "2D_lattice":
            pass  # Todo

        elif self.problem.get_neuron_structure() == "ring":

            if first_run is True:
                global fig, ax, neuron_plot
                fig, ax = plt.subplots()

                max_x = max(c.x for c in self.problem_elements)
                max_y = max(c.y for c in self.problem_elements)

                ax.plot([c[0] for c in self.problem.coordinates], [c[1] for c in self.problem.coordinates], marker='*', c='gold',
                        markersize=15, linestyle='None')
                ax.set_xlim(0, max_x * 1.05)  # adjust figure axes to max x- and y-values, i.e. 0 and 1 (as they are normalized)
                ax.set_ylim(0, max_y * 1.05)  # use 1.05 to have some margin on the top and right side

                neuron_plot, = ax.plot([n.x for n in self.output_neurons], [n.y for n in self.output_neurons],
                               marker='o', markerfacecolor='None', c='green', markersize=10, linestyle=':')
            else:
                neuron_plot.set_data([neuron.x for neuron in self.output_neurons], [neuron.y for neuron in self.output_neurons])

            if has_found_solution:
                # Plot solution route between cities on top of map
                global solution_plot
                solution_plot,  = ax.plot([n.x for n in self.solution_route], [n.y for n in self.solution_route],
                        marker='*', c='blue', markersize=12, linestyle='-')

            plt.pause(PLOT_SPEED)

    def run(self):
        self.time_counter = 0
        first_plot = True
        while not self.convergence_reached(self.time_counter):
            # Sample input vector
            self.set_sample_index(random.randint(0, len(self.problem_elements)-1))
            x_sample = self.problem_elements[self.sample_index]

            # Match
            self.winner_index, _ = self.compute_winning_neuron(self.sample_index, x_sample)
            self.update_topologies(self.time_counter)

            # Update
            self.update_weights(self.time_counter)
            self.time_counter += 1

            if PRINTING_MODE is True and self.time_counter % self.printing_frequency == 0:
                self.plot_TSP(first_plot)
                first_plot = False

            if CLASSIFICATION_MODE is True and self.time_counter % self.classification_frequency == 0:

                # Turn off learning and run each case through the network and record the cases and comp
                training_performance = self.do_classification(self.problem_elements, "Training")
                testing_performance = self.do_classification(self.testing_elements, "Testing")

            if SINGLE_RUN:
                if self.time_counter % 1000 == 0:
                    print(self.time_counter)

        if self.problem.get_neuron_structure() == "2D_lattice":
            return 0
        else:
            return self.compute_input_output_distance(), self.compute_total_cost()

    def do_classification(self, data, data_description):
        # List of boolean variables indicating whether the network guessed right on sample x or not
        correct_values = []
        winners = {}

        # 1 Run each case through the network without learning
        for sample_index, sample_x in enumerate(data):
            winner_index, winner = self.compute_winning_neuron(sample_index, sample_x)
            winners[sample_x] = winner

        # 2 Update output neuron class labels
        for output_neuron in self.output_neurons:
            output_neuron.get_majority_class()

        # 3 Add indicator indicating whether the guessed target was correct or not
        for sample_index, sample_x in enumerate(data):
            correct_values.append(1 if winners[sample_x].get_majority_class() == sample_x.get_target() else 0)

        # Compute the classification performance
        performance = float(sum(correct_values)) / float(len(correct_values))

        print(data_description + " Score: " + str(performance) + "%")
        return performance

    def run_more(self, iterations):
        # Increase the iteration cap
        global MAX_ITERATIONS
        MAX_ITERATIONS += iterations

        while not self.convergence_reached(self.time_counter):
            # Sample input vector
            self.set_sample_index(random.randint(0, len(self.problem_elements)-1))
            x_sample = self.problem_elements[self.sample_index]

            # Match
            self.winner_index, _ = self.compute_winning_neuron(self.sample_index, x_sample)
            self.update_topologies(self.time_counter)

            # Update
            self.update_weights(self.time_counter)
            self.time_counter += 1

            if PRINTING_MODE is True and self.time_counter % self.printing_frequency == 0:
                self.plot_TSP(False)    # Firstplot = False

            # Continue classifying elements
            if CLASSIFICATION_MODE is True and self.time_counter % self.classification_frequency == 0:

                # Turn off learning and run each case through the network and record the cases and comp
                training_performance = self.do_classification(self.problem_elements, "Training")
                testing_performance = self.do_classification(self.testing_elements, "Testing")

        return self.compute_input_output_distance(), self.compute_total_cost()


# -----------------------------------------


# Abstract class for input neurons
class InputNeuron(object):
    def __init__(self, n_output_neurons):
        self.output_neuron_values = [0 for x in range(n_output_neurons)]   # Todo: Necessary?

    def set_output_neuron_value(self, index, output_neuron):
        self.output_neuron_values[index] = output_neuron
    
    def get_output_neuron(self, index):
        return self.output_neuron_values[index]


class OutputNeuron(object):
    def __init__(self, x, y, len_weights):
        super(OutputNeuron, self).__init__()
        self.x = x
        self.y = y
        self.weights = [0] * len_weights
        self.neighbors = []
        self.attached_input_vectors = []
        self.majority_class = None

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors

    def attach_input_vector(self, input_vector):
        if input_vector not in self.attached_input_vectors:
            self.attached_input_vectors.append(input_vector)

    def remove_input_vector(self, input_vector):
        if input_vector in self.attached_input_vectors:
            self.attached_input_vectors.remove(input_vector)

    def get_attached_input_vectors(self):
        return self.attached_input_vectors

    def set_majority_class(self):
        # Necessary data containers
        classes = {}
        best_class = None
        best_class_count = 0

        # Try catch to eliminate non-target input_vector cases
        try:
            # Fill classes dictionary
            for input_vector in self.get_attached_input_vectors():
                if(input_vector.target_value in classes.keys()):
                    classes[input_vector.target_value] += 1
                else:
                    classes[input_vector.target_value] = 1

            # Find best key max(classes[key])
            for k in classes.keys():
                if classes[k] > best_class_count:
                    best_class = k
                    best_class_count = classes[k]

        except:
            print("ERROR. Input vector has no feature target_value")

        self.majority_class = best_class

    # New
    def get_majority_class(self):
        return self.majority_class


# Generic Problem element
class ProblemElement(object):
    def __init__(self):
        self.output_neuron = None

    def set_output_neuron(self, OutputNeuron):
        self.output_neuron = OutputNeuron
    
    def get_output_neuron(self):
        return self.output_neuron


# Sub-class for TSP-problems
class City(ProblemElement):
    def __init__(self, x, y):
        super(City, self).__init__()
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y


# Sub-class for problems using images from MNIST
class Image(ProblemElement):
    def __init__(self, pixels, target):
        super(Image, self).__init__()
        self.pixels = pixels
        self.target = target

    def get_target(self):
        return self.target

# ------------------------------------------


class Problem(object):

    def __init__(self, output_neuron_structure, n_output_neurons):
        self.neuron_structure = output_neuron_structure
        self.n_output_neurons = n_output_neurons

    def get_elements(self):
        pass

    def get_neuron_structure(self):
        return self.neuron_structure


class TSP(Problem):

    def __init__(self, file_name):
        Problem.__init__(self, 'ring', None)
        self.data = file_reader(file_name)
        # self.coordinates, self.scale_down_factor = scale_coordinates([[float(row[1]), float(row[2])] for row in self.data])
        self.coordinates = [[float(row[1]), float(row[2])] for row in self.data]    # todo: coordinates are now not scaled
        self.cities = [City(city[0], city[1]) for city in self.coordinates]

    def get_elements(self):
        return self.cities


class MNIST(Problem):

    def __init__(self, n_output_neurons):
        Problem.__init__(self, '2D_lattice', n_output_neurons)
        self.image_data, self.target_data = load_mnist()
        self.images = self.init_images()
        self.n_output_neurons = n_output_neurons

    def init_images(self):
        image_list = []
        for i, image in enumerate(self.image_data):
            try:
                flat_image = flatten_image(image)
            except:
                flat_image = image

            image_list.append(Image(flat_image, self.target_data[i]))

        return image_list

    def get_elements(self):
        return self.images

# ------------------------------------------


def multiple_runs(problem, L_RATE0s, L_RATE_taus, sigma0s, tau_sigmas):
    print('Number of iterations to do:', len(L_RATE0s)*len(L_RATE_taus)*len(sigma0s)*len(tau_sigmas))

    iteration_counter = 0
    with open('results_of_testing.txt', 'w') as file:
        for L_RATE0 in L_RATE0s:
            for L_RATE_tau in L_RATE_taus:
                for sigma0 in sigma0s:
                    for tau_sigma in tau_sigmas:
                        som = SOM(problem, L_RATE0, L_RATE_tau, printing_frequency, sigma0, tau_sigma)
                        dist, cost = som.run()

                        res = [L_RATE0, L_RATE_tau, sigma0, tau_sigma, dist, cost]
                        file.write('\t'.join([str(i) for i in res] + ['\n']))
                        iteration_counter += 1
                        print(iteration_counter)
                        plt.close()

                    file.flush()


# ------------------------------------------

# ****  Parameters ****
fig, ax, neuron_plot, solution_plot = None, None, None, None

RUN_MODE = "TSP"
FILE = 1

L_RATE0 = 0.5
L_RATE_tau = 50000
printing_frequency = 100
classification_frequency = 100000
sigma0 = 3
tau_sigma = 5000
MAX_ITERATIONS = 2000

SINGLE_RUN = True
CLASSIFICATION_MODE = RUN_MODE == "MNIST"
PLOT_SPEED = 0.001
LEGAL_RADIUS = 10
PRINTING_MODE = True

# ------------------------------------------


# ****  MAIN function ****

if __name__ == '__main__':

    start_time = time.time()

    if RUN_MODE == "TSP":
        # Instantiate TSP
        if(USER == "Sverre"):
            problem = TSP('/Users/sverreakersveen/Documents/Skole/5klasse/AIprogg/module4/data/' + str("djibouti89") + '.txt')

        else:
            problem = TSP('./data/' + str(FILE) + '.txt')
    elif RUN_MODE == "MNIST":
        problem = MNIST(100)    # Todo

    if SINGLE_RUN:
        # Create and run SOM
        som = SOM(problem, L_RATE0, L_RATE_tau, printing_frequency, sigma0, tau_sigma)
        print('Number of elements in data set:', len(som.problem_elements))
        print('Total route cost:', som.run()[1])

        som.plot_TSP(has_found_solution=True)

        # Continue?
        more_runs = input("\n\n Run more? ") == "y"
        while more_runs:
            solution_plot.remove()  # remove solution layer from plot until new solution is found

            # Get the number of additional iterations
            n_iterations = int(input("Number of iterations: "))

            # Run n more iterations
            print("... Running more iterations ... ")
            print('Total route cost:', som.run_more(n_iterations)[1])
            som.plot_TSP(has_found_solution=True)

            # Continue?
            more_runs = input("\n\n Run more? ") == "y"
    else:
        L_RATE0s = [0.3 + x * 0.2 for x in range(0, 4, )]
        L_RATE_taus = [10000 * x for x in range(1, 10, 2)]
        sigma0s = [x for x in range(2, 11, 3)]
        tau_sigmas = [10000, 20000]  # * x for x in range(1, 10)]

        PRINTING_MODE = False
        multiple_runs(problem, L_RATE0s, L_RATE_taus, sigma0s, tau_sigmas)

    print('Run time:', time.time() - start_time, 'seconds')
