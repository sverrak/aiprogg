import time
import numpy as np
import random
from matplotlib import pyplot as plt
import networkx as nx
import matplotlib.animation as animation
import scipy.spatial.distance as SSD
USER = "Sverre1"
if USER == "Sverre":
    from SOM_tools import *
else:
    from module4.SOM_tools import *
    from module4.mnist_basics import *


# ------------------------------------------


# *** CLASSES ***

class SOM(object):
    def __init__(self, problem, learning_rate0, learning_rate_tau, printing_frequency, sigma0, tau_sigma,
                 n_input_neurons=2, tfrac=0.8):
        self.problem = problem
        self.learning_rate0 = learning_rate0
        self.learning_rate_tau = learning_rate_tau
        self.sigma0 = sigma0
        self.tau_sigma = tau_sigma
        self.printing_frequency = printing_frequency
        self.n_output_neurons = len(problem.get_elements()) if problem.n_output_neurons is None else int(problem.n_output_neurons)
        self.n_input_neurons = n_input_neurons
        self.legal_radius = LEGAL_RADIUS

        self.tfrac = tfrac

        self.problem_elements, self.testing_elements = self.init_problem_elements()
        self.input_neurons = self.init_input_neurons()
        self.output_neurons = self.init_output_neurons()
        self.lateral_distances = self.compute_lateral_distances()

        self.winners = {}
        self.sample_index = 0
        self.winner_index = 0
        self.training_accuracy = 0
        self.testing_accuracy = 0
        self.time_counter = 0
        self.first_run = True
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

        if CLASSIFICATION_MODE:
            all_elements = self.problem.get_elements()
            random.shuffle(all_elements)
            training_elements = all_elements[0:int(self.tfrac*len(all_elements))]
            testing_elements = all_elements[int(self.tfrac*len(all_elements)):]
            return training_elements, testing_elements

        return self.problem.get_elements(), []

    def init_output_neurons(self, weights=None):
        # Targeted data structures
        output_neurons = []

        if type(self.problem) is TSP:
            # Distribute points over circle circumference
            center_x = sum(c.weights[0] for c in self.problem_elements)/len(self.problem_elements)
            center_y = sum(c.weights[1] for c in self.problem_elements)/len(self.problem_elements)

            temp = PointsInCircum(center_y*0.5, center_x, center_y, n=self.n_output_neurons)
            #xs, ys = [row[0] for row in temp], [row[1] for row in temp]

            # Create output neurons
            for i in range(self.n_output_neurons):
                output_neurons.append(OutputNeuron(temp[i])) # Now, we only initiate OutputNeurons with their weight vector

            # Set output neuron neighbors in OutputNeuron class
            for i, n in enumerate(output_neurons):
                if i == 0:
                    n.set_neighbors([output_neurons[-1], output_neurons[1]])
                elif i == len(output_neurons) - 1:
                    n.set_neighbors([output_neurons[i-1], output_neurons[0]])
                else:
                    n.set_neighbors([output_neurons[i-1], output_neurons[i+1]])

        else:   # if problem is MNIST
            # Create output neurons
            grid_size = 10
            for j in range(grid_size):
                for i in range(grid_size):
                    if(weights == None):
                        output_neurons.append(OutputNeuron([random.uniform(0, 0.1)] * 784))
                    else:
                        output_neurons.append(OutputNeuron(weights[j*grid_size + i]))

            # Set output neuron neighbors in OutputNeuron class. We use the networkX-package to do this smoothly
            grid = nx.grid_2d_graph(10, 10)
            for j in range(10):
                for i in range(10):
                    output_neurons[j*10+i].set_neighbors([output_neurons[x[0]*10+x[1]] for x in grid.neighbors((j, i))])

        return output_neurons

    def init_topology_matrix(self):
        for i, x in enumerate(self.problem_elements):
            self.winner_index, _ = self.compute_winning_neuron(i, x, 'Training')
            self.update_topologies(0)

    # *** WEIGHTS ***

    def set_winner_index(self, index):
        self.winner_index = index
        self.winner = self.output_neurons[index]

    def set_sample_index(self, index):
        self.sample_index = index

    def update_weights(self, time_step):
        # Set iteration dependent variables
        lr = self.compute_learning_rate(time_step)
        topology_c = self.topology_matrix[self.winner_index]

        for j in range(len(self.output_neurons)):
            # Compute deltas
            delta_w_j = lr * topology_c[j] * np.subtract(self.problem_elements[self.sample_index].get_feature_values(),
                                                          self.output_neurons[j].weights)

            # Update coordinates
            self.output_neurons[j].weights = np.add(self.output_neurons[j].weights, delta_w_j)

    def compute_learning_rate(self, time_step):
        return self.learning_rate0 * math.exp(-time_step / self.learning_rate_tau)

    def update_topologies(self, time_step):
        self.sigma = self.compute_sigma_t(time_step)
        for j in range(len(self.topology_matrix[self.winner_index])):
            self.topology_matrix[self.winner_index][j] = math.exp(- self.lateral_distances[self.winner_index][j] ** 2 / (2 * self.sigma ** 2))

    def compute_sigma_t(self, time_step):
        return max(self.sigma0 * math.exp(- time_step / self.tau_sigma), 0.01)

    # Assuming highest weight value decides which output neuron is winning
    def compute_winning_neuron(self, i, x, data_description):
        # Something is wrong
        previous_winner = x.get_output_neuron()    # Previous winner is an output neuron. Want to remove City from this neuron
        discriminant_list = self.discriminant_function(i, data_description)   # To do: Improve - we don't need the whole matrix
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

    @staticmethod
    def compute_cost_of_path(path):
        return sum(euclidian_distance_input(path[i], path[i+1]) for i in range(len(path)-1))

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
            dist = euclidian_distance_input(c0, n)
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
                if(len(solution_route) > 0):
                    optimally_ordered_elements = self.compute_optimal_order(solution_route[-1], n.get_attached_input_vectors(), self.next_city_heuristic(solution_route[-1], n.get_attached_input_vectors()))
                else:
                    optimally_ordered_elements = n.get_attached_input_vectors()
                solution_route = solution_route + optimally_ordered_elements

        if len(solution_route) != len(self.problem_elements):
            raise RuntimeError("A non-correct number of cities have been added to the solution route.", len(solution_route),
                               'have been added, of the correct', len(self.problem_elements))

        self.solution_route = solution_route

        # Cost is equal to the euclidian distance between the cities in the order they appear
        return euclidian_distance(solution_route[0], solution_route[-1]) + \
               sum([euclidian_distance(solution_route[i], solution_route[i + 1]) for i in
                    range(len(solution_route)-1)])

    # Help function for compute_lateral_distances returning the topological coordinates of each neuron
    # Returns a dict with (neuron, topological_coordinates) mappings
    def get_topologic_indices(self):
        indices = {}
        grid_size = 10
        for i, neuron in enumerate(self.output_neurons):
            row = i // grid_size
            column = i - grid_size * row
            indices[neuron] = list([row, column])

        return indices

    def compute_lateral_distances(self):
        lateral_distances = [[0 for _ in range(self.n_output_neurons)] for _ in range(self.n_output_neurons)]

        # Depending on output neuron structure, create the lateral distance matrix
        if type(self.problem) is MNIST:
            topological_indices = self.get_topologic_indices()

            neuron_coordinates = [topological_indices[neuron] for neuron in self.output_neurons]
            return SSD.cdist(neuron_coordinates, neuron_coordinates, metric='cityblock')

        elif type(self.problem) is TSP:
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

    def convergence_reached(self):

        if self.time_counter > MAX_ITERATIONS:
            return True

        elif type(self.problem) is TSP:

            # Check distance between
            for city in self.problem_elements:
                if euclidian_distance(city, city.get_output_neuron()) > self.legal_radius:
                    return False

            for city in self.output_neurons:
                if len(city.get_attached_input_vectors()) == 0:  # if there is not a one-to-one relationship between input and output nodes
                    return False

        return False

    def discriminant_function(self, i, data_description):
        # Should suffice in both cases: for TSP and MNIST
        if data_description == 'Training':
            # inputs = [neuron.get_feature_values() for neuron in self.problem_elements]
            inputs = self.problem_elements[i].get_feature_values()
        elif data_description == 'Testing':
            # inputs = [neuron.get_feature_values() for neuron in self.testing_elements]
            inputs = self.testing_elements[i].get_feature_values()
        outputs = [neuron.weights for neuron in self.output_neurons]
        return SSD.cdist(np.atleast_2d(inputs), outputs, metric='euclidean')[0]

    def animate(self, i):
        pos = nx.spring_layout(self.grid, iterations=1000)
        classes = [self.node_classes.get(node) for node in self.grid.nodes()]
        nx.draw(self.grid, pos=pos, cmap=plt.get_cmap('jet'), node_color=classes)
        plt.pause(1)

    def update_node_classes(self):
        node_classes = dict()
        for i, n in enumerate(self.grid.nodes()):
            node_classes[n] = self.output_neurons[i].majority_class
            # node_classes[n] = random.randint(0, 9)  # TO DO: erstatt med linja ovenfor når vi har begynt å sette majority class til noe
        nx.set_node_attributes(self.grid, node_classes, 'class')
        self.node_classes = node_classes

    # Animate how the TSP or MNIST is solved
    def plot(self, has_found_solution=False):

        # Depending on output neuron structure
        if type(self.problem) is MNIST and has_found_solution:
            # if self.first_run is True and self.time_counter == MAX_ITERATIONS:
            # if self.time_counter == MAX_ITERATIONS or has_found_solution:

            self.grid = nx.grid_2d_graph(10, 10)
            self.update_node_classes()

            classes = [self.node_classes.get(node) for node in self.grid.nodes()]
            pos = nx.spring_layout(self.grid, iterations=1000)      # TODO: make grid straight, not bended

            nx.draw(self.grid, pos=pos, cmap=plt.get_cmap('jet'), node_color=classes)
            nx.draw_networkx_labels(self.grid, pos, self.node_classes, font_size=16)

                # global fig2
                # fig2 = plt.gcf()
                # plt.pause(2)

            # else:
            #     self.update_node_classes()
            #     animation.FuncAnimation(fig2, self.animate, interval=1, blit=True)
            #     plt.pause(1)
            #     classes = [self.node_classes.get(node) for node in self.grid.nodes()]
            #     print(classes)

            plt.show()

        elif type(self.problem) is TSP:
            if self.first_run is True:
                self.first_run = False
                global fig, ax, neuron_plot
                fig, ax = plt.subplots()

                max_x = max(c.weights[0] for c in self.problem_elements)
                max_y = max(c.weights[1] for c in self.problem_elements)

                ax.plot([c[0] for c in self.problem.coordinates], [c[1] for c in self.problem.coordinates], marker='*', c='gold',
                        markersize=15, linestyle='None')
                ax.set_xlim(0, max_x * 1.05)  # adjust figure axes to max x- and y-values, i.e. 0 and 1 (as they are normalized)
                ax.set_ylim(0, max_y * 1.05)  # use 1.05 to have some margin on the top and right side

                neuron_plot, = ax.plot([n.weights[0] for n in self.output_neurons], [n.weights[1] for n in self.output_neurons],
                               marker='o', markerfacecolor='None', c='green', markersize=10, linestyle=':')
            else:
                start_point = self.output_neurons[0].weights
                neuron_plot.set_data([neuron.weights[0] for neuron in self.output_neurons]+[start_point[0]],
                                     [neuron.weights[1] for neuron in self.output_neurons]+[start_point[1]])

            if has_found_solution:
                # Plot solution route between cities on top of map
                start_point = self.solution_route[0].weights
                global solution_plot
                solution_plot,  = ax.plot([n.weights[0] for n in self.solution_route]+[start_point[0]],
                                          [n.weights[1] for n in self.solution_route]+[start_point[1]],
                        marker='*', c='blue', markersize=12, linestyle='-')

            plt.pause(PLOT_SPEED)

    def run(self):
        while not self.convergence_reached():
            # Sample input vector
            self.set_sample_index(random.randint(0, len(self.problem_elements)-1))
            x_sample = self.problem_elements[self.sample_index]

            # Match
            self.winner_index, _ = self.compute_winning_neuron(self.sample_index, x_sample, 'Training')
            self.update_topologies(self.time_counter)

            # Update
            self.update_weights(self.time_counter)
            self.time_counter += 1

            if type(self.problem) is TSP and PRINTING_MODE and self.time_counter % self.printing_frequency == 0:
                self.plot()

            elif type(self.problem) is MNIST:

                if CLASSIFICATION_MODE and self.time_counter % classification_frequency == 0:
                    # Turn off learning and run each case through the network and record the cases and comp
                    self.training_accuracy = self.do_classification(self.problem_elements, "Training")
                    self.testing_accuracy = self.do_classification(self.testing_elements, "Testing")
                    print()

            if SINGLE_RUN:
                if self.time_counter % 1000 == 0:
                    print(self.time_counter)

        if PRINTING_MODE:
            som.plot(has_found_solution=True)

        if type(self.problem) is MNIST:

            if CLASSIFICATION_MODE:     # do last classification to record results
                # Turn off learning and run each case through the network and record the cases and comp
                self.training_accuracy = self.do_classification(self.problem_elements, "Training")
                self.testing_accuracy = self.do_classification(self.testing_elements, "Testing")
                print()

            res = self.training_accuracy, self.testing_accuracy
            print('\nTraining Score:', res[0])
            print('Testing Score:', res[1])

        else:
            res = self.compute_input_output_distance(), self.compute_total_cost()
            print('\nTotal route cost:', res[1])

        return res

    def do_classification(self, data, data_description):
        # List of boolean variables indicating whether the network guessed right on sample x or not
        correct_values = []
        winners = {}

        # 1 Run each case through the network without learning
        for sample_index, sample_x in enumerate(data):
            winner_index, winner = self.compute_winning_neuron(sample_index, sample_x, data_description)
            winners[sample_x] = winner

        # 2 Update output neuron class labels
        for output_neuron in self.output_neurons:
            output_neuron.set_majority_class()

        # 3 Add indicator indicating whether the guessed target was correct or not
        for sample_index, sample_x in enumerate(data):
            correct_values.append(1 if winners[sample_x].get_majority_class() == sample_x.get_target() else 0)

        # Compute the classification performance
        performance = float(sum(correct_values)) / float(len(correct_values)) * 100

        print(data_description + " Score: " + str(performance) + "%")
        return performance

    def run_more(self, iterations):
        # Increase the iteration cap
        global MAX_ITERATIONS
        MAX_ITERATIONS += iterations
        return self.run()

    # Necessary if we want to pre-train our system on MNIST (as in assignment text)
    def save_state(self):

        # Save all relevant state information to a file
        with open("saved_state.txt", "w") as text_file:
            
            # Row 1: Time counter
            text_file.write(str(self.time_counter)+'\n')

            # Row 2-n: [Majority class, weights]
            for i, elem in enumerate(self.output_neurons):
                text_file.write(str(elem.get_majority_class()) + '\t' + '\t'.join([str(w) for w in elem.get_weights()]))
                text_file.write('\n')
             
    # Necessary if we want to pre-train our system on MNIST (as in assignment text)
    def load_state(self):
        # To do
        # Read from files
        with open("saved_state.txt", "r") as text_file:

            file = text_file.readlines()

            # Timecounter is the first element
            self.time_counter = int(file[0])

            # Retrieving the output neuron data
            content = map(int, file[1:].split('\t'))
            majority_classes = [row[0] for row in content]
            weight_array = [row[1:] for row in content]

            # Initializing output neurons
            self.init_output_neurons(weights=weight_array)

            # Set majority classes equal to the trained majority classes
            for i, neuron in enumerate(self.output_neurons):
                neuron.set_majority_class(majority_class=majority_classes[i])

        
        

# -----------------------------------------


# Abstract class for input neurons
class InputNeuron(object):
    def __init__(self, n_output_neurons):
        self.output_neuron_values = [0 for x in range(n_output_neurons)]   # Todo: Necessary?



class OutputNeuron(object):
    def __init__(self, weights):    # We only initiate OutputNeurons with their weight vector
        self.weights = weights
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

    def get_weights(self):
        return self.weights

    def set_majority_class(self,majority_class=None):
        if majority_class != None:
            self.majority_class = majority_class

        else:
            # Necessary data containers
            classes = {}
            best_class = None
            best_class_count = 0

            # Try catch to eliminate non-target input_vector cases
            try:
                # Fill classes dictionary
                for input_vector in self.get_attached_input_vectors():
                    if input_vector.get_target() in classes.keys():
                        classes[input_vector.get_target()] += 1
                    else:
                        classes[input_vector.get_target()] = 1

                # Find best key max(classes[key])
                for k in classes.keys():
                    if classes[k] > best_class_count:
                        best_class = k
                        best_class_count = classes[k]
            except:
                print("ERROR. Input vector has no feature target_value")

            self.majority_class = best_class

    def get_majority_class(self):
        return self.majority_class


# Generic Problem element
class ProblemElement(object):
    def __init__(self):
        self.output_neuron = None

    def set_output_neuron(self, output_neuron):
        self.output_neuron = output_neuron
    
    def get_output_neuron(self):
        return self.output_neuron

    def get_feature_values(self):
        pass


# Sub-class for TSP-problems
class City(ProblemElement):
    def __init__(self, x, y):
        super(City, self).__init__()
        self.weights = [x, y]

    def get_feature_values(self):
        return self.weights


# Sub-class for problems using images from MNIST
class Image(ProblemElement):
    def __init__(self, pixels, target):
        super(Image, self).__init__()
        self.weights = pixels
        self.target = target[0]

    def get_target(self):
        return self.target

    def get_feature_values(self):
        return self.weights 

# ------------------------------------------


class Problem(object):

    def __init__(self, n_output_neurons):
        self.n_output_neurons = n_output_neurons

    def get_elements(self):
        pass


class TSP(Problem):

    def __init__(self, file_name):
        Problem.__init__(self, None)
        self.data = file_reader(file_name)
        # self.coordinates, self.scale_down_factor = scale_coordinates([[float(row[1]), float(row[2])] for row in self.data])
        self.coordinates = [[float(row[1]), float(row[2])] for row in self.data]    # coordinates are now not scaled
        self.cities = [City(city[0], city[1]) for city in self.coordinates]

    def get_elements(self):
        return self.cities


class MNIST(Problem):

    def __init__(self, n_output_neurons, n_images=1000):
        Problem.__init__(self, n_output_neurons)
        self.n_images = n_images
        self.image_data, self.target_data = load_mnist()
        self.images = self.init_images()
        self.n_output_neurons = n_output_neurons

    def init_images(self):
        image_list = []
        for i, image in enumerate(self.image_data[:self.n_images]):     # use the specified number of images
            try:
                flat_image = flatten_image(image)
            except:
                flat_image = image

            image_list.append(Image(flat_image, self.target_data[i]))

        return image_list

    def get_elements(self):
        return self.images

# ------------------------------------------


def multiple_runs(problem, L_RATE0s, L_RATE_taus, sigma0s, tau_sigmas, FILES):
    print('Number of iterations to do:', len(L_RATE0s)*len(L_RATE_taus)*len(sigma0s)*len(tau_sigmas)*len(FILES))

    iteration_counter = 0
    if RUN_MODE == "TSP":
        with open('results_of_testing.txt', 'a') as file:
            for L_RATE0 in L_RATE0s:
                for L_RATE_tau in L_RATE_taus:
                    for sigma0 in sigma0s:
                        for tau_sigma in tau_sigmas:
                            som = SOM(problem, L_RATE0, L_RATE_tau, printing_frequency, sigma0, tau_sigma)
                            dist, cost = som.run()

                            res = [f, L_RATE0, L_RATE_tau, sigma0, tau_sigma, dist, cost]
                            file.write('\t'.join([str(i) for i in res] + ['\n']))
                            iteration_counter += 1
                            print(iteration_counter)
                            plt.close()

                        file.flush()
    else:
        with open('results_of_MNIST_testing.txt', 'a') as file:
            for L_RATE0 in L_RATE0s:
                for L_RATE_tau in L_RATE_taus:
                    for sigma0 in sigma0s:
                        for tau_sigma in tau_sigmas:
                            som = SOM(problem, L_RATE0, L_RATE_tau, printing_frequency, sigma0, tau_sigma)
                            train_p, test_p = som.run()

                            res = ['MNIST', L_RATE0, L_RATE_tau, sigma0, tau_sigma, train_p, test_p]
                            file.write('\t'.join([str(i) for i in res] + ['\n']))
                            iteration_counter += 1
                            print(iteration_counter)
                            plt.close()

                        file.flush()


# ------------------------------------------

# ****  Parameters ****
fig, ax, neuron_plot, solution_plot = None, None, None, None
PRINTING_MODE = True
PLOT_SPEED = 0.000001
LEGAL_RADIUS = 10
printing_frequency = 500

# ------------------------------------------

# RUN_MODE = "TSP"
RUN_MODE = "MNIST"

SINGLE_RUN = True

L_RATE0 = 0.4
L_RATE_tau = 10000*4
sigma0 = 4
tau_sigma = 10000*1

if RUN_MODE == 'TSP':
    FILE = 1
    MAX_ITERATIONS = 10000*9
else:
    MAX_ITERATIONS = 1000 #10000*2
    N_IMAGES = 2000

classification_frequency = int(MAX_ITERATIONS/5)
CLASSIFICATION_MODE = (RUN_MODE == "MNIST")
SAVE_STATE = True
LOAD_STATE = False

# ------------------------------------------


# ****  MAIN function ****

if __name__ == '__main__':

    start_time = time.time()

    if SINGLE_RUN:
        if RUN_MODE == "TSP":
            # Instantiate TSP
            if USER == "Sverre":
                problem = TSP('./data/' + str("djibouti89") + '.txt')
            else:
                problem = TSP('./data/' + str(FILE) + '.txt')
        elif RUN_MODE == "MNIST":
            problem = MNIST(100, n_images=N_IMAGES)

        # Create and run SOM
        som = SOM(problem, L_RATE0, L_RATE_tau, printing_frequency, sigma0, tau_sigma)
        print('Number of elements in data set:', len(som.problem_elements))
        
        # Load state?
        if LOAD_STATE:
            som.load_state()
        
        som.run()

        if SAVE_STATE:
            som.save_state()

        # Continue?
        more_runs = input("\n\n Run more? If yes, then how many iterations? ")
        while more_runs:
            if type(problem) is TSP and PRINTING_MODE:
                solution_plot.remove()  # remove solution layer from plot until new solution is found

            # Run n more iterations
            print("... Running more iterations ... ")
            som.run_more(int(more_runs))

            # Continue further?
            more_runs = input("\n\n# Run more? ")
    else:
        # Multiple runs to tune parameters

        PRINTING_MODE = False

        if RUN_MODE == "TSP":
            with open('results_of_testing.txt', 'w') as file:
                pass  # empty results file

            FILES = range(6, 8)
            for f in FILES:
                # Instantiate TSP
                problem = TSP('./data/' + str(f) + '.txt')

                L_RATE0s = [0.4 + x * 0.1 for x in range(2, 4)]
                L_RATE_taus = [20000 * x for x in range(3, 5)]
                sigma0s = [x for x in range(2, 15, 3)]
                tau_sigmas = [25000]

                multiple_runs(problem, L_RATE0s, L_RATE_taus, sigma0s, tau_sigmas, FILES)

        else:
            # Instantiate MNIST
            problem = MNIST(100, n_images=N_IMAGES)

            L_RATE0s = [0.4 + x * 0.1 for x in range(1, 3)]
            L_RATE_taus = [10000 * x for x in range(3, 5)]
            sigma0s = [x for x in range(2, 10, 3)]
            tau_sigmas = [10000]

            multiple_runs(problem, L_RATE0s, L_RATE_taus, sigma0s, tau_sigmas, FILES)

    print('Run time:', time.time() - start_time, 'seconds')
