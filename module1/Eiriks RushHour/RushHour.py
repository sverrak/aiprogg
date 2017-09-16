import numpy as np
import time
from Node import *
# from Vehicle import *     # Vehicle is already imported in Node.py
import matplotlib.pyplot as plt

# Remove annoying warnings from matplotlib
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# file = 'easy-3'
# file = 'medium-1'
file = 'hard-3'
# file = 'expert-2'

DISPLAY_MODE = False
PRINTING_PROGRESSION = False
DISPLAY_SPEED = 0.2		  # seconds between each update of the visualization
DISPLAY_PROGRESS_SPEED = 0.005		  # seconds between each update of the visualization

BOARD_SIZE = (6, 6)
EXIT = (5, 2)          # position of goal point in puzzle
IMAGE = plt.imshow(np.full(BOARD_SIZE, np.NaN), interpolation='nearest', vmin=1, vmax=14)


# --------------------------------- #

class RushHour(object):
    def __init__(self, start_node):
        self.start_node = start_node

        # Initialize empty board
        self.board = np.full(BOARD_SIZE, np.NaN)

        self.opened = dict()            # the actual list of nodes already opened/expanded
        self.closed = dict()            # a list of already closed nodes - does not have to be ordered
        self.nodes = set()              # a set of all unique nodes generated

    def h_estimate1(self, node):
        """ Heuristic 1: distance to goal position (along x-axis) + number of blocking cars. """
        # select the target vehicle among the vehicle list
        target_vehicle = None
        for v in node.vehicles:
            if v.id == 0:
                target_vehicle = v

        # the target vehicle is always horizontal and in correct y-position
        dist = EXIT[0] - target_vehicle.x_end

        # find number of blocking cars along path to goal
        for x in range(target_vehicle.x_end + 1, EXIT[0] + 1):
            position_status = self.board[EXIT[1]][x]
            if position_status is not np.NaN:    # if parking spot not empty => a car is blocking
                dist += 1
        return dist

    def h_estimate2(self, node):
        """ Heuristic 2: distance to goal position + number of blocking cars + blocking cars that are block """
        # select the target vehicle among the vehicle list
        target_vehicle = None
        for v in node.vehicles:
            if v.id == 0:
                target_vehicle = v

        # the target vehicle is always horizontal and in correct y-position
        dist = EXIT[0] - target_vehicle.x_end

        self.get_puzzle(node)

        # find number of blocking cars along path to goal
        blocking_cars = set()       # which cars block the road. I use a set to get a unique list
        for x in range(target_vehicle.x_end + 1, EXIT[0] + 1):
            position_status = self.board[EXIT[1]][x]
            if position_status is not np.NaN:    # if parking spot not empty => a car is blocking
                dist += 1
                blocking_cars.add(position_status)

        # find number of cars blocking those cars that block our target car
        for v in node.vehicles:
            if v.id in blocking_cars:
                if v.orientation == 0:  # horizontally parked vehicle
                    if (v.x_start == 0 or (v.x_start > 0 and not np.isnan(self.board[v.y_start][v.x_start - 1]))) and \
                            (v.x_end == 5 or (v.x_end < 5 and not np.isnan(self.board[v.y_start][v.x_end + 1]))):
                            dist += 1
                else:   # vertically parked vehicle
                    if (v.y_start == 0 or (v.y_start > 0 and not np.isnan(self.board[v.y_start][v.y_start - 1]))) and \
                            (v.y_end == 5 or (v.y_end < 5 and not np.isnan(self.board[v.y_start][v.y_end + 1]))):
                            dist += 1
        return dist

    def get_puzzle(self, node):
        """ Set the representation of the Rush Hour-board as a 2D numpy array of floating points. """
        # reset board before inserting the updated list of vehicles
        self.board = np.full(BOARD_SIZE, np.NaN)

        # insert vehicles onto board
        for v in node.vehicles:
            for y in range(v.y_start, v.y_end+1):
                for x in range(v.x_start, v.x_end+1):
                    self.board[y][x] = v.id

    @staticmethod
    def mission_completed(node):
        for v in node.vehicles:     # find target vehicle
            if v.id == 0:
                return v.x_end == EXIT[0]

    def generate_adj_nodes(self, node):
        self.get_puzzle(node)
        adj_nodes = []
        for v in node.vehicles:
            if v.orientation == 0:   # 0 means vehicle is horizontally parked
                if v.x_start > 0 and np.isnan(self.board[v.y_start][v.x_start - 1]):
                    new_v = node.vehicles.copy()
                    new_v.remove(v)
                    new_v.append(Vehicle(v.id, v.orientation, v.x_start - 1, v.y_start, v.size))
                    adj_nodes.append(self.create_node(new_v))
                if v.x_end < 5 and np.isnan(self.board[v.y_start][v.x_end + 1]):
                    new_v = node.vehicles.copy()
                    new_v.remove(v)
                    new_v.append(Vehicle(v.id, v.orientation, v.x_start + 1, v.y_start, v.size))
                    adj_nodes.append(self.create_node(new_v))
            else:
                if v.y_start > 0 and np.isnan(self.board[v.y_start - 1][v.x_start]):
                    new_v = node.vehicles.copy()
                    new_v.remove(v)
                    new_v.append(Vehicle(v.id, v.orientation, v.x_start, v.y_start - 1, v.size))
                    adj_nodes.append(self.create_node(new_v))
                if v.y_end < 5 and np.isnan(self.board[v.y_end + 1][v.x_start]):
                    new_v = node.vehicles.copy()
                    new_v.remove(v)
                    new_v.append(Vehicle(v.id, v.orientation, v.x_start, v.y_start + 1, v.size))
                    adj_nodes.append(self.create_node(new_v))
        return adj_nodes

    def create_node(self, vehicles: list):
        node_id = ''
        v_id = ''
        vehicles.sort(key=lambda x: x.id)
        for v in vehicles:
            if v.id > 9:             # all vehicle IDs must be two digits, to allow for =< 99 vehicles in puzzle
                v_id = str(v_id)
            else:
                v_id = '0' + str(v.id)
            node_id = ''.join([node_id, v_id, str(v.orientation), str(v.x_start), str(v.y_start), str(v.size)])
        node = Node(node_id)                # save ID as large integer instead of string for faster comparison of IDs

        if node.id not in self.nodes:       # if node is not already added to node list
            self.nodes.add(node.id)         # add node_id to the set of all generated nodes
        return node

    # Finds the state s in open_nodes that has the lowest total cost
    def find_best_node(self):
        lowest_cost = 9999
        id_of_best_node = ''
        for node_id, node in self.opened.items():
            if node.f_cost < lowest_cost:
                id_of_best_node = node_id
                lowest_cost = node.f_cost
        return id_of_best_node

    def print_solution_path(self, node):
        solution_nodes = []
        while node.id is not self.start_node.id:    # Create list of solution nodes
            solution_nodes.append(node)
            node = node.parent
        solution_nodes.append(self.start_node)
        if DISPLAY_MODE:
            self.animate_solution(solution_nodes)
        print('Solution steps:', len(solution_nodes)-1)     # Minus one because it includes the start position

    def animate_solution(self, solution_nodes):
        for node in reversed(solution_nodes):   # reversed to get from start to end position
            self.get_puzzle(node)               # update node board
            self.board[self.board == 0] = 12    # change vehicle id before drawing to get a very different color
            plt.title('Rush Hour simulation')
            IMAGE.set_data(self.board)
            plt.pause(DISPLAY_SPEED)            # seconds between each update of the visualization

    def attach_and_evaluate(self, adj: Node, node: Node):
        adj.g_cost = node.g_cost + self.arc_cost(node, adj)  # arc_cost = the cost of moving from one node to the next
        adj.f_cost = adj.g_cost + self.h_estimate2(adj)
        adj.heuristic = self.h_estimate2(adj)
        adj.parent = node

    @staticmethod
    # This function is now trivial, but it could e.g. give different moving costs for different vehicles or board areas
    def arc_cost(node, adj_node):
        return 1

    def propagate_path_improvements(self, parent: Node):
        """ Recursively reconstructs best/shortest path to a node. """
        for kid in parent.kids:
            # print(kid.g_cost)
            if parent.g_cost + self.arc_cost(parent, kid) < kid.g_cost:
                print('##############')
                kid.parent = parent
                kid.g_cost = parent.g_cost + self.arc_cost(kid.parent, kid)
                kid.f_cost = kid.g_cost + kid.heuristic
                self.propagate_path_improvements(kid)   # do recursively for all kids of parent node

    # -------------------

    def a_star_search(self):
        self.opened[self.start_node.id] = self.start_node

        while len(self.opened):

            current_node_id = self.find_best_node()
            current_node = self.opened.pop(current_node_id)
            self.closed[current_node.id] = current_node     # Add node to closed list so I don't expand it twice

            # Printing progression
            if PRINTING_PROGRESSION:
                self.get_puzzle(current_node)       # update node board
                self.board[self.board == 0] = 12    # change vehicle id before drawing to get a very different color
                plt.title('Rush Hour PROGRESS simulation')
                IMAGE.set_data(self.board)
                plt.pause(DISPLAY_PROGRESS_SPEED)  # seconds between each update of the visualization

            if self.mission_completed(current_node):        # if the goal is reached, return and display the path
                self.print_solution_path(current_node)
                return len(self.nodes), len(self.closed)

            adjacent_nodes = self.generate_adj_nodes(current_node)   # expand (i.e. generate) all adjacent nodes
            for adj_node in adjacent_nodes:
                current_node.kids.add(adj_node)

                # If the adjacent node is neither in the opened nor in the closed list
                if adj_node.id not in self.opened and adj_node.id not in self.closed:
                    self.attach_and_evaluate(adj_node, current_node)
                    self.opened[adj_node.id] = adj_node

                # Check if I have found a cheaper path to the adjacent node
                elif current_node.g_cost + self.arc_cost(current_node, adj_node) < adj_node.g_cost:
                    self.attach_and_evaluate(adj_node, current_node)

                    # If node is on the closed-list (meaning it probably has child nodes), then all improvements ...
                    # ... need to be passed on to all descendants
                    if adj_node.id in self.closed:
                        print('##############')
                        self.propagate_path_improvements(adj_node)

        return 0, 0


def load_scenario(file_name):
    with open('./data/' + file_name + '.txt', 'r') as scenario_file:
        quads = scenario_file.read()
    node = quads.replace('\n', '').replace(',', '')

    # Add vehicle IDs to the quads and return a node_id
    node = [node[v:v+4] for v in range(0, len(node), 4)]
    node_id = ''
    for v_id, quad in enumerate(node):
        if v_id > 9:    # all vehicle IDs must be two digits, to allow for =< 99 vehicles in puzzle
            v_id = str(v_id)
        else:
            v_id = '0' + str(v_id)
        node_id = str(v_id).join([node_id, quad])
    return node_id     # return ID of start_node


if __name__ == '__main__':
    t_0 = time.time()

    puzzle = RushHour(Node(load_scenario(file)))

    number_of_nodes_generated, number_of_nodes_closed = puzzle.a_star_search()
    print('number_of_nodes_generated:', number_of_nodes_generated)
    print('number_of_nodes_closed:', number_of_nodes_closed)

    print('\nRun time:', time.time() - t_0, 'seconds')
