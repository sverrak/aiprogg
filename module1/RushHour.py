from module1.Astar import Node, AStar   # super classes used in this code

import numpy as np
import time
import matplotlib.pyplot as plt

# Remove annoying warnings from matplotlib
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# FILE_NAME = 'easy-3'
# FILE_NAME = 'medium-1'
# FILE_NAME = 'hard-3'
FILE_NAME = 'expert-2'

SEARCH_MODE = 'A*'              # Alternatives: 'A*', 'bfs', 'dfs'
DISPLAY_MODE = True
DISPLAY_PROGRESSION_MODE = False
HEURISTIC = 1                   # Alternatives: [1, 2]
DISPLAY_SPEED = 0.1		        # seconds between each update of the visualization
DISPLAY_PROGRESS_SPEED = 0.01   # seconds between each update of the visualization

BOARD_SIZE = (6, 6)
EXIT = (5, 2)                   # position of goal point in puzzle
IMAGE = plt.imshow(np.full(BOARD_SIZE, np.NaN), interpolation='nearest', vmin=1, vmax=13)


# --------------------------------------
# *** INHERITED CLASSES AND METHODS ***

class RushHourNode(Node):
    """A node is here a puzzle bord construction with vehicles with a range of opportunities for further moves. """

    def __init__(self, node_id, vehicles):
        Node.__init__(self, node_id)
        self.vehicles = vehicles


class RushHourSearch(AStar):
    def __init__(self, start_node):
        AStar.__init__(self, start_node, SEARCH_MODE, DISPLAY_MODE, DISPLAY_PROGRESSION_MODE)

        # Initialize empty board
        self.board = np.full(BOARD_SIZE, np.NaN)


# --------------------------------------
# *** PROBLEM SPECIFIC METHODS ***

    # Use the selected heuristic method
    if HEURISTIC == 1:
        def h_estimate(self, node):
            """ Heuristic 1: distance to goal position (along x-axis) + number of blocking cars. """
            if SEARCH_MODE in ["dfs", "bfs"]:
                return 0

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

    else:
        def h_estimate(self, node):
            """ Heuristic 2: distance to goal position + number of blocking cars + blocking cars that are block """
            if SEARCH_MODE in ["dfs", "bfs"]:
                return 0

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
                        if (v.y_start == 0 or (v.y_start > 0 and not np.isnan(self.board[v.y_start - 1][v.x_start]))) and \
                                (v.y_end == 5 or (v.y_end < 5 and not np.isnan(self.board[v.y_end + 1][v.x_start]))):
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
        vehicles.sort()
        for v in vehicles:
            if v.id > 9:             # all vehicle IDs must be two digits, to allow for =< 99 vehicles in puzzle
                v_id = str(v.id)
            else:
                v_id = '0' + str(v.id)
            node_id += ''.join([v_id, str(v.orientation), str(v.x_start), str(v.y_start), str(v.size)])
        node = RushHourNode(node_id, vehicles)                # save ID as large integer instead of string for faster comparison of IDs

        if node.id not in self.nodes:       # if node is not already added to node list
            self.nodes[node.id] = node      # add node_id to the set of all generated nodes
        return node

    def animate_progress(self, node):
        plt.title(' '.join(['Rush Hour PROGRESS simulation:', FILE_NAME]))
        self.get_puzzle(node)  # update node board
        self.board[self.board == 0] = 12  # change vehicle id before drawing to get a very different color
        IMAGE.set_data(self.board)
        plt.pause(DISPLAY_PROGRESS_SPEED)  # seconds between each update of the visualization

    def animate_solution(self, node):
        solution_nodes = []
        while node.id is not self.start_node.id:    # Create list of solution nodes
            solution_nodes.append(node)
            node = node.parent
        solution_nodes.append(self.start_node)
        if DISPLAY_MODE:
            for node in reversed(solution_nodes):  # reversed to get from start to end position
                self.get_puzzle(node)  # update node board
                self.board[self.board == 0] = 12  # change vehicle id before drawing to get a very different color
                plt.title(' '.join(['Rush Hour SOLUTION simulation:', FILE_NAME]))
                IMAGE.set_data(self.board)
                plt.pause(DISPLAY_SPEED)  # seconds between each update of the visualization
        print('Solution steps:', len(solution_nodes)-1)     # Minus one because it includes the start position

    @staticmethod
    def mission_completed(node):
        for v in node.vehicles:  # find target vehicle
            if v.id == 0:
                return v.x_end == EXIT[0]


class Vehicle(object):

    # Create a new vehicle with its necessary properties
    def __init__(self, id, orientation, x, y, size):
        self.id = id
        self.orientation = orientation  # 0 means horizontal, 1 means vertical
        self.x_start = x
        self.y_start = y
        self.size = size  # length of vehicle

        self.x_end = x + (size - 1) * (1 - orientation)
        self.y_end = y + (size - 1) * orientation

        # Check if any of the vehicles are outside the board
        for v in [self.x_start, self.x_end, self.y_start, self.y_end]:
            if v > 5 or v < 0:
                raise ValueError("All the vehicles need to be within the puzzle board.")

    # Necessary to make the Vehicles an orderable type
    def __lt__(self, other):
        return int(self.id) < int(other.id)


# --------------------------------------
# *** CLASS INDEPENDENT BUT PROBLEM SPECIFIC METHODS ***

# Read input from FILE_NAME and create first node
def load_scenario(FILE_NAME_name):
    with open('./data/' + FILE_NAME_name + '.txt', 'r') as scenario_FILE_NAME:
        quads = scenario_FILE_NAME.read()
    quads = quads.replace('\n', '').replace(',', '')

    # Add vehicle IDs to the quads and return a node_id
    quads = [quads[v:v+4] for v in range(0, len(quads), 4)]
    node_id = ''
    vehicles = []
    for v_id, quad in enumerate(quads):
        orientation, x, y, size = [quad[i] for i in range(4)]
        if v_id > 9:    # all vehicle IDs must be two digits, to allow for =< 99 vehicles in puzzle
            v_id = str(v_id)
        else:
            v_id = '0' + str(v_id)

        vehicles.append(Vehicle(int(v_id), int(orientation), int(x), int(y), int(size)))
        node_id = str(v_id).join([node_id, quad])

    return node_id, vehicles     # return ID of start_node and vehicles


# --------------------------------------
# *** MAIN ***

if __name__ == '__main__':
    t_0 = time.time()

    puzzle = RushHourSearch(RushHourNode(*load_scenario(FILE_NAME)))
    puzzle.search()

    print('\nRun time:', time.time() - t_0, 'seconds')
