import numpy as np
import time
import matplotlib.pyplot as plt
from collections import OrderedDict

# Remove annoying warnings from matplotlib
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

# file = 'easy-3'
# file = 'medium-1'
file = 'hard-3'
# file = 'expert-2'

SEARCH_MODE = 'bfs'              # Alternatives: 'A*', 'bfs', 'dfs'
DISPLAY_MODE = False
DISPLAY_PROGRESSION = False
DISPLAY_SPEED = 0.2		        # seconds between each update of the visualization
DISPLAY_PROGRESS_SPEED = 0.01   # seconds between each update of the visualization

BOARD_SIZE = (6, 6)
EXIT = (5, 2)                   # position of goal point in puzzle
IMAGE = plt.imshow(np.full(BOARD_SIZE, np.NaN), interpolation='nearest', vmin=1, vmax=13)


# --------------------------------------
# *** GENERAL METHODS ***

class Node(object):
    """A node is here a puzzle bord construction with vehicles with a range of opportunities for further moves. """

    def __init__(self, node_id):
        self.id = node_id

        self.g_cost = 0      # the distance from the root of the search tree to this node
        self.heuristic = None   # an estimate of the distance from the node to a goal state
        self.f_cost = 99999      # = g + h = the total expected cost of a solution path
        self.parent = None
        self.kids = set()

        self.vehicles = get_vehicles(node_id)

    # Necessary to make the Nodes an orderable type
    def __lt__(self, other):    # to let heapq sort nodes with similar f.costs
        return self.f_cost < other.f_cost


class RushHour(object):
    def __init__(self, start_node):
        self.start_node = start_node

        # Initialize empty board
        self.board = np.full(BOARD_SIZE, np.NaN)

        self.opened = dict()            # the actual list of nodes already opened/expanded
        self.closed = dict()            # a list of already closed nodes - does not have to be ordered
        self.nodes = dict()             # a set of all unique nodes generated

    # Finds the node in self.open_nodes that has the lowest f_cost (total cost)
    def find_best_node(self):
        lowest_cost = 999999
        id_of_best_node = ''
        for node_id, node in self.opened.items():
            if node.f_cost < lowest_cost:
                id_of_best_node = node.id
                lowest_cost = node.f_cost
        return id_of_best_node

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
            if parent.g_cost + self.arc_cost(parent, kid) < kid.g_cost:
                print('##############')
                kid.parent = parent
                kid.g_cost = parent.g_cost + self.arc_cost(kid.parent, kid)
                kid.f_cost = kid.g_cost + kid.heuristic
                self.propagate_path_improvements(kid)  # do recursively for all kids of parent node

    def search(self):
        print("Solving puzzle...")
        self.opened[self.start_node.id] = self.start_node

        if SEARCH_MODE == "dfs":
            # In DFS, we always examine the most previous state added to the agenda. Therefore we used an ordered dict
            self.opened = OrderedDict(self.opened)

        # # Initialization lists for tracking search performance
        # best_cost_development = []
        # number_of_open_nodes = []

        while len(self.opened):

            # # Display algorithm progression
            # if len(self.opened) % 100 == 0:
            #     print("NUMBER OF NODES: " + str(len(self.opened)))

            # DFS mode
            if SEARCH_MODE == "dfs":
                current_node = self.opened.popitem()[1]

            # BFS and A* mode:
            # For graphs with unit arcs, BFS is a specific instance of A* where the heuristic function is
            # simply set to zero. # Therefore, BFS and A* are treated equally at this stage. The difference
            # is implemented in the h_estimate function.
            else:
                best_node_id = self.find_best_node()
                current_node = self.opened.pop(best_node_id)

            # Add node to closed list so I don't expand it twice
            self.closed[current_node.id] = current_node

            # # Saves information about search progression
            # best_cost_development.append(current_node.f_cost)
            # number_of_open_nodes.append(len(self.opened))

            if DISPLAY_PROGRESSION:
                self.animate_progress(current_node)

            # If the goal is reached, return and animate the solution path
            if mission_completed(current_node):
                print("\n*****RESULTS*****")
                print('GENERATED NODES:', len(self.nodes))
                print('EXPANDED NODES:', len(self.closed))
                self.animate_solution(current_node)
                return 1

            # Expand (i.e. generate) all adjacent nodes
            adjacent_nodes = self.generate_adj_nodes(current_node)
            for adj_node in adjacent_nodes:
                current_node.kids.add(adj_node)

                # If the adjacent node is neither in the opened nor in the closed list, i.e. not previously discovered
                if adj_node.id not in self.opened and adj_node.id not in self.closed:
                    self.attach_and_evaluate(adj_node, current_node)
                    self.opened[adj_node.id] = adj_node

                # Check if I have found a cheaper path to the adjacent node
                elif current_node.g_cost + self.arc_cost(current_node, adj_node) < adj_node.g_cost:
                    self.attach_and_evaluate(adj_node, current_node)

                    # If node is on the closed-list (meaning it probably has child nodes), then all improvements
                    # need to be passed on to all descendants
                    if adj_node.id in self.closed:
                        print('##############')
                        self.propagate_path_improvements(adj_node)
        return 0, 0

# --------------------------------------
# *** PROBLEM SPECIFIC METHODS ***

    def h_estimate1(self, node):
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

    def h_estimate2(self, node):
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
        node = Node(node_id)                # save ID as large integer instead of string for faster comparison of IDs

        if node.id not in self.nodes:       # if node is not already added to node list
            self.nodes[node.id] = node      # add node_id to the set of all generated nodes
        return node

    def animate_progress(self, node):
        plt.title('Rush Hour PROGRESS simulation')
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
                plt.title('Rush Hour simulation')
                IMAGE.set_data(self.board)
                plt.pause(DISPLAY_SPEED)  # seconds between each update of the visualization
        print('Solution steps:', len(solution_nodes)-1)     # Minus one because it includes the start position


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

def get_vehicles(node_id):
    vehicles = []
    quads = [node_id[i:i + 6] for i in range(0, len(node_id), 6)]
    for quad in quads:
        id = quad[0:2]
        orientation, x, y, size = [quad[i] for i in range(2, 6)]
        vehicles.append(Vehicle(int(id), int(orientation), int(x), int(y), int(size)))
    return vehicles


def mission_completed(node):
    for v in node.vehicles:     # find target vehicle
        if v.id == 0:
            return v.x_end == EXIT[0]


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


# --------------------------------------
# *** MAIN ***

if __name__ == '__main__':
    t_0 = time.time()

    puzzle = RushHour(Node(load_scenario(file)))

    puzzle.search()

    print('\nRun time:', time.time() - t_0, 'seconds')
