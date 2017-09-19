# -*- coding: utf-8 -*-
# *** REPRESENTATION ***
# STATE DESCRIPTION
# Each state is a list of vehicles of the same format as input format.

# STATE TRANSITIONS (MOVES)
# Transitions (moves) are represented as a string consisting of two characters:
# - a number equal to the index of the car that is about to be moved
# - a letter indicating the direction of the move (N,S,W,E)

# EXTERNAL LIBRARIES
# - Matplotlib for visualizing the data

import numpy as np
import string
import time

SEARCH_MODE = 'A*'  # Alternatives: 'A*', 'bfs', 'dfs'
DISPLAY_MODE = True
PRINTING_PROGRESSION = True
PRINTING_MODE = False
LEVEL = "board2.txt"
DISPLAY_SPEED = 0.1  # seconds between each update of the visualization
DISPLAY_PROGRESS_SPEED = 0.01  # seconds between each update of the visualization

BOARD_SIZE = 6
EXIT_X = 5
EXIT_Y = 2

if DISPLAY_MODE or PRINTING_PROGRESSION:
    import matplotlib.pyplot as plt
    import matplotlib.cbook

    # Remove annoying warning from matplotlib.animation
    import warnings

    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

    IMAGE = plt.imshow(np.zeros((BOARD_SIZE, BOARD_SIZE)), interpolation='nearest', vmin=0, vmax=15)


def read_board_from_file(input_file):
    with open('./data/' + input_file, 'r') as f:
        raw_board = f.readlines()
        return raw_board


def init_vehicles():
    print("\n************************************\n************************************\nLevel: " + str(LEVEL))
    vehicles_strings = read_board_from_file(LEVEL)
    vehicles_nonintegers = [vehicles_strings[i].split(",") for i in range(len(vehicles_strings))]

    # TODO: legg til beskrivelse
    for car in vehicles_nonintegers:
        if len(car[-1]) > 1:
            car[-1] = car[-1][0]

    vehicles = [[int(i) for i in car] for car in vehicles_nonintegers]
    return vehicles


# We have used two representations of the states. This method converts the state from one representation to another:
# 1) One equal to the list of vehicles given in the input files
# 2) One equal to the visual representation of the game
def from_vehicles_to_board(vehicles):
    # Initialize new stuff
    board = [[" " for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]
    letters = range(0, len(vehicles))
    # TODO: hva er letters? er det car_ids?

    # Transform car data to readable board
    for car in vehicles:
        if car[0] == 0:  # Horizontal case

            for i in range(car[-1]):
                if (car[1] + i) <= BOARD_SIZE - 1:
                    board[car[2]][car[1] + i] = letters[0]

        elif car[0] == 1:  # Vertical case
            for i in range(car[-1]):
                if (car[2] + i) <= BOARD_SIZE - 1:
                    board[car[2] + i][car[1]] = letters[0]

        # If car is not vertically or horizontally, we have a problem...
        else:
            print("Error")

        # We want every letter to only be assigned to a single vehicle
        letters = letters[1:]

    return board


# Prints a board to the terminal
def print_board(board):
    print('\n   ' + ' '.join([str(i) for i in range(BOARD_SIZE)]))
    print(" ---------------")

    for j in range(BOARD_SIZE):
        temp_string = str(j) + "| "
        for i in range(BOARD_SIZE):
            temp_string += str(board[j][i]) + " "
        if (j == 2):
            temp_string += "| <--- EXIT"
        else:
            temp_string += "|"
        print(temp_string)
    print(" ---------------\n")


def adapt_board_for_visualization(board):
    for n in range(BOARD_SIZE):
        for i in range(BOARD_SIZE):
            if board[n][i] == ' ':
                board[n][i] = np.NaN
            if board[n][i] == 0:
                board[n][i] = 15  # change vehicle id before drawing to get a very different color
    return board


def animate_solution(vehicles, moves):
    plt.title('Rush Hour ** SOLUTION ** simulation')
    solution_nodes = []
    for m in moves:
        vehicles = move(vehicles, m)
        solution_nodes.append(vehicles)

    for node in solution_nodes:
        board = from_vehicles_to_board(node)
        board = adapt_board_for_visualization(board)
        IMAGE.set_data(board)
        plt.pause(DISPLAY_SPEED)


# *** Problem dependent ***
# If the move is legal, do the move
def move(vehicles, move):
    # Interpreting the move
    if (len(move) == 3):
        vehicle = int(move[:-1])
        direction = move[-1]
    elif (len(move) == 2):
        vehicle = int(move[0])
        direction = move[1]
    elif (len(move) == 1):
        print("AN ERROR OCCURED 2")
        return vehicles

    # Checking whether the move is legal or not
    if is_legal_move(from_vehicles_to_board(vehicles), move, vehicles):

        # If it is, do the move
        vehicles_mod = [x[:] for x in vehicles]
        if (direction == "N"):
            vehicles_mod[vehicle][2] -= 1
        elif (direction == "S"):
            vehicles_mod[vehicle][2] += 1
        elif (direction == "W"):
            vehicles_mod[vehicle][1] -= 1
        elif (direction == "E"):
            vehicles_mod[vehicle][1] += 1

        return vehicles_mod
    return vehicles


# The logic of this method is fairly simple. A move is legal if certain characteristics are present:
# - The move must be horizontal or vertical
# - The post-move state must not have out-of-the-board vehicles
# - The post-move state must not have multiple vehicles in a certain board cell
def is_legal_move(board, move, vehicles):
    if (len(move) == 3):
        vehicle = int(move[:-1])
        direction = move[-1]
    elif (len(move) == 2):
        vehicle = int(move[0])
        direction = move[1]
    else:
        print("AN ERRROR OCCURED")
        print(move)
        return False

    # Horizontal case
    if vehicles[vehicle][0] == 0 and (direction == "W" or direction == "E"):
        if (direction == "W"):
            if (vehicles[vehicle][1] > 0):
                return board[vehicles[vehicle][2]][vehicles[vehicle][1] - 1] == " "
            return False
        elif (direction == "E"):
            if (move[0] == "0" and vehicles[0][2] == 2 and vehicles[0][1] >= BOARD_SIZE - 2):  # EXIT
                return True
            elif (vehicles[vehicle][1] < BOARD_SIZE - vehicles[vehicle][3]):
                return board[vehicles[vehicle][2]][vehicles[vehicle][1] + vehicles[vehicle][3]] == " "
            return False
        return False

    # Vertical case
    elif (vehicles[vehicle][0] == 1 and (direction == "N" or direction == "S")):

        if (direction == "N"):
            if (vehicles[vehicle][2] > 0):
                return board[vehicles[vehicle][2] - 1][vehicles[vehicle][1]] == " "
            return False
        elif (direction == "S"):
            if (vehicles[vehicle][2] < BOARD_SIZE - vehicles[vehicle][3]):
                return board[vehicles[vehicle][2] + vehicles[vehicle][3]][vehicles[vehicle][1]] == " "
            return False
        return False
    else:
        return False


# *** Problem dependent ***
# Checks whether a certain state is the target state or not
def is_finished_state(vehicles):
    return vehicles[0][2] == 2 and vehicles[0][1] == BOARD_SIZE - 2  # Car-0 is in exit position


def animate_progression(current_state):
    plt.title('Rush Hour PROGRESS simulation')
    board = from_vehicles_to_board(current_state)
    board = adapt_board_for_visualization(board)
    IMAGE.set_data(board)
    plt.pause(DISPLAY_PROGRESS_SPEED)  # seconds between each update of the visualization


# The A* search function
# Not touched yet. However, we must do something to the node linking etc
def astar(init_node):
    print("Solving puzzle...")

    # Initialization of the node data structures
    closed_nodes = []  # Indices of the nodes that are closed
    node_indices = {0: init_node}  # A dictionary containing all nodes and their respective indices
    closed_states = []  # The closed nodes
    open_nodes = [0]  # Indices of the nodes that are currently being examined or waiting to be examined
    open_states = [init_node]  # The nodes that are currently being examined or waiting to be examined

    # Initialization of data structures describing certain features of each state
    concrete_costs = {0: 0}  # The number of moves needed to reach a specific node
    estimated_costs = {0: estimate_cost(node_indices[0])}  # The estimated number of moves left to reach final state
    total_costs = {0: concrete_costs[0] + estimated_costs[
        0]}  # The sum of the concrete cost and the estimated cost of a certain state
    moves = {0: []}  # A dictionary containing a sequence of moves needed to reach the state indicated by the key

    # Initialization of iteration specific data structures
    best_cost_development = []
    number_of_open_nodes_development = []

    # Agenda loop
    while open_nodes:

        # Display algorithm progression
        if len(node_indices.keys()) % 100 == 0:
            print("NUMBER OF NODES: " + str(len(node_indices.keys())))

        # Update the node lists as the new node is being examined:

        # DFS mode
        if SEARCH_MODE == "dfs":
            # In this search mode, we always examine the most previous state added to the agenda
            index_of_current_state = len(node_indices.keys()) - 1
            lowest_cost = total_costs[index_of_current_state]
            current_state = open_states.pop()

            closed_nodes.append(open_nodes.pop())
            closed_states.append(current_state)
            # TODO: blir ikke helt riktig, for vi trenger ikke egentlig generate successors før vi fortsetter ned en gren

        # BFS and A* mode:
        # For graphs with unit arcs, BFS is a specific instance of A* where the heuristic function is simply set to zero.
        # Therefore, BFS and A* are treated equally at this stage. The difference is implemented in the estimate_cost function
        else:
            index_of_current_state, lowest_cost = find_best_state(open_nodes, total_costs)
            # print("Iteration: " + str(index_of_current_state) + ". Current cost: " + str(lowest_cost))
            current_state = node_indices[index_of_current_state]

            open_nodes.remove(index_of_current_state)
            open_states.remove(current_state)
            closed_nodes.append(index_of_current_state)
            closed_states.append(current_state)

        if PRINTING_PROGRESSION:
            animate_progression(current_state)

        if is_finished_state(current_state):
            print("\n*****RESULTS*****")
            print("GENERATED NODES: " + str(len(node_indices)))
            print("CLOSED/EXAMINED NODES: " + str(len(closed_states)))
            print("SOLUTION STEPS:", len(moves[index_of_current_state]))
            if DISPLAY_MODE:
                animate_solution(current_state, moves[index_of_current_state])
            return best_cost_development, number_of_open_nodes_development

        # Saves information about the state
        best_cost_development.append(lowest_cost)
        number_of_open_nodes_development.append(len(open_states))

        # Generate successors
        successors, how_to_get_to_successors = generate_successors(current_state, moves[index_of_current_state])

        # Explore the successors generated above
        for s in successors:

            # 1. The successor has already been examined (successor in closed_nodes)
            if contains(closed_states, s):
                continue  # Do nothing

            # 2. The successor has already been discovered (successor in agenda)
            elif contains(open_states, s):

                # Check if the state that is already in the agenda has a lower expected cost than that of the newly discovered, identical state
                # Compute the total cost of the newly discovered successor
                total_costs_temp = concrete_costs[index_of_current_state] + 1 + estimate_cost(s)

                # Determine the index of the state identical to the successor
                former_index_of_successor = 0
                for i in range(len(open_nodes)):
                    if open_states[i] == s:
                        former_index_of_successor = i
                        break

                # Check if the cost of the state in the agenda is higher than that of the successor
                if total_costs_temp < total_costs[former_index_of_successor]:
                    # If so, update all features of the state to the features of the successor
                    concrete_costs.update({former_index_of_successor: concrete_costs[index_of_current_state] + 1})
                    estimated_costs.update({former_index_of_successor: estimate_cost(s)})
                    total_costs.update({former_index_of_successor: concrete_costs[former_index_of_successor] +
                                                                   estimated_costs[former_index_of_successor]})
                    path_to_current_successor = [i[:] for i in moves[index_of_current_state]] + [
                        find_index_of(how_to_get_to_successors, s)]
                    moves.update({former_index_of_successor: path_to_current_successor})

            # 3. The successor has not been discovered yet.
            elif not contains(open_states, s):

                # Add the successor to the agenda!
                index_of_current_successor = len(node_indices.keys())
                node_indices.update({index_of_current_successor: s})
                open_nodes.append(index_of_current_successor)
                open_states.append(s)
                concrete_costs.update({index_of_current_successor: concrete_costs[index_of_current_state] + 1})
                estimated_costs.update({index_of_current_successor: estimate_cost(s)})
                total_costs.update({index_of_current_successor: concrete_costs[index_of_current_successor] +
                                                                estimated_costs[index_of_current_successor]})
                # print("tccc: ", total_costs[index_of_current_successor])

                # If the parent is the initial state
                if (index_of_current_state == 0):

                    # The path to the successor will simply be equal to the move from the initial state to the successor
                    moves.update({index_of_current_successor: [find_index_of(how_to_get_to_successors, s)]})

                # For all other parent states
                else:

                    # We append the move from the parent state to the successor to the path of the parent and save that as the path
                    # from the initial state to the successor
                    path_to_current_successor = [i[:] for i in moves[index_of_current_state]] + [
                        find_index_of(how_to_get_to_successors, s)]
                    moves.update({index_of_current_successor: path_to_current_successor})

            # Neither alternative 1., 2. nor 3.
            else:  # Something is wrong...
                print("Error")

    # If the loop does not find any solution
    raise ValueError('Could not find any solution')


# Check if s is contained in open_states
def contains(open_states, s):
    return any(state == s for state in open_states)


# Finds the index of s in how_to_get_to_successors
def find_index_of(how_to_get_to_successors, s):
    for i in (how_to_get_to_successors.keys()):
        if [k[:] for k in how_to_get_to_successors[i]] == [j[:] for j in s]:
            return i


# Finds the state s in open_nodes that has the lowest total cost
def find_best_state(open_nodes, total_costs):
    best = 99999999999999999999999999999
    index_of_best_node = 0

    for node in open_nodes:
        if total_costs[node] < best:
            index_of_best_node = node
            best = total_costs[node]

    return index_of_best_node, best


# *** Problem dependent ***
# Returns all possible neighbor states and which move that has been done to get there
def generate_successors(current_state, moves):
    candidate_moves = ["N", "E", "S", "W"]
    successors = []
    how_to = {}

    # Tests if each potential move is legal for each vehicle
    for i in range(len(current_state)):
        for m in candidate_moves:
            if is_legal_move(from_vehicles_to_board(current_state[:]), str(i) + m, current_state[:]):
                successor = move([k[:] for k in current_state], str(i) + m)
                successors.append(successor)
                how_to[str(i) + m] = successor
    return successors, how_to


# *** Problem dependent ***
# Computing the heuristic cost of a certain state: One step for each (5 - car0.x) and one for each car blocking the exit
def estimate_cost(vehicles):
    if SEARCH_MODE in ["dfs", "bfs"]:
        return 0

    # else, for 'A*':
    board = from_vehicles_to_board(vehicles)
    cost = BOARD_SIZE - vehicles[0][1] - 1
    for i in range(BOARD_SIZE - vehicles[0][1]):
        cost += 1 if board[2][i] not in ["A", " "] else 0
    return cost


if __name__ == '__main__':

    # Initializing the program
    start_time = time.time()
    vehicles = init_vehicles()

    # Solving the puzzle
    # Run A* algorithm
    best_cost_development, number_of_open_nodes_development = astar(vehicles)

    if PRINTING_MODE:
        print("\n\nDevelopment of best cost: " + str(best_cost_development))
        print("Development of number of open nodes: " + str(number_of_open_nodes_development))

    print('Running time:', time.time() - start_time)
