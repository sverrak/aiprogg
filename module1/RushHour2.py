""" STATE DESCRIPTION
Each state is a list of vehicles of the same format as input format.

STATE TRANSITIONS (MOVES)
Transitions (moves) are represented as a string consisting of two characters:
    - a number equal to the index of the car that is about to be moved
    - a letter indicating the direction of the move (N,S,W,E)

EXTERNAL LIBRARIES
    - Matplotlib for visualizing the data
"""

# **** DECLARATIONS AND PARAMETERS ****
from module1 import AStar2
import numpy as np
import time

LEVEL = "board2.txt"
SEARCH_MODE = 'A*'              # Alternatives: 'A*', 'bfs', 'dfs'
DISPLAY_MODE = True
DISPLAY_PROGRESS = False
PRINTING_MODE = False
DISPLAY_SPEED = 0.1             # seconds between each update of the visualization
DISPLAY_PROGRESS_SPEED = 0.001  # seconds between each update of the visualization

BOARD_SIZE = 6
EXIT_X = 5
EXIT_Y = 2

if DISPLAY_MODE or DISPLAY_PROGRESS:
    import matplotlib.pyplot as plt
    import matplotlib.cbook

    # Remove annoying warning from matplotlib.animation
    import warnings
    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

    IMAGE = plt.imshow(np.zeros((BOARD_SIZE, BOARD_SIZE)), interpolation='nearest', vmin=0, vmax=13)


# **** OVERRIDING MODULE CONSTANTS ****
AStar2.SEARCH_MODE = SEARCH_MODE
AStar2.DISPLAY_MODE = DISPLAY_MODE
AStar2.DISPLAY_PROGRESS = DISPLAY_PROGRESS
AStar2.PRINTING_MODE = PRINTING_MODE
AStar2.DISPLAY_SPEED = DISPLAY_SPEED
AStar2.DISPLAY_PROGRESS_SPEED = DISPLAY_PROGRESS_SPEED
AStar2.BOARD_SIZE = BOARD_SIZE
AStar2.EXIT_X = EXIT_X
AStar2.EXIT_Y = EXIT_Y


def read_board_from_file(input_file):
    with open('./data/' + input_file, 'r') as f:
        raw_board = f.readlines()
        return raw_board


def init_vehicles():
    print("\n************************************\n************************************\nLevel: " + str(LEVEL))
    vehicles_strings = read_board_from_file(LEVEL)
    vehicles_nonintegers = [vehicles_strings[i].split(",") for i in range(len(vehicles_strings))]

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
                board[n][i] = 12  # change vehicle id before drawing to get a very different color
    return board


def animate_solution(moves, state=None):
    plt.title('Rush Hour ** SOLUTION ** simulation')
    vehicles = init_vehicles()
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


def animate_progress(current_state):
    plt.title('Rush Hour PROGRESS simulation')
    board = from_vehicles_to_board(current_state)
    board = adapt_board_for_visualization(board)
    IMAGE.set_data(board)
    plt.pause(DISPLAY_PROGRESS_SPEED)  # seconds between each update of the visualization


# *** Problem dependent ***
# Returns all possible neighbor states and which move that has been done to get there
def generate_successors(current_state, moves=None):
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


# **** OVERRIDING MODULE FUNCTIONS ****
AStar2.is_finished_state = is_finished_state
AStar2.estimate_cost = estimate_cost
AStar2.generate_successors = generate_successors
AStar2.animate_solution = animate_solution
AStar2.animate_progress = animate_progress


if __name__ == '__main__':

    # Initializing the program
    start_time = time.time()
    vehicles = init_vehicles()

    # Run A* algorithm
    best_cost_development, number_of_open_nodes_development = AStar2.astar(vehicles)

    if PRINTING_MODE:
        print("\n\nDevelopment of best cost: " + str(best_cost_development))
        print("Development of number of open nodes: " + str(number_of_open_nodes_development))

    print('Running time:', time.time() - start_time)
