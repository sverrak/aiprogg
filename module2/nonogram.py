"""
This program has four sections:
    - Declarations and parameters
    - Setup functions
    - A* helping functions (Problem specific)
    - Running functions (main etc)

"""

### **** DECLARATIONS AND PARAMETERS ****
from module1 import rushhour  # allows for using all A*-functions written for the Rush Hour assignment
import numpy as np
import time

DISPLAY_MODE = True
PRINTING_PROGRESSION = False
DISPLAY_SPEED = 0.3		  # seconds between each update of the visualization
PRINTING_MODE = False

BOARD_SIZE = (10, 10)
IMAGE = None

if DISPLAY_MODE or PRINTING_PROGRESSION:
    import matplotlib.pyplot as plt
    import matplotlib.cbook

    # Remove annoying warning from matplotlib.animation
    import warnings
    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

    IMAGE = plt.imshow(np.full(BOARD_SIZE, np.NaN), cmap='Greys', interpolation='nearest', vmin=0, vmax=1)
    plt.title('Nonogram')


### **** SETUP FUNCTIONS ****

# Returns the size of the nonogram as well as the row and column constraints
def generate_pattern_input_from_file(input_file):
    with open(input_file, 'r') as f:
        raw = f.readlines()

        firstline = raw[0].split(" ")
        sizeX, sizeY = int(firstline[0]), int(firstline[1])

        # Put the information on the right containers
        row_patterns_input = raw[1:1 + sizeX]
        col_patterns_input = raw[1 + sizeX:]

        # Transform the data to list of lists of integers
        # Not working
        row_patterns_input = [[int(float(j)) for j in i.split(" ")] for i in row_patterns_input]
        col_patterns_input = [[int(float(j)) for j in i.split(" ")] for i in col_patterns_input]

        return sizeX, sizeY, row_patterns_input, col_patterns_input


# Returns the variable domain for a certain pattern (row constraints)
def create_patterns(pattern, size, is_last=False, is_length_one=False):
    if (len(pattern) == 1):
        patterns = []
        temp_pattern = []
        for i in range(size - pattern[0] + 1):
            if (is_last == True):
                patterns.append([0 for j in range(0, i)] + [1 for l in range(pattern[0])] + [0 for k in range(
                    size - i - pattern[0])])
                # elif(is_length_one == True):
            # patterns.append([0 for j in range(0,i)] + [1 for l in range(pattern[0])] + [0 for k in range(size-i-pattern[0])])
            else:
                patterns.append([0 for j in range(0, i)] + [1 for l in range(pattern[0])] + [
                    0])  # + [0 for k in range(i+1, size-pattern[0]+1)])
        return patterns
    elif size < sum(pattern) + len(pattern) - 1:
        return [-1]
    else:
        patterns = []
        temp_pattern = []

        prefixes = create_patterns([pattern[0]], size - (len(pattern[1:]) + sum(pattern[1:])))

        for i in range(len(prefixes)):
            if (len(pattern[1:]) == 1):
                tail_patterns = create_patterns(pattern[1:], size - len(prefixes[i]), True)
            else:
                tail_patterns = create_patterns(pattern[1:], size - len(prefixes[i]))
            for j in range(len(tail_patterns)):
                patterns.append(prefixes[i] + tail_patterns[j])

        return patterns


### ******* A* PROBLEM SPECIFIC HELP FUNCTIONS *********

# *** Problem dependent ***
# Checks whether a certain state is the target state or not, that is: the state has row/column domain size = 1 for each row/column
def is_finished_state(current_state):
    for i in current_state:
        if (len(i) != 1):
            return False

    return True


# *** Problem dependent ***
# Returns all possible neighbor states and which move that has been done to get there
def generate_successors(current_state):
    print("\n\nGENERATE SUCCESSORS")

    index_of_best_row, best_row = find_next_row_or_column(current_state)
    number_of_successors = len(current_state[index_of_best_row])
    successors = []
    modified_current_state = current_state[:]
    how_to = []  # How should this be implemented when we have modified A* code?
    number_of_columns = len(current_state[0][0])
    number_of_rows = len(current_state) - number_of_columns
    # print("number of columns " + str(number_of_columns))
    # print(number_of_rows)
    is_column = index_of_best_row >= number_of_rows  # True if the the "best row" is actually a column

    # print("Index of br: " + str(index_of_best_row))
    # Revise all domains of the current state so that invalid values wrt 
    # the newly fixed row are removed
    for i in range(number_of_successors):
        # print("gototo")
        # print([current_state[index_of_best_row][i]])
        # print(modified_current_state[index_of_best_row])
        modified_current_state[index_of_best_row] = []
        modified_current_state[index_of_best_row] = [current_state[index_of_best_row][i]]

        how_to.append("0")  # TO DO
        successors.append(
            revise(modified_current_state, index_of_best_row, current_state[index_of_best_row][i], is_column,
                   number_of_rows))

    return successors, how_to


# Help function for generate_successor
# Could we somehow improve this heuristic? (use entropy gain?) Current idea: smaller domain is better. 
# Determines the variable on which to base the next assumption.
def find_next_row_or_column(domains):
    # Here, the fitness of a candidate row or column is the size of the row/column domain

    index_of_best_row = -1
    fitness_of_best_row = 999999
    for i in range(len(domains)):
        if (len(domains[i]) < fitness_of_best_row):
            index_of_best_row = i
            fitness_of_best_row = len(domains[i])

    if (index_of_best_row == -1):
        print("ERROR OCCURED @ FIND NEXT ROW")

    return index_of_best_row, fitness_of_best_row


# Help function for generate_successors
# Given a reduced domain at index index_of_best_row in current_state, revise returns
# the reduced column/row domains for all other columns/rows. That is, this method
# gives birth to a successor
def revise(current_state, index_of_best_row, domain, is_column, number_of_rows):
    # Idea:
    # For each row/column in current_state do
    #       Remove all values that are not coherent with the examined pattern

    # Internal data structure setup
    modified_state = []
    number_of_columns = len(current_state) - number_of_rows

    # If a column is chosen
    if (is_column):

        for i in range(number_of_rows):
            modified_state.append([])
            for v in range(len(current_state[i])):
                if (current_state[i][v][index_of_best_row - number_of_rows] == domain[number_of_rows - i - 1]):
                    modified_state[i].append(current_state[i][v])

        modified_state += current_state[number_of_rows:]

    # If a row is chosen 
    else:
        modified_state += current_state[:number_of_rows]

        for i in range(number_of_rows, len(current_state)):
            modified_state.append([])
            for v in range(len(current_state[i])):
                if (current_state[i][v][number_of_columns - index_of_best_row - 1] == domain[i - number_of_rows]):
                    modified_state[i].append(current_state[i][v])

    return modified_state


# *** Problem dependent ***
# Computing the heuristic cost of a certain state
# Currently multiplies the sizes of each row/column domain
def estimate_cost(current_state):
    if (SEARCH_MODE == "bfs" or SEARCH_MODE == "dfs"):
        return 0
    elif (SEARCH_MODE == "A*"):
        product = 1
        for i in range(len(current_state)):
            product = product * len(current_state[i])

        if (product == 0):
            return 9999999999999999999
        else:
            return product

            # If A* is chosen, we compute the degree to which each column constraints is violated


def print_nonogram(solution):
    IMAGE.set_data(solution)
    plt.pause(DISPLAY_SPEED)
    plt.show()  # stops image from disappearing after the short pause


# Solving the problem using the A* algorithm
def solve(current_state):
    final_state, moves, best_cost_development_number_of_open_nodes_development = rushhour.astar(current_state)

    if PRINTING_MODE:
        print("\n\nDevelopment of best cost: " + str(best_cost_development))
        print("Development of number of open nodes: " + str(number_of_open_nodes_development))

    if is_finished_state(final_state):
        return final_state

    print("Did not find any solution")
    return "0"


if __name__ == '__main__':
    start = time.time()
    print("\n************************************\n************************************")
    # print("Level: " + str(LEVEL))
    # print("Algorithm: " + MODE + "\n")


    # Initializing the program
    sizeX, sizeY, row_patterns_input, col_patterns_input = generate_pattern_input_from_file("cat.txt")

    row_patterns = []
    column_patterns = []

    for i in range(len(col_patterns_input)):
        column_patterns.append(create_patterns(col_patterns_input[i], sizeY, len(col_patterns_input[i]) == 1,
                                               len(col_patterns_input[i]) == 1))

    for i in range(len(row_patterns_input)):
        row_patterns.append(create_patterns(row_patterns_input[i], sizeX, len(row_patterns_input[i]) == 1,
                                            len(row_patterns_input[i]) == 1))

    current_state = row_patterns + column_patterns

    # successors, how_to = generate_successors(current_state)

    final_state = solve(current_state)

    # print((row_patterns[0]))

    # initial_rows = init_rows(row_patterns_input, sizeX)
    # print(initial_rows)
    # print(estimate_cost(initial_rows, "A*", column_patterns))

    # Displaying run characteristics
    if (PRINTING_MODE == True):
        print("RUNTIME: " + str(end - start) + "\n")

    # Solving the puzzle
    if (False):
        moves, final_rows = solve(initial_rows, SEARCH_MODE)
        board = from_rows_to_board(final_rows)

        if (DISPLAY_MODE == True):
            visualize_development1(current_state, moves)
        else:
            print("\n\n\n***LEVEL SOLVED***")
            print_board(board)

    print('\nRunning time:', time.time() - start)

    # for testing purposes
    solution = [[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                              [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                              [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                              [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                              [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                              [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                              [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], ]
    print_nonogram(solution)

