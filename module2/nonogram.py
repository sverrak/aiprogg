# -*- coding: utf-8 -*-

""" 
This program has five sections:
    - Declarations and parameters
    - Setup functions
    - A* helping functions (Problem specific)
    - A* helping functions (Not problem specific)
    - Running functions (main etc)
"""

# --------------------------------------
# **** DECLARATIONS AND PARAMETERS ****
from module1.Astar import Node, AStar   # super classes used in this code
import time
import numpy as np

FILE_NAME = "elephant.txt"
DISPLAY_MODE = False
PRINTING_MODE = True
PRINTING_PROGRESSION = False    #TODO: implementer dette!
SEARCH_MODE = "A*"
DISPLAY_PROGRESS_SPEED = 0.1
DISPLAY_SPEED = 0.3  # seconds between each update of the visualization
BOARD_SIZE = None

with open('./data/' + FILE_NAME, 'r') as f:
    file = f.readlines()[0].split(" ")
    BOARD_SIZE = ([int(float(file[i])) for i in range(len(file))])

if DISPLAY_MODE or PRINTING_PROGRESSION:
    import matplotlib.pyplot as plt
    import matplotlib.cbook

    # Remove annoying warning from matplotlib.animation
    import warnings
    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

    IMAGE = plt.imshow(np.full(BOARD_SIZE, np.NaN), cmap='Greys', interpolation='nearest', vmin=0, vmax=1)
    plt.title(' '.join(['Nonogram:', FILE_NAME]))


# --------------------------------------
# **** SETUP FUNCTIONS ****

class Nonogram(object):
    def __init__(self):
        RushHour.__init__(object)


# Returns the size of the nonogram as well as the row and column constraints
def generate_pattern_input_from_file(input_file):
    with open('./data/' + input_file, 'r') as f:
        raw = f.readlines()
        BOARD_SIZE = (int(raw[0].split(" ")[0]), int(raw[0].split(" ")[1]))

        # Put the information on the right containers
        row_patterns_input = raw[1:1 + BOARD_SIZE[0]]
        col_patterns_input = raw[1 + BOARD_SIZE[1]:]

        print('####', row_patterns_input)
        # Transform the data to list of lists of integers
        row_patterns_input = [[int(j) for j in i.split(" ")] for i in row_patterns_input]
        col_patterns_input = [[int(j) for j in i.split(" ")] for i in col_patterns_input]

        return row_patterns_input, col_patterns_input


# Returns the variable domain for a certain pattern (row constraints)
def create_patterns(pattern, size, is_last=False, is_length_one=False):
    patterns = []
    if len(pattern) == 1:
        for i in range(size - pattern[0] + 1):
            if is_last == True:
                patterns.append([0 for j in range(0, i)] + [1 for l in range(pattern[0])] + [0 for k in range(
                    size - i - pattern[0])])
            else:
                patterns.append([0 for j in range(0, i)] + [1 for l in range(pattern[0])] + [
                    0])  # + [0 for k in range(i+1, size-pattern[0]+1)])
        return patterns
    elif size < sum(pattern) + len(pattern) - 1:
        return [-1]
    else:
        prefixes = create_patterns([pattern[0]], size - (len(pattern[1:]) + sum(pattern[1:])))

        for i in range(len(prefixes)):
            if len(pattern[1:]) == 1:
                tail_patterns = create_patterns(pattern[1:], size - len(prefixes[i]), True)
            else:
                tail_patterns = create_patterns(pattern[1:], size - len(prefixes[i]))
            for j in range(len(tail_patterns)):
                patterns.append(prefixes[i] + tail_patterns[j])

    return patterns


# --------------------------------------
# *** PROBLEM SPECIFIC FUNCTIONS ***

# *** Problem dependent ***
# Checks whether a certain state is the target state or not, that is: the state has row/column domain size = 1 for each row/column
def is_finished_state(current_state):
    for i in current_state:
        if (len(i) != 1):
            return False
    return True


# *** Problem dependent ***
# Returns all possible neighbor states and which move that has been done to get there
def generate_successors(current_state, moves):

    # Setup of data structures
    unmodified_rows_and_columns = [i for i in range(len(current_state))]

    if len(moves) > 0:
        for i in range(len(moves)):
            if any(x == int(moves[i][0:moves[i].index(',')]) for x in unmodified_rows_and_columns):
                unmodified_rows_and_columns.remove(int(moves[i][0:moves[i].index(',')]))

    index_of_best_row, best_row = find_next_row_or_column(current_state, unmodified_rows_and_columns)
    number_of_successors = len(current_state[index_of_best_row])
    successors = []
    modified_current_state = current_state[:]
    how_to = {}
    number_of_columns = len(current_state[0][0])
    number_of_rows = len(current_state) - number_of_columns
    is_column = index_of_best_row >= number_of_rows  # True if the the "best row" is actually a column

    # Revise all domains of the current state so that invalid values wrt
    # the newly fixed row are removed
    for i in range(number_of_successors):
        # Loop setup
        modified_current_state[index_of_best_row] = []
        modified_current_state[index_of_best_row] = [current_state[index_of_best_row][i]]

        # Compute revised state
        revised_state = revise(modified_current_state, index_of_best_row, current_state[index_of_best_row][i],
                               is_column, number_of_rows)

        if RushHour.contains([current_state], revised_state):
            successors, how_to = generate_successors(current_state,
                     moves + [str(index_of_best_row) + ", forcing not to use this row"])
            break

        # Check if the revised state is valid
        number_of_invalid_rows_or_columns = 0
        maxx = 0
        for j in range(number_of_rows + number_of_columns):
            if (len(revised_state[j]) == 0):
                number_of_invalid_rows_or_columns = 1
                break
            elif (len(revised_state[j]) > maxx):
                maxx = len(revised_state[j])
        # print("maxx", maxx)


        # If the revised state is valid, add it to the list of successors
        if (number_of_invalid_rows_or_columns == 0):
            how_to[str(index_of_best_row) + ", " + str(current_state[index_of_best_row][i])] = revised_state
            successors.append(revised_state)

    return successors, how_to


# Help function for generate_successor
# TODO: Could we somehow improve this heuristic? (use entropy gain?) Current idea: smaller domain is better.
# TODO: Potential improvement:
    # TODO: Legalise choosing domains of length 1. This requires restructuring the code so that current state involves

# Determines the variable on which to base the next assumption.

def find_next_row_or_column(domains, unmodified_rows_and_columns):
    # Here, the fitness of a candidate row or column is the size of the row/column domain
    index_of_best_row = -1
    fitness_of_best_row = 9999999999999999999
    for i in range(len(domains)):
        if (len(domains[i]) < fitness_of_best_row):
            if (len(domains[i]) > 1 or i in unmodified_rows_and_columns):
                # print("len domains", len(domains[i]))
                index_of_best_row = i
                fitness_of_best_row = len(domains[i])

    if index_of_best_row == -1:
        print("ERROR OCCURED @ FIND NEXT ROW")

    return index_of_best_row, fitness_of_best_row


# Help function for generate_successors
# Given a reduced domain at index index_of_best_row in current_state, revise returns
# the reduced column/row domains for all other columns/rows. That is, this method
# gives birth to a successor
def revise(current_state, index_of_best_row, domain, is_column, number_of_rows):
    """ Idea:
    For each row/column in current_state do
           Remove all values that are not coherent with the examined pattern """

    # Internal data structure setup
    modified_state = []
    # number_of_columns = len(current_state) - number_of_rows

    # If a column is chosen
    if is_column:
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

                if current_state[i][v][number_of_rows - index_of_best_row - 1] == domain[i - number_of_rows]:
                    # print(str(i) + "...." + str(v))
                    # print(number_of_columns - index_of_best_row - 1)
                    modified_state[i].append(current_state[i][v])

    return modified_state


# *** Problem dependent ***
# Computing the heuristic cost of a certain state
# Currently multiplies the sizes of each row/column domain
def estimate_cost(current_state):
    if (SEARCH_MODE == "bfs" or SEARCH_MODE == "dfs"):
        return 0

    elif (SEARCH_MODE == "A*"):
        number_of_columns = len(current_state[0][0])
        number_of_rows = len(current_state) - number_of_columns
        min_cost = 0

        # If there are more than one row that have row domain size > 0,
        # then there must be at least one column that is needed to reduce all column domains
        temp_counter = 0
        for i in range(number_of_rows):
            if (len(current_state[i]) > 1):
                temp_counter += 1
            if (temp_counter == 2):
                min_cost += 1
                break

        # Vice versa
        temp_counter = 0
        for i in range(number_of_rows, number_of_columns + number_of_rows):
            if (len(current_state[i]) > 1):
                temp_counter += 1
            if (temp_counter == 2):
                min_cost += 1
                break

        # If there are at least two row values that are equal to 0 and two that are equal to 1,
        # there must be yet another column that must be revised in order to decide which value that is correct
        # However, these tests are currently too slow.

        # Current best guess on potential improvements:
        """
        for rc in current_state[:number_of_rows]:
            breaker = False
            if(len(rc)>0):
                for i in range(len(rc[0])):
                    count_positives = [0 for i in rc[0]]
                    count_negatives = [0 for i in rc[0]]
                    for rd in rc:
                        if (rd[i]==1):
                            count_positives[i] += 1
                        else:
                            count_negatives[i] += 1
                    if ((all(pos >= 2 for pos in count_positives)) and all(neg >= 2 for neg in count_negatives)):
                        min_cost += 1
                        breaker = True
                        break
            if(breaker == True):
                break

        # Vice versa
        for rc in current_state[number_of_rows:]:
            breaker = False
            if(len(rc)>0):
                for i in range(len(rc[0])):
                    count_positives = [0 for i in rc[0]]
                    count_negatives = [0 for i in rc[0]]
                    for rd in rc:
                        if (rd[i]==1):
                            count_positives[i] += 1
                        else:
                            count_negatives[i] += 1
                    if ((all(pos >= 2 for pos in count_positives)) or all(neg >= 2 for neg in count_negatives)):
                        min_cost += 1
                        breaker = True
                        break
            if(breaker == True):
                break
        """

        return min_cost

    # Old method that is not correct
    """product = math.exp(1)
    for i in range(len(current_state)):
        product = product * math.log(len(current_state[i])+1)

    if (product == 0):
        return 9999999999999999999999999999999999999999999999999999999999
    else:
        return product
        return math.log(product+1)
        """

        # If A* is chosen, we compute the degree to which each column constraints is violated


# --------------------------------------
# **** HELPING FUNCTIONS (NOT PROBLEM SPECIFIC) ****

def print_nonogram(solution):
    IMAGE.set_data(solution)
    plt.pause(DISPLAY_SPEED)
    plt.show()  # stops image from disappearing after the short pause


def print_progression(current_state):
    import matplotlib.pyplot as plt
    plt.title('Nonogram PROGRESS simulation')
    state_sample = []
    for i in current_state:
        state_sample.append(i[0])

    print(state_sample)
    IMAGE.set_data(state_sample)
    plt.pause(DISPLAY_PROGRESS_SPEED)  # seconds between each update of the visualization


# --------------------------------------
# **** RUNNING FUNCTIONS ****

# Solving the problem using the A* algorithm
def solve(current_state):
    final_state, moves, best_cost_development, number_of_open_nodes_development = RushHour.RushHour.search()

    if PRINTING_MODE:
        print("\n\nDevelopment of best cost: " + str(best_cost_development))
        print("Development of number of open nodes: " + str(number_of_open_nodes_development))

    if is_finished_state(final_state):
        return final_state

    print("Did not find any solution")
    return "0"

# --------------------------------------
# **** OVERRIDING MODULE FUNCTIONS ****

RushHour.h_estimate = estimate_cost
RushHour.generate_successors = generate_successors


if __name__ == '__main__':
    print("\n************************************\n************************************")

    print("Level: " + str(FILE_NAME))
    print("Algorithm: " + SEARCH_MODE + "\n")

    # Initializing the program
    start = time.time()
    row_patterns_input, col_patterns_input = generate_pattern_input_from_file(FILE_NAME)

    row_patterns = []
    column_patterns = []

    for i in range(len(col_patterns_input)):
        column_patterns.append(create_patterns(col_patterns_input[i], BOARD_SIZE[1], len(col_patterns_input[i]) == 1,
                                               len(col_patterns_input[i]) == 1))

    for i in range(len(row_patterns_input)):
        row_patterns.append(create_patterns(row_patterns_input[i], BOARD_SIZE[0], len(row_patterns_input[i]) == 1,
                                            len(row_patterns_input[i]) == 1))

    current_state = row_patterns + column_patterns

    size = 0
    for i in current_state:
        if len(i) > size:
            size = len(i)
    print("Level complexity: " + str(size))
    # print(current_state)

    # for i in current_state:
    # print(i)

    # successors, how_to = generate_successors(current_state)

    """
    print("Number of successors: " + str(len(successors)))
    for i in successors:
        print("\nNew Successor")
        for j in i:
            print(j)

        print(estimate_cost(i))"""

    final_state = solve(current_state)
    # print(final_state)

    # print((row_patterns[0]))
    end = time.time()
    print("\nRUNTIME: " + str(end - start) + "\n")

    if DISPLAY_MODE:
        solution = []
        for r in range(BOARD_SIZE[1]):
            solution.append([])
            for i in range(len(final_state[BOARD_SIZE[1] - 1 - r][0])):
                solution[r].append(final_state[BOARD_SIZE[1] - 1 - r][0][i])
        print_nonogram(solution)

    else:
        for r in range(BOARD_SIZE[0]):
            temp = ""
            for i in range(len(final_state[BOARD_SIZE[1] - 1 - r][0])):
                temp += str(final_state[BOARD_SIZE[1] - 1 - r][0][i]) + " "
            print(temp)



