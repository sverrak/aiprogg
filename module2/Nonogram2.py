"""
This program has five sections:
    - Declarations and parameters
    - Setup functions
    - A* helping functions (Problem specific)
    - A* helping functions (Not problem specific)
    - Running functions (main etc)
"""

# **** DECLARATIONS AND PARAMETERS ****
from module1 import AStar2      # super classes for search used in this code
import time
import numpy as np

FILE_NAME = "nono-dog.txt"
SEARCH_MODE = "A*"
DISPLAY_MODE = True
DISPLAY_PROGRESS = True
PRINTING_MODE = False
DISPLAY_PROGRESS_SPEED = 0.000001      # seconds between each update of the visualization
SIZE_X, SIZE_Y = None, None

with open('./data/' + FILE_NAME, 'r') as f:
    file = f.readlines()[0].split(" ")
    SIZE_X, SIZE_Y = ([int(float(file[i])) for i in range(len(file))])

if DISPLAY_MODE or DISPLAY_PROGRESS:
    import matplotlib.pyplot as plt
    import matplotlib.cbook

    # Remove annoying warning from matplotlib.animation
    import warnings
    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

    IMAGE = plt.imshow(np.zeros((SIZE_Y, SIZE_X)), cmap='Greys', interpolation='nearest', vmin=0, vmax=1)


# **** OVERRIDING MODULE CONSTANTS ****
AStar2.SEARCH_MODE = SEARCH_MODE
AStar2.DISPLAY_MODE = DISPLAY_MODE
AStar2.DISPLAY_PROGRESS = DISPLAY_PROGRESS
AStar2.PRINTING_MODE = PRINTING_MODE
AStar2.DISPLAY_PROGRESS_SPEED = DISPLAY_PROGRESS_SPEED


# **** SETUP FUNCTIONS ****

# Returns the size of the nonogram as well as the row and column constraints
def generate_pattern_input_from_file(input_file):
    with open('./data/' + input_file, 'r') as f:
        raw_data = f.readlines()

        # Put the information on the right containers and transform it to lists of lists of integers
        row_patterns_input = [[int(j) for j in i.split(" ")] for i in raw_data[1:1+SIZE_Y]]
        col_patterns_input = [[int(j) for j in i.split(" ")] for i in raw_data[1+SIZE_Y:]]
        return row_patterns_input, col_patterns_input


# Returns the variable domain for a certain pattern (row constraints)
def create_patterns(pattern, size, is_last=False):
    patterns = []
    if len(pattern) == 1:
        for i in range(size - pattern[0] + 1):
            if is_last:
                patterns.append([0]*i + [1]*pattern[0] + [0]*(size - i - pattern[0]))
            else:
                patterns.append([0]*i + [1]*pattern[0] + [0])
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


# ******* A* PROBLEM SPECIFIC HELP FUNCTIONS *********

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
            # print(int(moves[i][0:moves[i].index(',')]))
            if any(x == int(moves[i][0:moves[i].index(',')]) for x in unmodified_rows_and_columns):
                unmodified_rows_and_columns.remove(int(moves[i][0:moves[i].index(',')]))

    index_of_best_row, best_row = find_next_row_or_column(current_state, unmodified_rows_and_columns)
    # print("IOBR: ", index_of_best_row)
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
        modified_current_state[index_of_best_row] = [current_state[index_of_best_row][i]]

        # Compute revised state
        revised_state = revise(modified_current_state, index_of_best_row, current_state[index_of_best_row][i],
                               is_column, number_of_rows)

        if AStar2.contains([current_state], revised_state):
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
    number_of_columns = len(current_state) - number_of_rows

    # If a column is chosen
    if is_column:
        for i in range(number_of_rows):
            modified_state.append([])
            for v in range(len(current_state[i])):
                if current_state[i][v][index_of_best_row - number_of_rows] == domain[number_of_rows - i - 1]:
                    modified_state[i].append(current_state[i][v])

        modified_state += current_state[number_of_rows:]

    # If a row is chosen 
    else:
        modified_state += current_state[:number_of_rows]

        for i in range(number_of_rows, len(current_state)):
            modified_state.append([])
            for v in range(len(current_state[i])):

                if current_state[i][v][number_of_rows - index_of_best_row - 1] == domain[i - number_of_rows]:
                    modified_state[i].append(current_state[i][v])

    return modified_state


# *** Problem dependent ***
# Computing the heuristic cost of a certain state. Mathematically explained:
def estimate_cost(current_state):
    """ max {      1 if count(len(current_state[i]) > 1, i: row indices) > 1           else 0
               +   1 if count(len(current_state[i]) > 1, i: column indices) > 1        else 0,
                   1 if count(len(current_state[i]) > 1, i: row+column indices) >= 1   else 0} """

    if SEARCH_MODE in ["bfs", "dfs"]:
        return 0

    else:  # A*
        number_of_columns = len(current_state[0][0])
        number_of_rows = len(current_state) - number_of_columns
        min_cost = 0

        # If there are more than one row that have row domain size > 0,
        # then there must be at least one column that is needed to reduce all column domains
        temp_counter_vertical = 0
        for i in range(number_of_rows):
            if (len(current_state[i]) > 1):
                temp_counter_vertical += 1
            if (temp_counter_vertical == 2):
                min_cost += 1
                break

        # Vice versa
        temp_counter_horizontal = 0
        for i in range(number_of_rows, number_of_columns + number_of_rows):
            if (len(current_state[i]) > 1):
                temp_counter_horizontal += 1
            if (temp_counter_horizontal == 2):
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

    return max(min_cost, 1 if temp_counter_horizontal + temp_counter_vertical > 1 else 0)


# **** A* HELPING FUNCTIONS (NOT PROBLEM DEPENDENT) ****

def animate_solution(state, moves=None):
    plt.title(' '.join(['Nonogram SOLUTION simulation:', FILE_NAME]))
    solution = []
    for r in range(SIZE_Y):
        solution.append([])
        for i in range(len(state[SIZE_Y - 1 - r][0])):
            solution[r].append(state[SIZE_Y - 1 - r][0][i])
    IMAGE.set_data(solution)
    plt.show()  # stops image from disappearing after the short pause


def animate_progress(state):
    plt.title(' '.join(['Nonogram PROGRESS simulation:', FILE_NAME]))
    solution = []
    for r in range(SIZE_Y):
        solution.append([])
        for i in range(len(state[SIZE_Y - 1 - r][0])):
            solution[r].append(state[SIZE_Y - 1 - r][0][i])
    IMAGE.set_data(solution)
    plt.pause(DISPLAY_PROGRESS_SPEED)  # seconds between each update of the visualization


def state_complexity(state):
    size = 0
    for i in state:
        if len(i) > size:
            size = len(i)
    return size

# **** OVERRIDING MODULE FUNCTIONS ****
AStar2.is_finished_state = is_finished_state
AStar2.estimate_cost = estimate_cost
AStar2.generate_successors = generate_successors
AStar2.animate_solution = animate_solution
AStar2.animate_progress = animate_progress
AStar2.DISPLAY_MODE = DISPLAY_MODE
AStar2.PRINTING_PROGRESSION = DISPLAY_PROGRESS
AStar2.SEARCH_MODE = SEARCH_MODE


# **** RUNNING FUNCTION ****
if __name__ == '__main__':

    print("\n************************************\n************************************")
    start = time.time()

    print("Level: " + str(FILE_NAME))
    print("Algorithm: " + SEARCH_MODE + "\n")

    # Initializing the program
    row_patterns = []
    column_patterns = []
    row_input, col_input = generate_pattern_input_from_file(FILE_NAME)

    for row in row_input:
        row_patterns.append(create_patterns(row, SIZE_X, len(row) == 1))
    for col in col_input:
        column_patterns.append(create_patterns(col, SIZE_Y, len(col) == 1))

    start_state = row_patterns + column_patterns

    print("Level complexity: " + str(state_complexity(start_state)))
    best_cost_development, number_of_open_nodes_development = AStar2.astar(start_state)

    if PRINTING_MODE:
        print("\n\nDevelopment of best cost: " + str(best_cost_development))
        print("Development of number of open nodes: " + str(number_of_open_nodes_development))

    end = time.time()
    print("\nRUNTIME: " + str(end - start))
