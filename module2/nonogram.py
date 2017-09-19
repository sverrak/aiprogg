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
DISPLAY_PROGRESSION_MODE = False    #TODO: implementer dette!
SEARCH_MODE = "A*"
DISPLAY_PROGRESS_SPEED = 0.1
DISPLAY_SPEED = 0.3  # seconds between each update of the visualization
BOARD_SIZE = None

with open('./data/' + FILE_NAME, 'r') as f:
    file = f.readlines()[0].split(" ")
    BOARD_SIZE = ([int(float(file[i])) for i in range(len(file))])

if DISPLAY_MODE or DISPLAY_PROGRESSION_MODE:
    import matplotlib.pyplot as plt
    import matplotlib.cbook

    # Remove annoying warning from matplotlib.animation
    import warnings
    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

    IMAGE = plt.imshow(np.full(BOARD_SIZE, np.NaN), cmap='Greys', interpolation='nearest', vmin=0, vmax=1)
    plt.title(' '.join(['Nonogram:', FILE_NAME]))


# --------------------------------------
# *** INHERITED CLASSES AND METHODS ***

class NonogramNode(Node):
    """A node is here a nonogram board made of lists with either 1- or 0-value. """

    def __init__(self, state, size):
        Node.__init__(self, node_id=None)
        self.state = state
        # self.row_patterns = create_patterns(row_in, size)
        # self.col_patterns = create_patterns(col_in, size)


class NonogramSearch(AStar):
    def __init__(self, starting_node):
        AStar.__init__(self, starting_node, SEARCH_MODE, DISPLAY_MODE, DISPLAY_PROGRESSION_MODE)

        # Initialize empty board
        self.board = np.zeros(BOARD_SIZE)


# --------------------------------------
# *** PROBLEM SPECIFIC METHODS ***

    def create_node(self, elements):
        # TODO: implement
        node_id = ''
        elements.sort()
        for v in elements:
            if v.id > 9:             # all vehicle IDs must be two digits, to allow for =< 99 elements in puzzle
                v_id = str(v.id)
            else:
                v_id = '0' + str(v.id)
            node_id += ''.join([v_id, str(v.orientation), str(v.x_start), str(v.y_start), str(v.size)])
        node = NonogramNode(node_id, elements)                # save ID as large integer instead of string for faster comparison of IDs

        if node.id not in self.nodes:       # if node is not already added to node list
            self.nodes[node.id] = node      # add node_id to the set of all generated nodes
        return node

    @staticmethod
    # Checks whether a certain state is the target state or not, that is: the state has row/column domain size = 1 for each row/column
    def mission_completed(node):
        for i in node.state:
            if len(i) != 1:
                return False
        return True

    def get_puzzle(self, node):
        """ Set the representation of the Nonogram-board as a 2D numpy array of 0s, or 1s where probable. """
        # reset board before inserting the updated list of vehicles
        self.board = np.zeros(BOARD_SIZE)

        # create board of node domains
        for r in range(BOARD_SIZE[1]):
            for i in range(len(node.state[BOARD_SIZE[0] - 1 - r][0])):
                self.board[r] = node.state[BOARD_SIZE[1] - 1 - r][0][i]

    # Returns all possible neighbor states and which move that has been done to get there
    def generate_adj_nodes(self, node):

        # Setup of data structures
        unmodified_rows_and_columns = [i for i in range(len(node.state))]

        if node.parent is not None:
            for i in range(len(moves)):
                if any(x == int(moves[i][0:moves[i].index(',')]) for x in unmodified_rows_and_columns):
                    unmodified_rows_and_columns.remove(int(moves[i][0:moves[i].index(',')]))

        index_of_best_row, best_row = self.find_next_row_or_column(node.state, unmodified_rows_and_columns)
        number_of_successors = len(node.state[index_of_best_row])

        successors = []
        modified_current_state = node.state[:]

        number_of_columns = len(node.state[0][0])
        number_of_rows = len(node.state) - number_of_columns
        is_column = index_of_best_row >= number_of_rows  # True if the the "best row" is actually a column

        # Revise all domains of the current state so that invalid values wrt
        # the newly fixed row are removed
        for i in range(number_of_successors):
            # Loop setup
            modified_current_state[index_of_best_row] = [node.state[index_of_best_row][i]]

            # Compute revised node
            revised_node = self.revise(modified_current_state, index_of_best_row, node.state[index_of_best_row][i],
                                   is_column, number_of_rows)

            if any(node == revised_node for node in [node.state]):
                successors, how_to = self.generate_adj_nodes(node.state)
                break

            # Check if the revised state is valid
            number_of_invalid_rows_or_columns = 0
            maxx = 0
            for j in range(number_of_rows + number_of_columns):
                if (len(revised_node[j]) == 0):
                    number_of_invalid_rows_or_columns = 1
                    break
                elif (len(revised_node[j]) > maxx):
                    maxx = len(revised_node[j])
            # print("maxx", maxx)


            # If the revised state is valid, add it to the list of successors
            if (number_of_invalid_rows_or_columns == 0):
                how_to[str(index_of_best_row) + ", " + str(node.state[index_of_best_row][i])] = revised_node
                successors.append(revised_node)

        return successors, how_to

    # Determines the variable on which to base the next assumption.
    def find_next_row_or_column(self, domains, unmodified_rows_and_columns):
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

    def revise(self, current_state, index_of_best_row, domain, is_column, number_of_rows):
        """ Help function for generating_successors. Given a reduced domain at index index_of_best_row in current_state,
        revise returns the reduced column/row domains for all other columns/rows. That is, this method
        gives birth to a successor. Idea:
        For each row/column in current_state do
               Remove all values that are not coherent with the examined pattern """

        # Internal data structure setup
        modified_node = []

        # If a column is chosen
        if is_column:
            for i in range(number_of_rows):
                modified_node.append([])
                for v in range(len(current_state[i])):
                    if current_state[i][v][index_of_best_row - number_of_rows] == domain[number_of_rows - i - 1]:
                        modified_node[i].append(current_state[i][v])

            modified_node += current_state[number_of_rows:]

        # If a row is chosen
        else:
            modified_node += current_state[:number_of_rows]

            for i in range(number_of_rows, len(current_state)):
                modified_node.append([])
                for v in range(len(current_state[i])):

                    if current_state[i][v][number_of_rows - index_of_best_row - 1] == domain[i - number_of_rows]:
                        # print(str(i) + "...." + str(v))
                        # print(number_of_columns - index_of_best_row - 1)
                        modified_node[i].append(current_state[i][v])

        return modified_node

    # Computing the heuristic cost of a certain state. Currently multiplies the sizes of each row/column domain
    def h_estimate(self, current_state):
        # Help function for generate_successor
        # TODO: Could we somehow improve this heuristic? (use entropy gain?) Current idea: smaller domain is better.
        # TODO: Potential improvement:
        # TODO: Legalise choosing domains of length 1. This requires restructuring the code so that current state involves

        if SEARCH_MODE in ["bfs", "dfs"]:
            return 0

        elif SEARCH_MODE == "A*":
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

    def animate_solution(self, node):
        plt.title('Nonogram SOLUTION simulation')
        solution = []
        for r in range(BOARD_SIZE[1]):
            solution.append([])
            for i in range(len(node.state[BOARD_SIZE[1] - 1 - r][0])):
                solution[r].append(node.state[BOARD_SIZE[1] - 1 - r][0][i])
        IMAGE.set_data(solution)
        plt.pause(DISPLAY_SPEED)
        plt.show()  # stops image from disappearing after the short pause

    def animate_progress(self, node):
        plt.title('Nonogram PROGRESS simulation')
        solution = []
        for r in range(BOARD_SIZE[1]):
            solution.append([])
            for i in range(len(node.state[BOARD_SIZE[1] - 1 - r][0])):
                solution[r].append(node.state[BOARD_SIZE[1] - 1 - r][0][i])
        IMAGE.set_data(solution)
        plt.pause(DISPLAY_PROGRESS_SPEED)  # seconds between each update of the visualization


# ------------------------------------------------------
# *** CLASS INDEPENDENT BUT PROBLEM SPECIFIC METHODS ***

# Returns the row and column constraints of the nonogram as they are inputted
def load_pattern_input_from_file(input_file):
    with open('./data/' + input_file, 'r') as f:
        raw_data = f.readlines()
        BOARD_SIZE = (int(raw_data[0].split(" ")[0]), int(raw_data[0].split(" ")[1]))

        # Put the information on the right containers and transform it to lists of lists of integers
        row_input = [[int(j) for j in i.split(" ")] for i in raw_data[1:1 + BOARD_SIZE[0]]]
        col_input = [[int(j) for j in i.split(" ")] for i in raw_data[1 + BOARD_SIZE[1]:]]

        return row_input, col_input


# Returns the variable domain for a certain pattern (row constraints)
def create_patterns(input_pattern, size, is_last=False, is_length_one=False):
    patterns = []
    if len(input_pattern) == 1:
        for i in range(size - input_pattern[0] + 1):
            if is_last:
                patterns.append([0 for j in range(0, i)] + [1 for l in range(input_pattern[0])] + [0 for k in range(
                    size - i - input_pattern[0])])
            else:
                patterns.append([0 for j in range(0, i)] + [1 for l in range(input_pattern[0])] + [0])  # + [0 for k in range(i+1, size-pattern[0]+1)])
        return patterns
    elif size < sum(input_pattern) + len(input_pattern) - 1:
        return [-1]
    else:
        prefixes = create_patterns([input_pattern[0]], size - (len(input_pattern[1:]) + sum(input_pattern[1:])))

        for i in range(len(prefixes)):
            if len(input_pattern[1:]) == 1:
                tail_patterns = create_patterns(input_pattern[1:], size - len(prefixes[i]), True)
            else:
                tail_patterns = create_patterns(input_pattern[1:], size - len(prefixes[i]))
            for j in range(len(tail_patterns)):
                patterns.append(prefixes[i] + tail_patterns[j])
    return patterns


def state_complexity(state):
    size = 0
    for i in state:
        if len(i) > size:
            size = len(i)
    return size

# --------------------------------------
# *** MAIN ***

if __name__ == '__main__':
    print("\n************************************\n************************************")
    start = time.time()
    print("Level: " + str(FILE_NAME))
    print("Algorithm: " + SEARCH_MODE + "\n")

    # Initializing the program
    row_patterns = []
    column_patterns = []
    for (row, col) in load_pattern_input_from_file(FILE_NAME):
        row_patterns.append(create_patterns(row, BOARD_SIZE[0], len(row) == 1, len(row) == 1))
        column_patterns.append(create_patterns(col, BOARD_SIZE[1], len(col) == 1, len(col) == 1))

    start_state = row_patterns + column_patterns
    print("Level complexity: " + str(state_complexity(start_state)))
    puzzle = NonogramSearch(NonogramNode(start_state, BOARD_SIZE[0]))

    end = time.time()
    print("\nRUNTIME: " + str(end - start) + "\n")

