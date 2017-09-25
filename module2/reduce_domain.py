# -*- coding: utf-8 -*-

"""
This program has five sections:
    - Declarations and parameters
    - Setup functions
    - A* helping functions (Problem specific)
    - A* helping functions (Not problem specific)
    - Running functions (main etc)

"""

### **** DECLARATIONS AND PARAMETERS ****
import math
import time
# from module1 import rushhour  # allows for using all A*-functions written for the Rush Hour assignment
import numpy as np

PRINTING_MODE = True
PRINTING_PROGRESSION = False
DISPLAY_MODE = False
DISPLAY_PROGRESS_SPEED = 0.1
DISPLAY_SPEED = 0.3  # seconds between each update of the visualization
SEARCH_MODE = "A*"
FILE_NAME = "clover.txt"

IMAGE = None

if DISPLAY_MODE or PRINTING_PROGRESSION:
    with open('./data/' + FILE_NAME, 'r') as f:
        raw = f.readlines()[0].split(" ")
        raw = [int(float(raw[i])) for i in range(len(raw))]

        BOARD_SIZE = (raw[0], raw[1])

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
    with open('./data/' + input_file, 'r') as f:
        raw = f.readlines()

        firstline = raw[0].split(" ")
        sizeX, sizeY = int(firstline[0]), int(firstline[1])

        # Put the information on the right containers
        row_patterns_input = raw[1:1 + sizeY]
        col_patterns_input = raw[1 + sizeY:]

        # Transform the data to list of lists of integers
        row_patterns_input = [[int(float(j)) for j in i.split(" ")] for i in row_patterns_input]
        col_patterns_input = [[int(float(j)) for j in i.split(" ")] for i in col_patterns_input]
        BOARD_SIZE = (sizeX, sizeY)

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
            #    patterns.append([0 for j in range(0,i)] + [1 for l in range(pattern[0])] + [0 for k in range(size-i-pattern[0])])
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
def generate_successors(current_state, moves):
    # Setup of data structures
    unmodified_rows_and_columns = [i for i in range(len(current_state))]

    if (len(moves) > 0):
        for i in range(len(moves)):
            # print(int(moves[i][0:moves[i].index(',')]))
            if (any(x == int(moves[i][0:moves[i].index(',')]) for x in unmodified_rows_and_columns)):
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
        modified_current_state[index_of_best_row] = []
        modified_current_state[index_of_best_row] = [current_state[index_of_best_row][i]]

        # Compute revised state
        revised_state = revise(modified_current_state, index_of_best_row, current_state[index_of_best_row][i],
                               is_column, number_of_rows)

        if (contains([current_state], revised_state)):
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
# Could we somehow improve this heuristic? (use entropy gain?) Current idea: smaller domain is better.
# Determines the variable on which to base the next assumption.

def find_next_row_or_column(domains, unmodified_rows_and_columns):
    # Potential improvement:
    # - Legalise choosing domains of length 1. This requires restructuring the code so that current state involves

    # Here, the fitness of a candidate row or column is the size of the row/column domain

    index_of_best_row = -1
    fitness_of_best_row = 99999999999999999999999999999999
    for i in range(len(domains)):
        if (len(domains[i]) < fitness_of_best_row):
            if (len(domains[i]) > 1 or i in unmodified_rows_and_columns):
                # print("len domains", len(domains[i]))
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

                if (current_state[i][v][number_of_rows - index_of_best_row - 1] == domain[i - number_of_rows]):
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

        # max {     1 if count(len(current_state[i]) > 1, i: row indices) > 1           else 0
        #       +   1 if count(len(current_state[i]) > 1, i: column indices) > 1        else 0,
        #           1 if count(len(current_state[i]) > 1, i: row+column indices) >= 1   else 0}

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

        return min_cost

        # Old method that is not
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


### **** A* HELPING FUNCTIONS (NOT PROBLEM DEPENDENT) ****
### **** THE FOLLOWING FUNCTIONS ARE NOT CHANGED FROM RUSHHOUR.PY ****

# Finds the state s in open_nodes that has the lowest total cost
def find_best_state(open_nodes, total_costs):
    best = 9999999999999999999999999999999999999999999999999999999999
    index_of_best_node = 0

    for node in open_nodes:
        if total_costs[node] < best:
            index_of_best_node = node
            best = total_costs[node]

    return index_of_best_node, best


# Check if s is contained in open_states
def contains(open_states, s):
    return any(state == s for state in open_states)


# Finds the index of s in how_to_get_to_successors
def find_index_of(how_to_get_to_successors, s):
    for i in (how_to_get_to_successors.keys()):
        if [k[:] for k in how_to_get_to_successors[i]] == [j[:] for j in s]:
            return i


def reduce_domain(current_state):
    number_of_rows = len(current_state[0][0])
    number_of_columns = len(current_state[-1][0])
    number_of_removed_permutations = 0
    result_array = [i for i in range(100)]

    while not all(result_array[i] == result_array[0] for i in range(len(result_array))):
        # Choose an index (at random) for a row/column of length > 1
        indices = []
        for i in range(len(current_state)):
            if (len(current_state[i]) > 1):
                indices.append(i)

        chosen_index = indices[np.random.randint(0, len(indices))]
        # chosen_index = indices[9]

        # Loop through all cells in the chosen row and check which cell values that are equal for each permutations

        if (chosen_index < number_of_rows):
            number_of_iterations = len(current_state[chosen_index][0])
        else:
            number_of_iterations = len(current_state[chosen_index][-1])

        number_of_zeros = [0] * number_of_iterations
        # print("nZeros", number_of_zeros)
        all_zeros = [0] * number_of_iterations
        all_ones = [0] * number_of_iterations
        all_zeros_indices = []
        all_ones_indices = []

        # print(len(perm))

        print("Chose index " + str(chosen_index))
        # print(current_state[chosen_index])
        # for i in current_state[chosen_index]:
        #    print(i)

        # print("\n Part 1")
        # print("nPerms: " + str(len(current_state[chosen_index])))

        for p in range(len(current_state[chosen_index])):  # 0-1
            for c in range(len(current_state[chosen_index][p])):  # 0-8
                if (current_state[chosen_index][p][c] == 0):
                    number_of_zeros[c] = number_of_zeros[c] + 1

        """for perm in range(len(current_state[chosen_index])):

            for k in range(len(current_state[chosen_index][perm])):
                #print("permk", k)

                if(current_state[chosen_index][perm][k] == 0):

                    #print(k)
                    #print(number_of_zeros[k])
                    a = number_of_zeros[k] + 1
                 #   print("a", a)

                    #print(number_of_zeros[k] + 1)
                    #print(number_of_zeros)
                    number_of_zeros.remove(number_of_zeros[k])
                    number_of_zeros.insert(k, a)
                    #print(number_of_zeros)
                    #print("\n\n\n\n\n")
                    #print(number_of_zeros[k])
                    #number_of_zeros[k] = number_of_zeros[k] + 1
            #print("nZeros", number_of_zeros)"""

        # print("\n Part 2")
        # print(1, current_state[chosen_index][0])
        # print([i for i in range(len(number_of_zeros))])
        # print(number_of_zeros)

        for k in range(number_of_iterations):
            # print(k, number_of_zeros[k], len(current_state[chosen_index]))
            if (number_of_zeros[k] == len(current_state[chosen_index]) and number_of_zeros[k] > 0):
                all_zeros[k] = 1
                all_zeros_indices.append(k)
            elif (number_of_zeros[k] == 0):
                all_ones_indices.append(k)

        # print("nZeros", number_of_zeros)
        # print("All permutations agree on 0/1s for:", all_zeros_indices, all_ones_indices)

        # Update all column/row domains so that they are coherent with the new information

        # Part 3
        # Chosen index is a row
        if (chosen_index < number_of_rows):

            for i in range(number_of_rows, len(current_state)):
                number_of_permutations = len(current_state[i])
                active_permutations = current_state[i]

                for perm_iterator in range(0, number_of_permutations):
                    for cell in all_zeros_indices:
                        # print("Cell: ", int(cell))
                        # print("perm::", perm_iterator)
                        # print("len(i)::", number_of_permutations)
                        print("text:: ", active_permutations)
                        if (active_permutations[perm_iterator][int(cell)] != 0):
                            valid_permutations = current_state[0:i] + \
                                                 valids(active_permutations, cell, 0) + current_state[i + 1:]
                            # valid_permutations = valids(active_permutations, cell, 0)
                            current_state = valid_permutations
                            number_of_removed_permutations += 1
                    for cell in all_ones_indices:
                        if (active_permutations[perm_iterator][int(cell)] != 1):
                            valid_permutations = valids(active_permutations, cell, 1)
                            current_state[i] = valid_permutations



        # Chosen index is a column
        else:
            # print("\n\nCurrently examining rows")
            for i in range(number_of_rows):

                number_of_permutations = len(current_state[i])
                active_permutations = current_state[i]
                # print("numb perm: ", number_of_permutations)
                # print("Actives: ", current_state[i])
                # print("All zeros:", all_zeros_indices)

                for perm_iterator in range(0, number_of_permutations):
                    for cell in all_zeros_indices:
                        # print("Cell: ", int(cell))
                        # print("perm::", perm_iterator)
                        # print("len(i)::", number_of_permutations)
                        # print("text:: ", active_permutations)
                        print(active_permutations[perm_iterator])
                        if (active_permutations[perm_iterator][int(cell)] != 0):
                            # valid_permutations = valids(active_permutations, cell, 0)
                            valid_permutations = current_state[0:i] + \
                                                 valids(active_permutations, cell, 0) + current_state[i + 1:]

                            current_state[i] = valid_permutations
                            number_of_removed_permutations += 1  # sum([x for x in active_permutations if not notequal(x[int(cell)], 0)])
                    for cell in all_ones_indices:
                        if (active_permutations[perm_iterator][int(cell)] != 1):
                            valid_permutations = valids(active_permutations, cell, 1)
                            current_state[i] = valid_permutations

        result_array.append(sum(len(i) for i in current_state))
        result_array.pop(0)
        current_size = sum(len(i) for i in current_state)
        # print("Number of removed permutations: " + str(number_of_removed_permutations) + "\n\n\n\n\n\n")
        # print("Remaining size: ", sum(len(i) for i in current_state))
        if (is_finished_state(current_state)):
            return current_state

    # print(result_array)
    return current_state


def valids(actives, pos, num):
    new = []
    for a in actives:
        if a[pos] == num:
            new.append(a)
    return new


def notequal(considered_object, num):
    return considered_object != num


def print_solution(current_state):
    for i in range(len(current_state)):
        print(len(current_state[i]))


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


### **** RUNNING FUNCTIONS ****
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

        # Printing progression

        if PRINTING_PROGRESSION:
            print_progression(current_state)

        if is_finished_state(current_state):
            print("Puzzle solved.")
            print("\n\n*****RESULTS*****")
            print("GENERATED NODES: " + str(len(node_indices.keys())))
            print("CLOSED/EXAMINED NODES: " + str(len(closed_states)))
            print("Opened", len(open_nodes))
            # print("MOVES: " + str(len(moves[index_of_current_state])) + " - " + str(moves[index_of_current_state]))

            return current_state, moves[index_of_current_state], best_cost_development, number_of_open_nodes_development

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


# Solving the problem using the A* algorithm
def solve(current_state):
    final_state, moves, best_cost_development, number_of_open_nodes_development = astar(current_state)

    if PRINTING_MODE:
        print("\n\nDevelopment of best cost: " + str(best_cost_development))
        print("Development of number of open nodes: " + str(number_of_open_nodes_development))

    if is_finished_state(final_state):
        return final_state

    print("Did not find any solution")
    return "0"
    return 0


if __name__ == '__main__':
    print("\n************************************\n************************************")

    print("Level: " + str(FILE_NAME))
    print("Algorithm: " + SEARCH_MODE + "\n")

    # Initializing the program
    start = time.time()
    sizeX, sizeY, row_patterns_input, col_patterns_input = generate_pattern_input_from_file(FILE_NAME)

    row_patterns = []
    column_patterns = []

    for i in range(len(col_patterns_input)):
        column_patterns.append(create_patterns(col_patterns_input[i], sizeY, len(col_patterns_input[i]) == 1,
                                               len(col_patterns_input[i]) == 1))

    for i in range(len(row_patterns_input)):
        row_patterns.append(create_patterns(row_patterns_input[i], sizeX, len(row_patterns_input[i]) == 1,
                                            len(row_patterns_input[i]) == 1))

    current_state = row_patterns + column_patterns

    size = 0

    for i in current_state:
        if (len(i) > size):
            size = len(i)
    print("Level complexity: " + str(size))
    # print(current_state)

    # for i in current_state:
    # print(i)

    # successors, how_to = generate_successors(current_state)




    final_state = reduce_domain(current_state)
    #    final_state = solve(current_state)
    # print(final_state)

    # print((row_patterns[0]))
    for r in range(sizeY):
        temp = ""
        for i in range(len(final_state[sizeY - 1 - r][0])):
            temp += str(final_state[sizeY - 1 - r][0][i]) + " "

        print(temp)

    # initial_rows = init_rows(row_patterns_input, sizeX)
    # print(initial_rows)
    # print(estimate_cost(initial_rows, "A*", column_patterns))


    end = time.time()

    # Displaying run characteristics
    if (PRINTING_MODE == True):
        print("RUNTIME: " + str(end - start) + "\n")

