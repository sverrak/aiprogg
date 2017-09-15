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

import time
PRINTING_MODE = False
SEARCH_MODE = "A*"



### **** SETUP FUNCTIONS ****

# Returns the size of the nonogram as well as the row and column constraints
def generate_pattern_input_from_file(input_file):

    with open(input_file, 'r') as f:
        raw = f.readlines()

        firstline = raw[0].split(" ")
        sizeX, sizeY = int(firstline[0]), int(firstline[1])

        # Put the information on the right containers
        row_patterns_input = raw[1:1+sizeX]
        col_patterns_input = raw[1+sizeX:]

      

        # Transform the data to list of lists of integers
        # Not working
        row_patterns_input = [[int(float(j)) for j in i.split(" ")] for i in row_patterns_input]
        col_patterns_input = [[int(float(j)) for j in i.split(" ")] for i in col_patterns_input]

        return sizeX, sizeY, row_patterns_input, col_patterns_input

# Returns the variable domain for a certain pattern (row constraints) 
def create_patterns(pattern, size, is_last=False, is_length_one=False):
        if (len(pattern)==1):
            patterns = []
            temp_pattern = []
            for i in range(size-pattern[0]+1):
                if(is_last == True):
                    patterns.append([0 for j in range(0,i)] + [1 for l in range(pattern[0])] + [0 for k in range(size-i-pattern[0])])    
                #elif(is_length_one == True):
                #    patterns.append([0 for j in range(0,i)] + [1 for l in range(pattern[0])] + [0 for k in range(size-i-pattern[0])])    
                else: 
                    patterns.append([0 for j in range(0,i)] + [1 for l in range(pattern[0])] + [0])#+ [0 for k in range(i+1, size-pattern[0]+1)])
            return patterns
        elif size < sum(pattern)+len(pattern) - 1:
            return [-1]
        else:
            patterns = []
            temp_pattern = []

            prefixes = create_patterns([pattern[0]], size - (len(pattern[1:]) + sum(pattern[1:])))
            

            for i in range(len(prefixes)):
                if(len(pattern[1:]) == 1):
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
        if (len(i)!=1):
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
    how_to = [] # How should this be implemented when we have modified A* code?
    number_of_columns = len(current_state[0][0])
    number_of_rows = len(current_state) - number_of_columns
    #print("number of columns " + str(number_of_columns))
    #print(number_of_rows)
    is_column = index_of_best_row >= number_of_rows # True if the the "best row" is actually a column
    
    #print("Index of br: " + str(index_of_best_row))
    # Revise all domains of the current state so that invalid values wrt 
    # the newly fixed row are removed
    for i in range(number_of_successors): 
        #print("gototo")
        #print([current_state[index_of_best_row][i]])
        #print(modified_current_state[index_of_best_row])
        modified_current_state[index_of_best_row] = []
        modified_current_state[index_of_best_row] = [current_state[index_of_best_row][i]]
        
        how_to.append("0") # TO DO
        successors.append(revise(modified_current_state, index_of_best_row, current_state[index_of_best_row][i], is_column, number_of_rows))

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
        print ("ERROR OCCURED @ FIND NEXT ROW")

    
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
    if(is_column):

        for i in range(number_of_rows):
            modified_state.append([])
            for v in range(len(current_state[i])):
                if(current_state[i][v][index_of_best_row - number_of_rows] == domain[number_of_rows - i - 1]):
                    modified_state[i].append(current_state[i][v])

        modified_state += current_state[number_of_rows:]
    
    # If a row is chosen 
    else:
        modified_state += current_state[:number_of_rows]


        for i in range(number_of_rows, len(current_state)):
            modified_state.append([])
            for v in range(len(current_state[i])):
                if(current_state[i][v][number_of_columns - index_of_best_row - 1] == domain[i-number_of_rows]):
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
        
### **** A* HELPING FUNCTIONS (NOT PROBLEM DEPENDENT) ****
### **** THE FOLLOWING FUNCTIONS ARE NOT CHANGED FROM RUSHHOUR.PY ****

# Finds the state s in open_nodes that has the lowest total cost
def find_best_state(open_nodes, total_costs):
    best = 9999
    index_of_best_node = 0
    #print(open_nodes)
    for node in open_nodes:
        #print(total_costs[node])
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


### **** RUNNING FUNCTIONS ****
# The A* search function
# Not touched yet. However, we must do something to the node linking etc
def astar(init_node):

    # Initialization of the node data structures
    closed_nodes = []               # Indices of the nodes that are closed
    node_indices = {0: init_node}   # A dictionary containing all nodes and their respective indices
    closed_states = []              # The closed nodes
    open_nodes = [0]                # Indices of the nodes that are currently being examined or waiting to be examined
    open_states = [init_node]       # The nodes that are currently being examined or waiting to be examined

    # Initialization of data structures describing certain features of each state
    concrete_costs = {0: 0}     # The number of moves needed to reach a specific node
    estimated_costs = {0: estimate_cost(node_indices[0])} # The estimated number of moves left to reach final state
    total_costs = {0: concrete_costs[0] + estimated_costs[0]} # The sum of the concrete cost and the estimated cost of a certain state
    moves = {0: []}     # A dictionary containing a sequence of moves needed to reach the state indicated by the key

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
            current_state = node_indices[index_of_current_state]

            open_nodes.remove(index_of_current_state)
            open_states.remove(current_state)
            closed_nodes.append(index_of_current_state)
            closed_states.append(current_state)

        # Printing progression
        #if PRINTING_PROGRESSION:
        #    plt.title('Rush Hour PROGRESS simulation')
        #    board = current_state[0:]
        #    IMAGE.set_data(board)
        #    plt.pause(DISPLAY_PROGRESS_SPEED)  # seconds between each update of the visualization

        if is_finished_state(current_state):
            print("\n\n*****RESULTS*****")
            print("GENERATED NODES: " + str(len(node_indices.keys())))
            print("CLOSED/EXAMINED NODES: " + str(len(closed_states)))
            print("MOVES: " + str(len(moves[index_of_current_state])) + " - " + str(moves[index_of_current_state]))
            return current_state, moves[index_of_current_state], best_cost_development, number_of_open_nodes_development

        # Saves information about the state
        best_cost_development.append(lowest_cost)
        number_of_open_nodes_development.append(len(open_states))

        # Generate successors
        successors, how_to_get_to_successors = generate_successors(current_state)

        # Explore the successors generated above
        for s in successors:

            # 1. The successor has already been examined (successor in closed_nodes)
            if contains(closed_states, s):
                continue        # Do nothing

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
                    total_costs.update({former_index_of_successor: concrete_costs[former_index_of_successor] + estimated_costs[former_index_of_successor]})
                    path_to_current_successor = [i[:] for i in moves[index_of_current_state]] + [find_index_of(how_to_get_to_successors, s)]
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
                total_costs.update({index_of_current_successor: concrete_costs[index_of_current_successor] + estimated_costs[index_of_current_successor]})

                # If the parent is the initial state
                if(index_of_current_state==0):

                    # The path to the successor will simply be equal to the move from the initial state to the successor
                    moves.update({index_of_current_successor: [find_index_of(how_to_get_to_successors, s)]})

                # For all other parent states
                else:

                    # We append the move from the parent state to the successor to the path of the parent and save that as the path
                    # from the initial state to the successor
                    path_to_current_successor = [i[:] for i in moves[index_of_current_state]] + [find_index_of(how_to_get_to_successors, s)]
                    moves.update({index_of_current_successor: path_to_current_successor})

            # Neither alternative 1., 2. nor 3.
            else:       # Something is wrong...
                print("Error")

    # If the loop does not find any solution
    raise ValueError('Could not find any solution')

# Solving the problem using the A* algorithm
def solve(current_state):
    
    final_state, moves, best_cost_development_number_of_open_nodes_development = astar(current_state)

    if PRINTING_MODE:
        print("\n\nDevelopment of best cost: " + str(best_cost_development))
        print("Development of number of open nodes: " + str(number_of_open_nodes_development))

    if is_finished_state(final_state):
        return final_state

    print ("Did not find any solution")
    return "0"
    return 0

def main():
    print("\n************************************\n************************************")
    #print("Level: " + str(LEVEL))
    #print("Algorithm: " + MODE + "\n")


    # Initializing the program
    #start = time.time()
    sizeX, sizeY, row_patterns_input, col_patterns_input = generate_pattern_input_from_file("cat.txt")
    
    row_patterns = []
    column_patterns = []

    for i in range(len(col_patterns_input)):
        column_patterns.append(create_patterns(col_patterns_input[i], sizeY, len(col_patterns_input[i])==1, len(col_patterns_input[i])==1))
    
    for i in range(len(row_patterns_input)):
        row_patterns.append(create_patterns(row_patterns_input[i], sizeX, len(row_patterns_input[i])==1, len(row_patterns_input[i])==1))
    
    current_state = row_patterns + column_patterns

    

    #successors, how_to = generate_successors(current_state)

    
    final_state = solve(current_state)
    
    #print((row_patterns[0]))
    


    #initial_rows = init_rows(row_patterns_input, sizeX)
    #print(initial_rows)
    #print(estimate_cost(initial_rows, "A*", column_patterns))


    end = time.time()

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


main()