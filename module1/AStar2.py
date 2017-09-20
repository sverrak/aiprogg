
# Default values
if True:
    SEARCH_MODE = None
    DISPLAY_MODE = None
    DISPLAY_PROGRESS = None
    PRINTING_MODE = None
    DISPLAY_SPEED = None            # seconds between each update of the visualization
    DISPLAY_PROGRESS_SPEED = None   # seconds between each update of the visualization
    BOARD_SIZE = None
    EXIT_X = None
    EXIT_Y = None


def animate_solution(state=None, moves=None):
    return 0


# If the move is legal, do the move
def move(vehicles, move):
    return 0


# The logic of this method is fairly simple. A move is legal if certain characteristics are present:
# - The move must be horizontal or vertical
# - The post-move state must not have out-of-the-board vehicles
# - The post-move state must not have multiple vehicles in a certain board cell
def is_legal_move(board, move, vehicles):
    return 0


# Checks whether a certain state is the target state or not
def is_finished_state(vehicles):
    return 0


def animate_progress(current_state):
    return 0


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
        if len(node_indices) % 100 == 0:
            print("NUMBER OF NODES: " + str(len(node_indices)))

        # Update the node lists as the new node is being examined:

        # DFS mode
        if SEARCH_MODE == "dfs":
            # In this search mode, we always examine the most previous state added to the agenda
            index_of_current_state = len(node_indices) - 1
            lowest_cost = total_costs[index_of_current_state]
            current_state = open_states.pop()

            closed_nodes.append(open_nodes.pop())
            closed_states.append(current_state)

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

        if DISPLAY_PROGRESS:
            animate_progress(current_state)

        if is_finished_state(current_state):
            print("\n*****RESULTS*****")
            print("GENERATED NODES: " + str(len(node_indices)))
            print("CLOSED/EXAMINED NODES: " + str(len(closed_states)))
            print("SOLUTION STEPS:", len(moves[index_of_current_state]))
            if DISPLAY_MODE:
                animate_solution(state=current_state, moves=moves[index_of_current_state])
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
                index_of_current_successor = len(node_indices)
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
def generate_successors(current_state=None, moves=None):
    return 0


# *** Problem dependent ***
# Computing the heuristic cost of a certain state: One step for each (5 - car0.x) and one for each car blocking the exit
def estimate_cost(vehicles):
    return 0
