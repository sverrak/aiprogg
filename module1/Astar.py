from collections import OrderedDict


class Node(object):
    """A node is here a generalized super class adapted for A*-search. """

    def __init__(self, node_id):
        self.id = node_id       # unique node identifier for hash tables / dictionaries

        self.g_cost = 0         # the distance from the root of the search tree to this node
        self.heuristic = None   # an estimate of the distance from the node to a goal state
        self.f_cost = 99999     # = g + h = the total expected cost of a solution path
        self.parent = None
        self.kids = set()

    # Necessary to make the Nodes an orderable type
    def __lt__(self, other):    # to let heapq sort nodes with similar f.costs
        return self.f_cost < other.f_cost


class AStar(object):
    def __init__(self, start_node, search_mode=False, display_mode=False, display_progression_mode=False):
        self.start_node = start_node

        self.opened = dict()            # the actual list of nodes already opened/expanded
        self.closed = dict()            # a list of already closed nodes - does not have to be ordered
        self.nodes = dict()             # a set of all unique nodes generated

        self.search_mode = search_mode
        self.display_mode = display_mode
        self.display_progression_mode = display_progression_mode

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
        adj.f_cost = adj.g_cost + self.h_estimate(adj)
        adj.heuristic = self.h_estimate(adj)
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

        if self.search_mode == "dfs":
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
            if self.search_mode == "dfs":
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

            if self.display_progression_mode:
                self.animate_progress(current_node)

            # If the goal is reached, return and animate the solution path
            if self.mission_completed(current_node):
                print("\n*****RESULTS*****")
                print('GENERATED NODES:', len(self.nodes))
                print('EXPANDED NODES:', len(self.closed))
                self.animate_solution(current_node)
                # print(best_cost_development)
                # print(number_of_open_nodes)
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

    def h_estimate(self, node):
        return 0

    def get_puzzle(self, node):
        return 0

    def generate_adj_nodes(self, node):
        return 0

    def create_node(self, elements):
        return 0

    def animate_progress(self, node):
        return 0

    def animate_solution(self, node):
        return 0

    @staticmethod
    def mission_completed(node):
        return 0
