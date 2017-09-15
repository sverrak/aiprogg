import copy
import itertools
import numpy as np

DISPLAY_MODE = True
PRINTING_PROGRESSION = False
DISPLAY_SPEED = 0.3		  # seconds between each update of the visualization

BOARD_SIZE = 10
IMAGE = None

if DISPLAY_MODE or PRINTING_PROGRESSION:
    import matplotlib.pyplot as plt
    import matplotlib.cbook

    # Remove annoying warning from matplotlib.animation
    import warnings
    warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

    IMAGE = plt.imshow(np.zeros((BOARD_SIZE, BOARD_SIZE)), cmap='Greys', interpolation='nearest', vmin=0, vmax=1)
    plt.title('Nonogram')


class CSP:
    def __init__(self):
        # self.variables is a list of the variable names in the CSP
        self.variables = []

        # self.domains[i] is a list of legal values for variable i
        self.domains = {}

        # self.constraints[i][j] is a list of legal value pairs for
        # the variable pair (i, j)
        self.constraints = {}

        self.backtracking = 0
        self.failure = 0
        self.infer = 0

    def add_variable(self, name, domain):
        """Add a new variable to the CSP. 'name' is the variable name
        and 'domain' is a list of the legal values for the variable.
        """
        self.variables.append(name)
        self.domains[name] = list(domain)
        self.constraints[name] = {}

    @staticmethod
    def get_all_possible_pairs(a, b):
        """Get a list of all possible pairs (as tuples) of the values in
        the lists 'a' and 'b', where the first component comes from list
        'a' and the second component comes from list 'b'.
        """
        return itertools.product(a, b)

    def get_all_arcs(self):
        """Get a list of all arcs/constraints that have been defined in
        the CSP. The arcs/constraints are represented as tuples (i, j),
        indicating a constraint between variable 'i' and 'j'.
        """
        return [(i, j) for i in self.constraints for j in self.constraints[i]]

    def get_all_neighboring_arcs(self, var):
        """Get a list of all arcs/constraints going to/from variable
        'var'. The arcs/constraints are represented as in get_all_arcs().
        """
        return [(i, var) for i in self.constraints[var]]

    def add_constraint_one_way(self, i, j, filter_function):
        """Add a new constraint between variables 'i' and 'j'. The legal
        values are specified by supplying a function 'filter_function',
        that returns True for legal value pairs and False for illegal
        value pairs. This function only adds the constraint one way,
        from i -> j. You must ensure that the function also gets called
        to add the constraint the other way, j -> i, as all constraints
        are supposed to be two-way connections!
        """
        if j not in self.constraints[i]:
            # First, get a list of all possible pairs of values between variables i and j
            self.constraints[i][j] = self.get_all_possible_pairs(self.domains[i], self.domains[j])

        # Next, filter this list of value pairs through the function
        # 'filter_function', so that only the legal value pairs remain
        self.constraints[i][j] = [value_pair for value_pair in self.constraints[i][j] if filter_function(*value_pair)]

    # TODO: ikke nødvendig for nonogram
    def add_all_different_constraint(self, variables):
        """Add an Alldiff constraint between all of the variables in the list 'variables'. """
        for (i, j) in self.get_all_possible_pairs(variables, variables):
            if i != j:
                self.add_constraint_one_way(i, j, lambda x, y: x != y)

    def backtracking_search(self):
        """This functions starts the CSP solver and returns the found solution. """
        # Make a so-called "deep copy" of the dictionary containing the
        # domains of the CSP variables. The deep copy is required to
        # ensure that any changes made to 'domain' does not have any
        # side effects elsewhere.
        domain = copy.deepcopy(self.domains)

        # Run AC-3 on all constraints in the CSP, to weed out all of the
        # values that are not arc-consistent to begin with
        self.inference(domain, self.get_all_arcs())

        # Call backtrack with the partial domain 'domain'
        return self.backtrack(domain)

    def backtrack(self, domain):
        """The function 'Backtrack' from the pseudocode in the
        textbook.

        The function is called recursively, with a partial domain of
        values 'domain'. 'domain' is a dictionary that contains
        a list of all legal values for the variables that have *not* yet
        been decided, and a list of only a single value for the
        variables that *have* been decided.

        When all of the variables in 'domain' have lists of length
        one, i.e. when all variables have been assigned a value, the
        function should return 'domain'. Otherwise, the search
        should continue. When the function 'inference' is called to run
        the AC-3 algorithm, the lists of legal values in 'domain'
        should get reduced as AC-3 discovers illegal values.

        IMPORTANT: For every iteration of the for-loop in the
        pseudocode, you need to make a deep copy of 'domain' into a
        new variable before changing it. Every iteration of the for-loop
        should have a clean slate and not see any traces of the old
        domains and inferences that took place in previous
        iterations of the loop.
        """
        # TODO: IMPLEMENT THIS  (=> this was part of the assignment)

        self.backtracking += 1
        length = 0

        for value in domain:
            if len(domain[value]) == 1:
                length += 1
        if length == len(domain):
            # print("Backtracking found a SOLUTION!")
            self.domains = copy.deepcopy(domain)    # updating the domain to the current (and correct) domain
            return domain

        variable = self.select_unassigned_variable(domain)
        for value in domain[variable]:

            # Deep copy domain to make sure no permanent changes are being made
            deep_copy = copy.deepcopy(domain)

            # add variable = value to deep_copy and remove all the other values from the domain
            deep_copy[variable] = value

            # PRINT only for debugging
            # print("domain:", deep_copy)
            # print("Variable:", variable)
            # print("Domain:", domain[variable])
            # print("Value: ", value)

            # making all the variables arc-consistent
            inferences = self.inference(deep_copy, self.get_all_neighboring_arcs(variable))
            if inferences:
                result = self.backtrack(deep_copy)
                if result:
                    return result
            self.failure += 1
        return False

    def select_unassigned_variable(self, domain):
        """The function 'Select-Unassigned-Variable' from the pseudocode
        in the textbook. Should return the name of one of the variables
        in 'domain' that have not yet been decided, i.e. whose list
        of legal values has a length greater than one.
        """
        # TODO: IMPLEMENT THIS
        for value in domain:
            if len(domain[value]) > 1:
                return value
        return 0

    def inference(self, domain, queue):
        """The function 'AC-3' from the pseudocode in the textbook.
        'domain' is the current partial domain, that contains
        the lists of legal values for each undecided variable. 'queue'
        is the initial queue of arcs that should be visited.
        """
        # TODO: IMPLEMENT THIS
        while queue:
            (i, j) = queue.pop(0)

            # making i arc-consistent with respect to j
            if self.revise(domain, i, j):
                if len(domain[i]) == 0:
                    return False    # the CSP has not consistent solution
                for k in self.get_all_neighboring_arcs(i):
                    if k[0] != j and k[1] != j:
                        queue.append(k)
        return True

    def revise(self, domain, i, j):
        """The function 'Revise' from the pseudocode in the textbook.
        'domain' is the current partial domain, that contains
        the lists of legal values for each undecided variable. 'i' and
        'j' specifies the arc that should be visited. If a value is
        found in variable i's domain that doesn't satisfy the constraint
        between i and j, the value should be deleted from i's list of
        legal values in 'domain'.
        """
        # TODO: IMPLEMENT THIS
        revised = False
        for x in domain[i]:
            # keeping track of the number of incidences where (x,y) is not possible

            counter = 0
            for y in domain[j]:
                if not ((x, y) in self.constraints[i][j]):
                    counter += 1

            # if x in i conflicts with y in j for every possible y, eliminate x in i
            if counter == len(domain[j]):
                domain[i].remove(x)     # eliminating for the domain
                revised = True
        return revised


def create_sudoku_csp(filename):
    """Instantiate a CSP representing the Sudoku board found in the text
    file named 'filename' in the current directory.
    """
    csp = CSP()
    # board = [x.strip() for x in open(filename, 'r')]
    board = '000030040\n109700000\n000851070\n002607830\n906010207\n031502900\n010369000\n000005703\n090070000'.split('\n')

    for row in range(9):
        for col in range(9):
            if board[row][col] == '0':
                csp.add_variable('%d-%d' % (row, col), list(map(str, list(range(1, 10)))))
            else:
                csp.add_variable('%d-%d' % (row, col), [board[row][col]])

    for row in range(9):
        csp.add_all_different_constraint(['%d-%d' % (row, col) for col in range(9)])
    for col in range(9):
        csp.add_all_different_constraint(['%d-%d' % (row, col) for row in range(9)])
    for box_row in range(3):
        for box_col in range(3):
            cells = []
            for row in range(box_row * 3, (box_row + 1) * 3):
                for col in range(box_col * 3, (box_col + 1) * 3):
                    cells.append('%d-%d' % (row, col))
            csp.add_all_different_constraint(cells)

    return csp


def print_nonogram(solution):
    solution = [[1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
             [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
             [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], [0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
             [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
             [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
             [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
             [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], ]

    IMAGE.set_data(solution)
    plt.pause(DISPLAY_SPEED)
    plt.show()  # stops image from disappearing after the short pause


# Test for Sudoku
sol = create_sudoku_csp("medium.txt")
solution = sol.backtracking_search()
print('Number of times backtracking: ', sol.backtracking)
print('Number of backtracking failures: ', sol.failure)
print_nonogram(solution)
