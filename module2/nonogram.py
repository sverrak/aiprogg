# -*- coding: utf-8 -*-
import time

PRINTING_MODE = False

class Segment:

    def __init__(self, x, y, length, direction):
        self.x = x
        self.y = y
        self.length = length
        self.description = "This shape has not been described yet"

    def area(self):
        return self.x * self.y

    def cells(self):
        if(self.direction == "h"):
            return [Cell(self.x + i, self.y, True) for i in range(length)]
        elif(self.direction == "v"):
            return [Cell(self.x, self.y + i, True) for i in range(length)]
        else:
            print "ERROR CREATING SEGMENT"


class Cell:

    def __init__(self, x, y, tag):
        self.x = x
        self.y = y
        self.description = "This shape has not been described yet"

    def area(self):
        return self.x * self.y

    def perimeter(self):
        return 2 * self.x + 2 * self.y

    def describe(self, text):
        self.description = text

    def authorName(self, text):
        self.author = text

    def scaleSize(self, scale):
        self.x = self.x * scale
        self.y = self.y * scale


class Row:

    def __init__(self, pattern, length):
        self.pattern = pattern
        self.length = length
        self.patterns = []

    # Generate the variable domain for each row
    def generate_patterns(self, pattern, size):
        number_of_filled_cells = sum(pattern)
        number_of_empty_cells = size - number_of_filled_cells
        number_of_segments = len(pattern)


        # List of legal patterns
        legal_patterns = []

        number_of_patterns_to_be_generated = compute_number_of_patterns(pattern,size)
        # While there are other patterns to be generated
        while True:

            temp_row = []
            
            # Generate
            for n in number_of_segments:
                continue
            break
        return 0



    def compute_number_of_patterns(self, pattern, size):
        if (len(pattern)==1):
            return max(size - pattern[0] + 1, 0)
        elif size < sum(pattern)+len(pattern) - 1:
            return 0
        else:
            temp_sum = 0
            for i in range(pattern[0], size - (sum(pattern[1:])+len(pattern[1:]))):
                print(i)
                temp_sum += self.compute_number_of_patterns(pattern[1:], size - i)
            return temp_sum

    def create_patterns(self, pattern, size, is_last=False, is_length_one=False):

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

            prefixes = self.create_patterns([pattern[0]], size - (len(pattern[1:]) + sum(pattern[1:])))
            

            for i in range(len(prefixes)):
                if(len(pattern[1:]) == 1):
                    tail_patterns = self.create_patterns(pattern[1:], size - len(prefixes[i]), True)
                else:
                    tail_patterns = self.create_patterns(pattern[1:], size - len(prefixes[i]))
                for j in range(len(tail_patterns)):
                    print("HEAD: " + str(prefixes[i])+ " - TAIL: " + str(tail_patterns[j]))
                    patterns.append(prefixes[i] + tail_patterns[j])

            return patterns

    def pattern_generation_helper(pattern, size):
        return 0




# ******* A* HELP FUNCTIONS *********

# *** Problem dependent ***

def move(args):
    
    return 0

# *** Problem dependent ***
# Checks whether a certain state is the target state or not
def is_finished_state(args):

    return 0
    
# *** Problem dependent ***
# Returns all possible neighbor states and which move that has been done to get there
def generate_successors(current_state):
    index_of_best_row, best_row = find_next_row(current_state)
    number_of_successors = len(best_row)
    successors = []
    modified_current_state = current_state[:]
    how_to = [] # How should this be implemented when we have modified A* code?

    # Revise all domains of the current state so that invalid values wrt 
    # the newly fixed row are removed
    for i in range(number_of_successors): 
        modified_current_state[index_of_best_row] = best_row[i]
        successors.append(revise(modified_current_state, i, best_row[i]))

    return successors, how_to

def revise(current_state, index, domain):


    return 0

# *** Problem dependent ***
# Computing the heuristic cost of a certain state
# Not updated to handle both columns AND rows: currently only rows 
def estimate_cost(rows, mode, column_domain):
    if (mode == "bfs" or mode == "dfs"):
        return 0
    elif (mode == "A*"):

        # If A* is chosen, we compute the degree to which each column constraints is violated
        columns = map(list, zip(*rows))
        cost = 0

        # For each column, compute the lowest 
        for i in range(len(columns)):
            cost += compute_lowest_error(columns[i], column_domain[i])
        
        return cost

# Returns the most similar row within the column_domain to the input column and the corresponding Manhattan distance
# TO DO: Not sure if the most similar row is needed
def compute_lowest_error(column, column_domain):
    closest_legal_column = []
    distance_to_closest_legal_column = 9999999

    for cd in range(len(column_domain)):
        temp_distance = 0

        for i in range(len(column)):
            temp_distance += 1 if (column_domain[cd][i] != column[i]) else 0

        if temp_distance < distance_to_closest_legal_column:
            distance_to_closest_legal_column = temp_distance
            closest_legal_column = column_domain[cd]
    print("Closest legal column to " + str(column) + " is " + str(closest_legal_column) + " with cost " + str(distance_to_closest_legal_column) + "\n")
    return distance_to_closest_legal_column

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

# Not implemented
# Is this needed?
def from_rows_to_board():

    return 0

# CURRENTLY NOT IN USE
# Heuristic function to initiate the best possible starting rows
# Currently, this method simply returns the first pattern in the pattern domain. 
# Should be improved
def init_rows(patterns_input, size):
    rows = []
    for i in range(size):
        patterns = create_patterns(patterns_input[i], size, len(patterns_input[i])==1, len(patterns_input[i])==1)
        rows.append(patterns[0])

    return rows

# Not yet completed
# Returns the size of the nonogram as well as the row and column constraints
def generate_pattern_input_from_file(input_file):

    with open(input_file, 'r') as f:
        raw = f.readlines()

        firstline = raw[0].split(" ")
        sizeX, sizeY = int(firstline[0]), int(firstline[1])

        # Put the information on the right containers
        row_patterns_input = raw[1:1+sizeX]
        row_patterns_input.reverse()
        col_patterns_input = raw[1+sizeX:]

      

        # Transform the data to list of lists of integers
        # Not working
        row_patterns_input = [[int(float(j)) for j in i.split(" ")] for i in row_patterns_input]
        col_patterns_input = [[int(float(j)) for j in i.split(" ")] for i in col_patterns_input]

        return sizeX, sizeY, row_patterns_input, col_patterns_input

# To be implemented
# Could we somehow use entropy gain here? Current idea: smaller domain is better. 
# Determines the variable on which to base the next assumption.
def find_next_row(domains):
    # Here, the fitness of a candidate row or column is the degree to which the row or column splits 

    best_row = -1
    best_fitness = 999999

    for i in range(len(domains)):
        if (len(domains[i]) < best_fitness):
            best_row = i
            best_fitness = len(domains[i])

    if (best_row == -1):
        print ("ERROR OCCURED @ FIND NEXT ROW")

    return best_row



# To be implemented
# Solving the problem using the A* algorithm
def solve(current_state):
    
    final_state, moves, best_cost_development_number_of_open_nodes_development = astar(vehicles)

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
    
    current_state = column_patterns + row_patterns

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
        moves, final_rows = solve(initial_rows, MODE)
        board = from_rows_to_board(final_rows)
        
        
        if (DISPLAY_MODE == True):
            visualize_development1(vehicles, moves)
        else:
            print("\n\n\n***LEVEL SOLVED***")
            print_board(board)

#pattern = [2,1]
#print("RESULT: " + str(compute_number_of_patterns(pattern, 7, False, len(pattern)==1)))
#print("Create_patterns: " + str(create_patterns(pattern, 5, len(pattern)==1, len(pattern)==1)))


main()