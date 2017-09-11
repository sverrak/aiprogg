import string
import time
#import pygame, sys
#from pygame.locals import *
#import pygame # This is needed to access the PyGame framework.
#pygame.init() # This kicks things off. It initializes all the modules required for PyGame.
#pygame.display.set_mode((500, 300)) #- This will launch a window of the desired size. The return value is a Surface object which is the object you will perform graphical operations on.

DISPLAY_MODE = True
BOARD_SIZE = 6
EXIT_X = 5
EXIT_Y = 2
LEVEL = "board4.txt"
TESTING_MODE = False

# MOVE: A string consisting of two characters: 
# - a number indicating the index of the car to be moved 
# - a letter indicating the direction of the move (N,S,W,E)

def read_board_from_file(input_file):
	with open(input_file, 'r') as f:
		raw_board = f.readlines()
		return raw_board

def init_vehicles():
	print("Level: " + str(LEVEL))
	vehicles_strings = read_board_from_file(LEVEL)
	vehicles_nonintegers = [vehicles_strings[i].split(",") for i in range(len(vehicles_strings))]

	for car in vehicles_nonintegers:
		if len(car[-1])>1: car[-1] = car[-1][0]
	
	vehicles = [[int(car[i]) for i in range(len(car))] for car in vehicles_nonintegers]

	return vehicles

# I have used two representations of the states. This method converts the state from one representation to another: 
# 1) One equal to the list of vehicles given in the input files 
# 2) One equal to the visual representation of the game
def from_vehicles_to_board(vehicles):
	board = [[" " for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]
	letters = list(string.ascii_uppercase)
	#letters = [i for i in range(0,len(vehicles))]

	# Transform car data to readable board
	for car in vehicles:	
		if car[0]==0: # Horizontal case
			
			for i in range(car[-1]):
				if(car[1]+i) <= BOARD_SIZE - 1:
					board[car[2]][car[1]+i] = letters[0]
		elif car[0]==1: # Vertical case
			for i in range(car[-1]):
				if(car[2]+i) <= BOARD_SIZE - 1:
					board[car[2]+i][car[1]] = letters[0]
		else:
			"Error"

		letters = letters[1:]	
	
	return board

def print_board(board):
	temp_string = ""
	print('   ' + ' '.join([str(i) for i in range(BOARD_SIZE)]))
	print(" ---------------")
	for j in range(BOARD_SIZE):
		temp_string = str(j) + "| "
		for i in range(BOARD_SIZE):
			temp_string += str(board[j][i]) + " "
		if(j == 2):
			temp_string += " <--- EXIT"
		else:
			temp_string += "|"
		print(temp_string)
	print(" ---------------\n")

# If the move is legal, do the move
def move(vehicles, move):
	if(len(move)==3):
		vehicle = int(move[:-1])
		direction = move[-1]	
	elif(len(move)==2):
		vehicle = int(move[0])
		direction = move[1]
	elif(len(move)==1):
		print("AN ERROR HAPPENED")
		print(move)
		return vehicles
	
	if (TESTING_MODE):
		print("\n Moving vehicle " + move[0] + " in direction " + direction)
	if is_legal_move(from_vehicles_to_board(vehicles), move, vehicles):
		
		# Do the move
		vehicles_mod = [x[:] for x in vehicles]
		if(direction == "N"):
			vehicles_mod[vehicle][2] -= 1
		elif(direction == "S"):
			vehicles_mod[vehicle][2] += 1
		elif(direction == "W"):
			vehicles_mod[vehicle][1] -= 1
		elif(direction == "E"):
			vehicles_mod[vehicle][1] += 1

		return vehicles_mod
	elif (TESTING_MODE):
			print("Error. Not a legal move")
	return vehicles

# The logic of this method is fairly easy. A move is legal if certain characteristics are present:
# - The move must be horizontal or vertical
# - The post-move state must not have out-of-the-board vehicles
# - The post-move state must not have multiple vehicles in a certain board cell
def is_legal_move(board, move, vehicles):
	if(len(move)==3):
		vehicle = int(move[:-1])
		direction = move[-1]	
	elif(len(move)==2):
		vehicle = int(move[0])
		direction = move[1]
	else:
		print("AN ERRROR OCCURED")
		print(move)
		return False

	# Horizontal case
	if (vehicles[vehicle][0] == 0 and (direction == "W" or direction == "E")): 
		if(direction == "W"):
			if(vehicles[vehicle][1] > 0):
				return board[vehicles[vehicle][2]][vehicles[vehicle][1]-1] == " "
			elif (TESTING_MODE):
				print("error, not a legal direction")
			return False
		elif(direction == "E"):
			if (move[0] == "0" and vehicles[0][2] == 2 and vehicles[0][1] >= BOARD_SIZE - 2): # EXIT	
				return True
			elif(vehicles[vehicle][1] < BOARD_SIZE - vehicles[vehicle][3]):
				return board[vehicles[vehicle][2]][vehicles[vehicle][1]+vehicles[vehicle][3]] == " "
			elif (TESTING_MODE):
				print("error, E is not a legal direction")
			return False
		elif (TESTING_MODE):
			print("error, not a legal direction")
		return False

	# Vertical case
	elif (vehicles[vehicle][0] == 1 and (direction == "N" or direction == "S")): 

		if(direction == "N"):
			if(vehicles[vehicle][2] > 0):
				return board[vehicles[vehicle][2]-1][vehicles[vehicle][1]] == " "
			elif (TESTING_MODE):
				print("error, not a legal direction")
			return False
		elif(direction == "S"):
			if(vehicles[vehicle][2] < BOARD_SIZE - vehicles[vehicle][3]):
				return board[vehicles[vehicle][2]+vehicles[vehicle][3]][vehicles[vehicle][1]] == " "
			elif (TESTING_MODE):
				print("error, not a legal direction")
			return False
		elif (TESTING_MODE):
			print "Error, not legal direction"
		return False
	else:
		return False

def is_finished_state(vehicles):
	#print(vehicles)
	return vehicles[0][2] == 2 and vehicles[0][1] == BOARD_SIZE - 2# Car-0 is in exit position

def astar(init_node, mode):
	
	
	closed_nodes = [] # Indices of the nodes that are closed
	node_indices = {0: init_node} # A dictionary containing all nodes and their respective indices
	closed_states = [] # The closed nodes
	
	open_nodes = [0] # Indices of the nodes that are currently being examined or waiting to be examined
	open_states = [init_node] # The nodes that are currently being examined or waiting to be examined
	concrete_costs = {0: 0} # The number of moves needed to reach a specific node
	estimated_costs = {0: estimate_cost(node_indices[0], mode)} # The estimated number of moves left to reach final state
	total_costs = {0: concrete_costs[0] + estimated_costs[0]} # The sum of the concrete cost and the estimated cost of a certain state
	moves = {0: []} # A dictionary containing a sequence of moves needed to reach the state indicated by the key

	best_cost_development = []
	number_of_open_nodes_development = []

	# AGENDA LOOP
	while (open_nodes):
		if(len(node_indices.keys()) % 100 == 0):#in [5, 10, 100, 200, 300, 500, 1000, 2000, 3000, 5000]):
			print("NUMBER OF NODES: " + str(len(node_indices.keys())))

		if (mode == "dfs"):
			index_of_current_state = len(node_indices.keys()) - 1
			lowest_cost = total_costs[index_of_current_state]
		else:	
			index_of_current_state, lowest_cost = find_best_state(open_nodes, total_costs)

		current_state = node_indices[index_of_current_state]
		
		if(is_finished_state(current_state)): # Problem dependent
			print("\n\n*****RESULTS*****")
			print("DISCOVERED SIZE: " + str(len(node_indices.keys())))
			print("EXAMINED NODES: " + str(len(closed_states)))
			print("MOVES: " + str(len(moves[index_of_current_state])) + " - " + str(moves[index_of_current_state]))

			print_board(from_vehicles_to_board(current_state))

			return current_state, moves[index_of_current_state], best_cost_development, number_of_open_nodes_development
		
		# Update the node lists as the new node is being examined
		open_nodes.remove(index_of_current_state)
		open_states.remove(current_state)
		closed_nodes.append(index_of_current_state)
		closed_states.append(current_state)

		#
		best_cost_development.append(lowest_cost)
		number_of_open_nodes_development.append(len(open_states))
		
		# Generate successors
		successors, how_to_get_to_successors = generate_successors(current_state)
		
		# Explore the successors generated above
		for s in successors:
			# Determine the move that gets you to the successor state
			for k in how_to_get_to_successors.keys():
				if how_to_get_to_successors[k] == s:
					move = k
					break

			if contains(closed_states, s):
				continue
				for state in closed_states:
					if([k[:] for k in state] == [j[:] for j in s]):
						print(state)

			elif not contains(open_states, s): 
				index_of_current_successor = len(node_indices.keys())

				node_indices.update({index_of_current_successor: s})
				open_nodes.append(index_of_current_successor)
				open_states.append(s)
				concrete_costs.update({index_of_current_successor: concrete_costs[index_of_current_state] + 1})
				estimated_costs.update({index_of_current_successor: estimate_cost(s, mode)})
				total_costs.update({index_of_current_successor: concrete_costs[index_of_current_successor] + estimated_costs[index_of_current_successor]})
				if(index_of_current_state==0):
					moves.update({index_of_current_successor: [find_index_of(how_to_get_to_successors, s)]})
				else:
					path_to_current_successor = [i[:] for i in moves[index_of_current_state]] + [find_index_of(how_to_get_to_successors, s)]
					moves.update({index_of_current_successor: path_to_current_successor})
				
			elif contains(open_states, s):
				total_costs_temp = concrete_costs[index_of_current_state] + 1 + estimate_cost(s, mode)

				# Determine the index of the state
				former_index_of_successor = 0
				for i in range(len(open_nodes)):
					if (open_states[i] == s):
						former_index_of_successor = i
						break

				if total_costs_temp < total_costs[former_index_of_successor]:
					concrete_costs.update({former_index_of_successor: concrete_costs[index_of_current_state] + 1})
					estimated_costs.update({former_index_of_successor: estimate_cost(s, mode)})
					total_costs.update({former_index_of_successor: concrete_costs[former_index_of_successor] + estimated_costs[former_index_of_successor]})
					path_to_current_successor = [i[:] for i in moves[index_of_current_state]] + [find_index_of(how_to_get_to_successors, s)]
					moves.update({former_index_of_successor: path_to_current_successor})

			else:
				print("E")

	print ("Could not find any solution")

def contains(open_states, s):
	for state in open_states:
		if ([i[:] for i in state] == s):
			return True

	return False

def find_index_of(how_to_get_to_successors, s):
	for i in (how_to_get_to_successors.keys()):
		if([k[:] for k in how_to_get_to_successors[i]] == [j[:] for j in s]):
			return i
		

def find_best_state(open_nodes, total_costs):
	best = 999999999
	index_of_best_node = []
	#print(open_nodes)
	for node in open_nodes:
		#print(total_costs[node])
		if total_costs[node] < best:
			index_of_best_node = node
			best = total_costs[node]
	return index_of_best_node, best

def generate_successors(current_state):
	candidate_moves = ["N", "E", "S", "W"]
	successors = []
	how_to = {}
	cs = [i[:] for i in current_state]

	for i in range(len(current_state)):
		for m in candidate_moves:
			#print(from_vehicles_to_board(current_state[:]))
			#print(str(i)+m)
			#print(current_state[:])

			if(is_legal_move(from_vehicles_to_board(current_state[:]), str(i) + m, current_state[:])):
				#print(str(i) + m)
				#if(str(i) + m == "4S"):
				#	print_board(from_vehicles_to_board(current_state))
				#print(str(i))
				#print("AAA: " + str(i) + m)
				successor = move([k[:] for k in current_state], str(i) + m)
				#print("Successor: " + str(i) + m + " - " +  str(successor))
				successors.append(successor)
				how_to[str(i) + m] = successor
				
	return successors, how_to


def estimate_cost(vehicles, mode): # Problem dependent
	# One step for each (5 - car0.x) and one for each car blocking the exit
	if (mode in ["dfs", "bfs"]):
		return 0
	board = from_vehicles_to_board(vehicles)
	cost = BOARD_SIZE - vehicles[0][1] - 1
	for i in range(BOARD_SIZE - vehicles[0][1]):
		cost += 1 if board[2][i] not in ["A", " "] else 0
	return cost

def visualize_development1(vehicles, moves):
	#print("\nDevelopment")
	for m in moves:	
		vehicles = move(vehicles, m)
		#print("\n" + m)
		#print_board(from_vehicles_to_board(vehicles))


def solve(vehicles, mode):
	# A*, BFS or DFS
	if (mode == "M"):
		moves = ["0W", "0W", "4S", "4S", "3W", "3W", "3W", "3W", "5N", "5N", "4N", "4N", "0E", "0E", "0E", "0E", "0E", "0E"]
		for m in moves:
			#print("\n" + m)
			vehicles = move(vehicles, m)
			#print(vehicles)
		#print_board(from_vehicles_to_board(vehicles))
			
	elif (mode == "A*"):
		
		vehicles, moves, best_cost_development, number_of_open_nodes_development = astar(vehicles, mode)

		print(best_cost_development)

		print("\n\n\n\n\n\nHEIHEIHEI \n\n\n\n\n\n")
		print(number_of_open_nodes_development)
		#for j in (number_of_open_nodes_development):
		#	print(j)
		#print(moves)
	
	if(is_finished_state(vehicles)):
		#visualize_development1(vehicles, moves)
		#drawBoard(from_vehicles_to_board(vehicles))
		return moves
	print ("Did not find any solution")	
	return "0"



def main():

	start = time.time()
	vehicles = init_vehicles()
	board = from_vehicles_to_board(vehicles)
	print_board(board)

	# Solving the puzzle
	moves = solve(vehicles, "A*")
	end = time.time()
	print("RUNTIME: " + str(end - start) + "\n")


	if DISPLAY_MODE:
		visualize_development1(vehicles, moves)

	else:
		print(sequence)

def drawBoard(currentGame): 
	gridWidth = 80
	totalWidth = 12*gridWidth
	DISPLAYSURF = pygame.display.set_mode((totalWidth, totalWidth))
	FPS = 4 # frames per second setting	
	fpsClock = pygame.time.Clock()
	pygame.display.set_caption('Flatland Task 1')
	WHITE = (255, 255, 255)
	GREY = (192,192,192)
	BLACK = (0,0,0)
	DISPLAYSURF.fill(WHITE)
	mushroom = pygame.image.load("mushroom.bmp")
	mushroomEaten = pygame.image.load("mushroom.bmp")
	mushroomEaten.convert()
	mushroomEaten.set_alpha(30)
	bomb = pygame.image.load("bomb.bmp")
	bombEaten = pygame.image.load("bomb.bmp")
	bombEaten.convert()
	bombEaten.set_alpha(100)
	bombEaten.fill((255,255,255,30),None, pygame.BLEND_RGBA_MULT)
	
	
	wall = pygame.image.load("brick.bmp")

	mario = pygame.image.load("mario.bmp")

	
	#draw lines and walls
	for i in range(1,13):
		#lines
		pygame.draw.line(DISPLAYSURF, GREY, (gridWidth*i,0), (gridWidth*i,totalWidth), 1) 
		pygame.draw.line(DISPLAYSURF, GREY, (0,gridWidth*i), (totalWidth,gridWidth*i), 1)

	#draw
	for row in range(12):
		for col in range(12):
			if currentGame.boardRep[row][col]  == constants.WALL : 
				DISPLAYSURF.blit(wall, (col*gridWidth+5,row*gridWidth+5))
			elif currentGame.boardRep[row][col]  == constants.FOOD :
				DISPLAYSURF.blit(mushroom, (col*gridWidth+5,row*gridWidth+5))
			elif currentGame.boardRep[row][col]  == constants.FOOD_EATEN :
				DISPLAYSURF.blit(mushroomEaten, (col*gridWidth+5,row*gridWidth+5))
			elif currentGame.boardRep[row][col]  == constants.POISON :
				DISPLAYSURF.blit(bomb, (col*gridWidth-4,row*gridWidth-12))
			elif currentGame.boardRep[row][col]  == constants.POISON_EATEN :
				DISPLAYSURF.blit(bombEaten, (col*gridWidth-4,row*gridWidth-12))
			elif currentGame.boardRep[row][col]  == constants.AGENT :
				Dir = currentGame.direction
				mario = pygame.transform.rotate(mario, directions.rotateAgent[Dir])
				if Dir == directions.WEST:
					mario = pygame.transform.flip(mario,False,True)
				DISPLAYSURF.blit(mario, (col*gridWidth+14,row*gridWidth+5))
	pygame.display.update()
	fpsClock.tick(FPS)
	time.sleep(0)

main()