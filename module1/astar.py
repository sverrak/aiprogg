def astar(init_node):
	agenda = []
	closed_nodes = []
	open_nodes = [init_node]
	concrete_costs = [0]
	estimated_costs = [estimate_cost(init_node)]
	total costs = [concrete_costs[i] + estimated_costs[i] for i in range(len(concrete_costs))]

	while (open_nodes):
		current_state = open_nodes.pop()
		closed_nodes.append(current_state)
		if(is_finished_state(current_state)):
			return current_state
		successors = generate_successors(current_state)

		for s in successors:



	

def generate_successors(current_state):
	candidate_moves = ["N", "E", "S", "W"]
	successors = []
	for i in range(len(vehicles)):
		for m in candidate_moves:
			if(is_legal_move(from_vehicle_to_board(current_state), str(vehicles[i]) + m, )


def estimate_cost(node):
	# One step for each (5 - car0.x) and one for each car blocking the exit


def sort_agenda(agenda):
	sorted_agenda = []
	return sorted_agenda
