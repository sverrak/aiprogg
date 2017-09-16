from Vehicle import Vehicle


class Node(object):
    """A node is here a puzzle bord construction with vehicles with a range of opportunities for further moves. """

    def __init__(self, node_id):
        self.id = node_id

        self.g_cost = 0      # the distance from the root of the search tree to this node
        self.heuristic = None   # an estimate of the distance from the node to a goal state
        self.f_cost = 999      # = g + h = the total expected cost of a solution path
        self.parent = None
        self.kids = set()

        self.vehicles = get_vehicles(node_id)

    # Necessary to make the Nodes an orderable type
    def __lt__(self, other):    # to let heapq sort nodes with similar f.costs
        return self.f_cost < other.f_cost


def get_vehicles(node_id):
    vehicles = []
    quads = [node_id[i:i+6] for i in range(0, len(node_id), 6)]
    for quad in quads:
        id = quad[0:2]
        orientation, x, y, size = [quad[i:i+1] for i in range(2, 6)]
        vehicles.append(Vehicle(int(id), int(orientation), int(x), int(y), int(size)))
    return vehicles
