class Vehicle(object):

    # Create a new vehicle with its necessary properties
    def __init__(self, id, orientation, x, y, size):
        self.id = id
        self.orientation = orientation  # 0 means horizontal, 1 means vertical
        self.x_start = x
        self.y_start = y
        self.size = size                # length of vehicle

        self.x_end = x + (size - 1) * (1 - orientation)
        self.y_end = y + (size - 1) * orientation

        # Check if any of the vehicles are outside the board
        for v in [self.x_start, self.x_end, self.y_start, self.y_end]:
            if v > 5 or v < 0:
                print('Vehicle coordinate causing error:', v)
                raise ValueError("All the vehicles need to be within the puzzle board.")

        # Necessary to make the Vehicles an orderable type
        def __lt__(self, other):
            return int(self.id) < int(other.id)

