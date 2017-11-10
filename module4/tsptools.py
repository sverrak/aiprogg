

def filereader(filename):
	with open(filename) as f:
    	data = f.readlines()
    return data


class TSP(object):
	"""docstring for TSPInstance"""
	def __init__(self, data):
		super(TSPInstance, self).__init__()
		self.cities = [row[0] for row in data]
		self.distances = self.compute_distances(data)

	def compute_distances(self, data):
		distances = np.array([[]])

		for i in data:
			numpy.append(distances, np.array([]))
			for j in data:
				dist_i_j = distance(i,j)
				numpy.append(distances[i], np.array(dist_i_j))

		self.distances = distances

	def get_distances(self):
		return self.distances

	def distance(i, j):
		return math.sqrt((i[1] - j[1]) ** 2 + (i[2] - j[2]) ** 2)

	def print_distances(distances):
		for row in distances:
			temp_string = ""
			for j in row:
				temp_string += str(j) + "\t"
			print(temp_string)




data = filereader()
