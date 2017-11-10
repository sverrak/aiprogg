import numpy as np
from matplotlib import pyplot as plt

class TSP(object):
    """docstring for TSPInstance"""

    def __init__(self, file_name):
        # super(TSPInstance, self).__init__()
        self.data = file_reader(file_name)
        self.cities = [row[0] for row in self.data]
        self.coordinates = [[float(row[1]), float(row[2])] for row in self.data]
        self.distances = []

    def compute_distances(self):
        distances = []
        for index1, city1 in enumerate(self.coordinates):
            distances.append([])
            for city2 in self.coordinates:
                dist = distance_between_cities(city1, city2)
                distances[index1].append(dist)
        return np.array(distances)

    def get_distances(self):
        self.distances = self.compute_distances()
        return self.distances

    def get_total_distance(self):
        # TODO
        return 0

    def plot_map(self):
        fig, ax = plt.subplots()

        ax.plot([c[0] for c in self.coordinates], [c[1] for c in self.coordinates], marker='*', c='gold',
                       markersize=15, linestyle='None')
        ax.set_xlim(0, max([c[0] for c in self.coordinates])*1.05)  # adjust figure axes to max x- and y-values
        ax.set_ylim(0, max([c[1] for c in self.coordinates])*1.05)

        plt.pause(0.5)

        for i, sol in enumerate([[[x[0]*0.9, x[1]] for x in self.coordinates], [[x[0]*1.1, x[1]*1.1] for x in self.coordinates]]):
            # map.set_data(city[0], city[1])
            if i == 0:
                map, = ax.plot([c[0] for c in sol], [c[1] for c in sol], marker='o', markerfacecolor='None', c='green',
                       markersize=10, linestyle=':')
            else:
                map.set_data([c[0] for c in sol], [c[1] for c in sol])

            plt.pause(0.5)

        plt.show()


def distance_between_cities(i, j):
    return ((i[1] - j[1]) ** 2 + (i[0] - j[0]) ** 2) ** 0.5


def print_distances(distances):
    temp_string = ""
    for row in distances:
        temp_string += str(row[0]) + "\t" + row[1] + "\t" + row[2] + '\n'
    print(temp_string)


def file_reader(filename):
    with open(filename) as f:
        file = f.readlines()[5:]
        data = []
        for line in file:
            if line == 'EOF\n':
                break
            data.append(line.replace('\n', '').split())
    return data

# ------------------------------------------

# ****  MAIN functions ****

if __name__ == '__main__':
    FILE = 1
    test = TSP('./data/' + str(FILE) + '.txt')

    test.plot_map()
