import math


# *** GENERAL FUNCTIONS ***

def scale_coordinates(coordinates):
    scale_down_factor = []
    for i in range(2):

        # Max & min scaling
        c_max = max([c[i] for c in coordinates])
        c_min = min([c[i] for c in coordinates])

        c_diff = c_max - c_min

        # Scale each feature value
        for c in range(len(coordinates)):
            coordinates[c][i] = (coordinates[c][i] - c_min) / c_diff

        scale_down_factor.append(c_diff)

    return coordinates, scale_down_factor


def euclidian_distance(i, j):
    return ((i.x - j.x) ** 2 + (i.y - j.y) ** 2) ** 0.5


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


def PointsInCircum(r, center_x, center_y, n=100):
        return [(math.cos(2*math.pi/n*x)*r+center_x, math.sin(2*math.pi/n*x)*r+center_y) for x in range(0, n+1)]


def argmin(list):
        argmin, minvalue = 0, list[0]
        for i,val in enumerate(list):
            if val < minvalue:
                argmin, minvalue = i, val
        return argmin, minvalue


def manhattan_distance(x,y):
    return sum([abs(x[i] - y[i]) for i in range(len(x))])


# def write_to_file(res_array):
#     import xlsxwriter
#     book = xlsxwriter.Workbook('module4results.xlsx')
#     sheet = book.add_worksheet("Results")
#
#     headers = ['Learning rate0', 'Learning rateTAU', 'Sigma0', 'SigmaTau']
#
#     for h in range(len(headers)):
#         sheet.write(0, h, headers[h])
#
#     for r in range(len(res_array)):
#         for i in range(len(res_array[r])):
#             sheet.write(r + 1, i, res_array[r][i])

