import numpy as np
def scale_features(features, mode=1):
    # Max & min scaling
    if mode == 1:
        for c in range(len(features[0])):
            col_max = 0
            col_min = 9999999

            # Get the right min and max value for the column
            for f in features:
                if f[c] > col_max:
                    col_max = f[c]
                elif f[c] < col_min:
                    col_min = f[c]

            # Scale each feature value
            for f in features:
                f[c] = (f[c] - col_min) / (col_max - col_min)
    # Mean & stdev scaling
    else:
        for c in range(len(features[0])):
            col_my = sum([f[c] for f in features]) / len(features)
            col_sigma = np.std([f[c] for f in features])

            for f in features:
                f[c] = (f[c] - col_my) / col_sigma

    return features


features = [[1,1,1],[2,2,2],[3,20,211]]


def number_of_labels(labels):
    uniques = []
    for l in labels:
        if(int(l) not in uniques):
            uniques.append(int(l))
    return uniques

def int_to_one_hot(int, size, off_val=0, on_val=1, floats=False):
    if floats:
        off_val = float(off_val);
        on_val = float(on_val)
    if int < size:
        v = [off_val] * size
        v[int] = on_val
        return v
print(int_to_one_hot(6,7))