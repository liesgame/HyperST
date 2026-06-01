import numpy as np
import pandas as pd
import sklearn


def refine(sample_id, pred, dis, shape="hexagon"):
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    if shape == "hexagon":
        num_nbs = 6
    elif shape == "square":
        num_nbs = 4
    else:
        print(
            "Shape not recongized, shape='hexagon' for Visium data, 'square' for ST data."
        )
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values()
        nbs = dis_tmp[0 : num_nbs + 1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        if (v_c.loc[self_pred] < num_nbs / 2) and (np.max(v_c) > num_nbs / 2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    return refined_pred

def find_pix_dist_between_spots(points):
    print("points shape: " + str(points.shape))
    if points.shape[0] < 100000:
        dist_matrix = sklearn.metrics.pairwise.euclidean_distances(points, points)
        dist_matrix = dist_matrix + np.eye(dist_matrix.shape[0])*dist_matrix.max()
        min_distance = dist_matrix.min()
    else:
        min_distance = float('inf')  # Initialize with a very large value
        points = [(points[i, 0], points[i, 1]) for i in range(points.shape[0])]
        for pair in itertools.combinations(points, 2):
            distance = np.sqrt((pair[0][0] - pair[1][0]) ** 2 + (pair[0][1] - pair[1][1]) ** 2)
            min_distance = min(min_distance, distance)
    print(f"The minimum distance between any two points is {min_distance:.2f}")
    return min_distance