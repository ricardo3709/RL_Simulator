import torch
import numpy as np

def feature_preparation(state):
    [idle_cars, rebalancing_cars, request_gen_counts, request_rej_counts] = state

    request_gen_counts = [sum(request_gen_counts[i]) for i in range(len(request_gen_counts))]
    request_rej_counts = [sum(request_rej_counts[i]) for i in range(len(request_rej_counts))]
    idle_cars = torch.tensor(idle_cars)  # [4091]
    rebalancing_cars = torch.tensor(rebalancing_cars)  # [4091]
    request_gen_counts = torch.tensor(request_gen_counts)  # [4091]
    request_rej_counts = torch.tensor(request_rej_counts)  # [4091]

    # adjacency_matrix = np.array(adjacency_matrix)
    # edges = np.nonzero(adjacency_matrix)
    # edge_index = torch.tensor(edges, dtype=torch.long)

    # combine all features
    features = torch.stack((idle_cars, rebalancing_cars, request_gen_counts, request_rej_counts), dim=1)

    return features