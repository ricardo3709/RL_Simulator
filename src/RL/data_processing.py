import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def feature_preparation(state):
    [idle_cars, rebalancing_cars, request_gen_deque, request_rej_deque] = state

    request_gen_counts = [sum(request_gen_deque[i]) for i in range(1,len(request_gen_deque)+1)]
    request_rej_counts = [sum(request_rej_deque[i]) for i in range(1,len(request_rej_deque)+1)]

    idle_cars = torch.tensor(idle_cars, dtype=torch.float, device=device)  # [4091]
    rebalancing_cars = torch.tensor(rebalancing_cars, dtype=torch.float, device=device)  # [4091]
    request_gen_counts = torch.tensor(request_gen_counts, dtype=torch.float, device=device)  # [4091]
    request_rej_counts = torch.tensor(request_rej_counts, dtype=torch.float, device=device)  # [4091]

    # adjacency_matrix = np.array(adjacency_matrix)
    # edges = np.nonzero(adjacency_matrix)
    # edge_index = torch.tensor(edges, dtype=torch.long)

    # combine all features
    features = torch.stack((idle_cars, rebalancing_cars, request_gen_counts, request_rej_counts), dim=1)

    return features