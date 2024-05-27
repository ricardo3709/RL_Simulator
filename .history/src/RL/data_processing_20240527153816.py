import torch
import numpy as np
import pickle
import pandas as pd
from src.simulator.config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def feature_preparation(state):
    node_lookup_table = pd.read_csv(PATH_MANHATTAN_NODES_LOOKUP_TABLE)
    area_ids = node_lookup_table['zone_id'].unique()

    [avaliable_veh_nodes, working_veh_nodes, gen_counts_nodes, rej_counts_nodes, attraction_counts_nodes] = state

    request_gen_counts = [sum(gen_counts_nodes[i]) for i in range(1,len(gen_counts_nodes)+1)]
    request_rej_counts = [sum(rej_counts_nodes[i]) for i in range(1,len(rej_counts_nodes)+1)]
    request_attraction_counts = [sum(attraction_counts_nodes[i]) for i in range(1,len(attraction_counts_nodes)+1)]

    avaliable_veh_areas = np.zeros((63,2))
    avaliable_veh_areas[:,0] = area_ids
    working_veh_areas = np.zeros((63,2))
    working_veh_areas[:,0] = area_ids
    request_gen_counts_areas = np.zeros((63,2))
    request_gen_counts_areas[:,0] = area_ids
    request_rej_counts_areas = np.zeros((63,2))
    request_rej_counts_areas[:,0] = area_ids
    request_attraction_counts_areas = np.zeros((63,2))
    request_attraction_counts_areas[:,0] = area_ids

    for node_id, counts in enumerate(avaliable_veh_nodes):
        area_id = node_lookup_table[node_lookup_table['node_id'] == node_id]['zone_id'].values[0]
        avaliable_veh_areas[area_id][1] += counts
    
    for node_id, counts in enumerate(working_veh_nodes):
        area_id = node_lookup_table[node_lookup_table['node_id'] == node_id]['zone_id'].values[0]
        working_veh_areas[area_id][1] += counts
    
    for node_id, counts in enumerate(request_gen_counts):
        area_id = node_lookup_table[node_lookup_table['node_id'] == node_id]['zone_id'].values[0]
        request_gen_counts_areas[area_id][1] += counts
    
    for node_id, counts in enumerate(request_rej_counts):
        area_id = node_lookup_table[node_lookup_table['node_id'] == node_id]['zone_id'].values[0]
        request_rej_counts_areas[area_id][1] += counts
    
    for node_id, counts in enumerate(request_attraction_counts):
        area_id = node_lookup_table[node_lookup_table['node_id'] == node_id]['zone_id'].values[0]
        request_attraction_counts_areas[area_id][1] += counts

    avaliable_veh_areas_tensor = torch.tensor(avaliable_veh_areas, dtype=torch.float, device=device)  # [63, 2]
    working_veh_areas_tensor = torch.tensor(working_veh_areas[:,1], dtype=torch.float, device=device)  # [63]
    request_gen_counts_areas_tensor = torch.tensor(request_gen_counts_areas[:,1], dtype=torch.float, device=device)  # [63]
    request_rej_counts_areas_tensor = torch.tensor(request_rej_counts_areas[:,1], dtype=torch.float, device=device)  # [63]
    request_attraction_counts_areas_tensor = torch.tensor(request_attraction_counts_areas[:,1], dtype=torch.float, device=device)  # [63]


    # combine all features, [63, 6]
    features = torch.stack((avaliable_veh_areas_tensor, working_veh_areas_tensor, request_gen_counts_areas_tensor, request_rej_counts_areas_tensor, request_attraction_counts_areas_tensor), dim=1)

    return features