import torch
from src.RL.models import GNN_Encoder,DDPG_Agent
from src.RL.environment import ManhattanTrafficEnv
from src.RL.train import train
from torch_geometric.data import Data
import numpy as np
import os

if __name__ == "__main__":
    environment = ManhattanTrafficEnv()
    
    # Setup dimensions
    state_dim = 32  # Assume output dimension from GNN
    action_dim = environment.action_space.shape[0]
    max_action = float(environment.action_space.high[0])

    # Initialize models
    gnn_encoder = GNN_Encoder(num_features=4, hidden_dim=64, output_dim=state_dim)
    ddpg_agent = DDPG_Agent(state_dim=state_dim, action_dim=action_dim)
    # actor = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    # critic = Critic(state_dim=state_dim, action_dim=action_dim)

    models = (gnn_encoder,ddpg_agent)

    # Setup optimizer
    optimizer = torch.optim.Adam(
        list(gnn_encoder.parameters()) + list(actor.parameters()) + list(critic.parameters()), lr=1e-4
    )

    # Train the model
    train(models, environment, epochs=100)
    # train(gnn_encoder, actor, critic, environment, epochs=100, optimizer=optimizer)
