import torch
from src.RL.models import GNN_Encoder, Actor, Critic
from src.RL.environment import ManhattanTrafficEnv
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import os
from src.simulator.config import *
from tqdm import tqdm

base_dir = 'saved_models'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)
def train(models, environment, epochs):
    gnn_encoder, ddpg_agent = models

    # for epoch in range(WARM_UP_EPOCHS):
    #     state, network = environment.reset()
    #     done = False
    #     warm_up_step = 0
    #     while not done:
    #         warm_up_step += 1
    #         for step in tqdm(range(int(RL_STEP_LENGTH)), desc=f"WarmUP_{warm_up_step}"): # 2.5mins, 10 steps
    #             environment.simulator.system_time += TIME_STEP
    #             environment.simulator.run_cycle() # Run one cycle(15s)
    #         done = environment.warm_up_step()

    for epoch in range(epochs):
        state, network = environment.reset()
        edge_index = graph_to_data(network)
        total_critic_loss = 0
        total_actor_loss = 0
        done = False
        steps = 0

        while not done:
            for step in tqdm(range(int(RL_STEP_LENGTH))): # 2.5mins, 10 steps
                environment.simulator.system_time += TIME_STEP
                environment.simulator.run_cycle() # Run one cycle(15s)
            # state, _ = environment.simulator.get_simulator_state()
            x = state.clone().detach().requires_grad_(True)
            # x = torch.tensor(state, dtype=torch.float) 
            graph_data = Data(x=x, edge_index=edge_index)

            state_encoded = gnn_encoder(graph_data)  # encode the state
            action = ddpg_agent.select_action(state_encoded)
            next_state, reward, done = environment.step(action)

            # Update the model, get the loss
            critic_loss, actor_loss = ddpg_agent.update_policy(state, action, reward, next_state, edge_index)

            # Logging
            total_critic_loss += critic_loss
            total_actor_loss += actor_loss
            ddpg_agent.total_steps += 1
            steps += 1
            state = next_state

        avg_critic_loss = total_critic_loss / steps
        avg_actor_loss = total_actor_loss / steps
        print(f"Epoch {epoch}: Avg Critic Loss = {avg_critic_loss}, Avg Actor Loss = {avg_actor_loss}")
    
    # Save the models
    save_models(epoch, ddpg_agent)

def graph_to_data(network):
    # Convert the network to edge_index
    if isinstance(network, pd.DataFrame):
        network_array = network.values
    else:
        network_array = network
    # only consider the non-zero elements
    # edge_index = torch.tensor(np.nonzero(network_array != 0), dtype=torch.long)
    edge_index = torch.tensor(network_array)
    return edge_index

def save_models(epoch, model):
    actor_filename = os.path.join(base_dir, f'actor_{epoch}.pth')
    critic_filename = os.path.join(base_dir, f'critic_{epoch}.pth')
    actor_target_filename = os.path.join(base_dir, f'actor_target_{epoch}.pth')
    critic_target_filename = os.path.join(base_dir, f'critic_target_{epoch}.pth')

    torch.save(model.actor.state_dict(), actor_filename)
    torch.save(model.critic.state_dict(), critic_filename)
    torch.save(model.actor_target.state_dict(), actor_target_filename)
    torch.save(model.critic_target.state_dict(), critic_target_filename)
    print("Models saved successfully")

def load_models(epoch, model):
    model.actor.load_state_dict(torch.load(f'actor_{epoch}.pth'))
    model.critic.load_state_dict(torch.load(f'critic_{epoch}.pth'))
    model.actor_target.load_state_dict(torch.load(f'actor_target_{epoch}.pth'))
    model.critic_target.load_state_dict(torch.load(f'critic_target_{epoch}.pth'))
    print("Models loaded successfully")

def warm_up_train(models, environment, epochs = WARM_UP_EPOCHS):
    gnn_encoder, ddpg_agent = models

    for epoch in range(epochs):
        state, network = environment.reset()
        done = False
        warm_up_step = 0
        while not done:
            warm_up_step += 1
            for step in tqdm(range(int(RL_STEP_LENGTH)), desc=f"WarmUP_{warm_up_step}"): # 2.5mins, 10 steps
                environment.simulator.system_time += TIME_STEP
                environment.simulator.run_cycle() # Run one cycle(15s)
            done, past_rejections = environment.warm_up_step()
    return past_rejections

if __name__ == "__main__":
    # Initialize environment
    environment = ManhattanTrafficEnv()
    
    # Setup dimensions
    state_dim = 32  # Assume output dimension from GNN
    action_dim = environment.action_space.shape[0]
    max_action = float(environment.action_space.high[0])

    # Initialize models
    gnn_encoder = GNN_Encoder(num_features=NUM_FEATURES, hidden_dim=64, output_dim=state_dim)
    actor = Actor(state_dim=state_dim, action_dim=action_dim, max_action=max_action)
    critic = Critic(state_dim=state_dim, action_dim=action_dim)

    # Setup optimizer
    optimizer = torch.optim.Adam(
        list(gnn_encoder.parameters()) + list(actor.parameters()) + list(critic.parameters()), lr=1e-4
    )

    # Train the model
    train(gnn_encoder, actor, critic, environment, epochs=100, optimizer=optimizer)
