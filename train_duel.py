from env import CoinEnv
import cv2
import time
import math
import random
import torch
import numpy as np
from pathlib import Path

from env.player import Player
from env.render import render
from utils import ReplayBuffer, save_model
import argparse
from torch.utils.tensorboard import SummaryWriter



device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")



def train(q_net=None, target_q_net=None, replay_buffer=None, device=None,  optimizer = None, gamma=0.99):

    assert device is not None, "None Device input: device should be selected."

    # Get batch from replay buffer
    samples = replay_buffer.sample()
    
    states = torch.FloatTensor(samples["obs"]).to(device)
    actions = torch.LongTensor(samples["acts"].reshape(-1,1)).to(device)
    rewards = torch.FloatTensor(samples["rews"].reshape(-1,1)).to(device)
    next_states = torch.FloatTensor(samples["next_obs"]).to(device)
    dones = torch.FloatTensor(samples["done"].reshape(-1,1)).to(device)

    # Define loss
    argmax_a = q_net(next_states).argmax(dim=1).unsqueeze(-1)
    q_target_max  = target_q_net(next_states).gather(1, argmax_a)
    # q_target_max = target_q_net(next_states).max(1)[0].unsqueeze(1).detach()
    targets = rewards + gamma*q_target_max*dones


    q_out = q_net(states)
    q_a = q_out.gather(1, actions)

    # Multiply Importance Sampling weights to loss        
    # loss = torch.nn.functional.smooth_l1_loss(q_a, targets)
    loss = torch.nn.functional.mse_loss(q_a, targets)

    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

class Dual_Q_Net(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super(Dual_Q_Net, self).__init__()
        self.action_space = action_space

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(state_space, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU()
        )

        self.V = torch.nn.Linear(64, 1)
        self.A = torch.nn.Linear(64, action_space)

    def forward(self, x):
        s = self.layers(x)
        Adv = self.A(s)
        V = self.V(s)
        Q = V + (Adv - torch.mean(Adv, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        return Q
    
    def sample_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_space-1)
        else:
            y = self.forward(obs)
            return y.argmax().item()





if __name__ == "__main__":
    torch.manual_seed(0)

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    # Create Output Dir
    exp_name = time.strftime("%Y%m%d-%H%M%S") + "_exp_duel"
    output_dir = Path("./output").joinpath(exp_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Environment
    env = CoinEnv()

    player = Player()
        
    # Try to Train
    # Set parameters
    batch_size = 64
    learning_rate = 1e-3
    buffer_len = int(100000)
    min_buffer_len = batch_size*100
    episodes = 5000
    print_per_iter = 100
    target_update_period = 1000
    eps_start = 0.9
    eps_end = 0.001
    eps_decay = 0.995
    tau = 1e-2
    max_step = 100

    # Create Q functions
    # state_space = env.column*env.row



    Q = Dual_Q_Net(state_space=player.state_space,  action_space=4).to(device)

    if args.resume:
        print("Load Checkpoint : ", args.resume)
        Q.load_state_dict(torch.load(args.resume))

    Q_target = Dual_Q_Net(state_space=player.state_space,  action_space=4).to(device)    
    Q_target.load_state_dict(Q.state_dict())
    for p in Q_target.parameters(): p.requires_grad = False

    optimizer = torch.optim.Adam(Q.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[1500], gamma=0.001)

    # Create Replay buffer
    replay_buffer = ReplayBuffer(player.state_space, size=buffer_len, batch_size=batch_size)

    # Start Training
    epsilon = eps_start



    writer = SummaryWriter(log_dir=output_dir)
    
    for i in range(episodes):
        done = False
        
        space, position_index = env.reset()
        player.initialize(0, env.column, env.row)        
        state = player.preprocess(space, position_index) # space : col * row + 1

        Q.train()
        Q_target.train()
        for t in range(max_step):

            # Take Action
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = Q.sample_action(state_tensor, epsilon)

            # Next Step
            space, reward, done, position_index = env.step(action)            
            state_prime = player.preprocess(space, position_index)

            # Check Done State
            done = (t >= max_step) or done
            done_mask = 0.0 if done else 1.0
            
            replay_buffer.put(np.array(state), action, reward, np.array(state_prime), done_mask)

            # Update State
            state = state_prime

            loss = 0

            if len(replay_buffer) >= min_buffer_len:
                loss = train(Q, Q_target, replay_buffer, device, optimizer=optimizer)
                # scheduler.step()

                if (t+1) % target_update_period == 0:
                    # Q_target.load_state_dict(Q.state_dict()) <- naive update
                    for target_param, local_param in zip(Q_target.parameters(), Q.parameters()): #<- soft update
                            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
                
            if done:
                break
        
            # Animation
            if args.render:
                screen = env.render()
                cv2.imshow("render", screen)
                cv2.waitKey(1)
            # time.sleep(0.1)

        # Output
        writer.add_scalar("Score", env.score, i)
        writer.add_scalar("Reward", env.reward, i)
        writer.add_scalar("Loss", loss, i)
        print(f"episode {i}, score {env.score}, reward {env.reward} ")


        epsilon = max(eps_end, epsilon * eps_decay) #Linear annealing

        # TODO : Save
        
        if i % print_per_iter == 0 and i!=0 or i == episodes-1:
            save_path = output_dir.joinpath(f"eps_{i}.pth")
            save_model(Q, save_path)

            