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
from utils import ReplayBuffer, train, save_model
import argparse
from torch.utils.tensorboard import SummaryWriter



device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

class Q_net(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super(Q_net, self).__init__()

        self.action_space = action_space

        # NEtwork 
        self.Linear1 = torch.nn.Linear(state_space, 256)        
        self.Linear2 = torch.nn.Linear(256, 128)
        self.Linear3 = torch.nn.Linear(128, 64)
        self.Linear4 = torch.nn.Linear(64, action_space)

    def forward(self, x):        

        x = torch.nn.functional.relu(self.Linear1(x))
        x = torch.nn.functional.relu(self.Linear2(x))
        x = torch.nn.functional.relu(self.Linear3(x))

        y = self.Linear4(x)
        return y    

    def sample_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_space-1)
        else:
            y = self.forward(obs)
            return y.argmax().item()



if __name__ == "__main__":

    torch.random.manual_seed(0)

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # Create Output Dir
    exp_name = time.strftime("%Y%m%d-%H%M%S") + "_exp"
    output_dir = Path("./output").joinpath(exp_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Environment
    env = CoinEnv()

    player = Player()
    player.initialize(0, env.column, env.row)

    # Prepare Rendering
    if args.render:
        cv2.namedWindow("render", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("render", env.column*20, env.row*20)
        # cv2.namedWindow("player", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("player", 128, 128)
        
    # Try to Train
    # Set parameters
    batch_size = 64
    learning_rate = 1e-3
    buffer_len = int(100000)
    min_buffer_len = batch_size
    episodes = 1000
    print_per_iter = 100
    target_update_period = 4
    eps_start = 0.9
    eps_end = 0.001
    eps_decay = 0.995
    tau = 1*1e-2
    max_step = 2000

    # Create Q functions
    # state_space = env.column*env.row

    # player.preprocess = 9x9 subgrid method, state space 81
    # preprocess_append_position = append position, state space 12*20+1
    preprocess_method = player.preprocess
    
    state_space = player._sight * player._sight


    Q = Q_net(state_space=state_space,  action_space=4).to(device)
    Q_target = Q_net(state_space=state_space,  action_space=4).to(device)
    Q_target.load_state_dict(Q.state_dict())
    optimizer = torch.optim.Adam(Q.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[450, 500, 600 ,700, 800, 900], gamma=0.3)

    # Create Replay buffer
    replay_buffer = ReplayBuffer(state_space, size=buffer_len, batch_size=batch_size)

    # Start Training
    epsilon = eps_start



    writer = SummaryWriter(log_dir=output_dir)
    
    for i in range(episodes):
        space, position_index = env.reset()        
        done = False


        state = preprocess_method(space, position_index) # space : col * row + 1

        Q.train()
        Q_target.train()
        for t in range(max_step):
            
            # view = np.array(view).reshape(player._sight, player._sight)
            # player_view = render(view, [4,4])
            # cv2.imshow("player", player_view)

            # Take Action

            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = Q.sample_action(state_tensor, epsilon)


            # Next Step
            space, reward, done, position_index = env.step(action)
            state_prime = preprocess_method(space, position_index)
            
            
            done_mask = 0.0 if done else 1.0
            replay_buffer.put(np.array(state), action, reward/100.0, np.array(state_prime), done_mask)

            # Update State
            state = state_prime

            if len(replay_buffer) >= min_buffer_len:
                train(Q, Q_target, replay_buffer, device, optimizer=optimizer)
                scheduler.step()

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
        writer.add_scalar("Epsilon", epsilon, i)
        print(f"episode {i}, score {env.score}, epsilon {epsilon} ")


        epsilon = max(eps_end, epsilon * eps_decay) #Linear annealing

        # TODO : Save
        
        if i % print_per_iter == 0 and i!=0 or i == episodes-1:
            save_path = output_dir.joinpath(f"eps_{i}.pth")
            save_model(Q, save_path)

            