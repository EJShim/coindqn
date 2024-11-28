from env import CoinEnv
import cv2
import time
import math
import random
import numpy as np
import torch

from env.player import Player
from env.render import render
from utils import ReplayBuffer, train



device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

class Q_net(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super(Q_net, self).__init__()

        self.action_space = action_space

        # NEtwork 
        self.Linear1 = torch.nn.Linear(state_space, 64)        
        self.Linear2 = torch.nn.Linear(64, 128)
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

    # Create Environment
    env = CoinEnv()


    cv2.namedWindow("render", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("render", env.column*20, env.row*20)

    # CLO Player
    cv2.namedWindow("player", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("player", 128, 128)

    player = Player()
    player.initialize(0, env.column, env.row)



    # Try to Train
    # Set parameters
    batch_size = 64
    learning_rate = 1e-3
    buffer_len = int(100000)
    min_buffer_len = batch_size
    episodes = 650
    print_per_iter = 20
    target_update_period = 4
    eps_start = 0.9
    eps_end = 0.001
    eps_decay = 0.995
    tau = 1*1e-2
    max_step = 2000

    # Create Q functions
    state_space = env.column*env.row
    Q = Q_net(state_space=state_space,  action_space=4).to(device)
    Q_target = Q_net(state_space=state_space,  action_space=4).to(device)
    Q_target.load_state_dict(Q.state_dict())
    optimizer = torch.optim.Adam(Q.parameters(), lr=learning_rate)

    # Create Replay buffer
    replay_buffer = ReplayBuffer(state_space, size=buffer_len, batch_size=batch_size)

    # Start Training
    epsilon = eps_start

    for i in range(episodes):
        state, position = env.reset()
        done = False

        for t in range(max_step):
            # Show Status
            screen = env.render()
            cv2.imshow("render", screen)

            # view = player.preprocess(state, env.convert_index(env.position))
            # view = np.array(view).reshape(player._sight, player._sight)
            # player_view = render(view, [4,4])
            # cv2.imshow("player", player_view)

            # Take Action
            # action = player.move_next( state, position )
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = Q.sample_action(state_tensor, epsilon)


            state_prime, reward, done = env.step(action)
            state = state_prime

            done_mask = 0.0 if done else 1.0
            replay_buffer.put(state, action, reward/100.0, state_prime, done_mask)


            if len(replay_buffer) >= min_buffer_len:
                train(Q, Q_target, replay_buffer, device, optimizer=optimizer)

                if (t+1) % target_update_period == 0:
                    # Q_target.load_state_dict(Q.state_dict()) <- naive update
                    for target_param, local_param in zip(Q_target.parameters(), Q.parameters()): #<- soft update
                            target_param.data.copy_(tau*local_param.data + (1.0 - tau)*target_param.data)
                
            if done:
                break
        
            # Animation
            cv2.waitKey(1)
            time.sleep(0.1)

        # Output
        print(f"episode {i}, score {env.score}, epsilon {epsilon} ")
        epsilon = max(eps_end, epsilon * eps_decay) #Linear annealing

        # TODO : Save

            