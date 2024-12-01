from env import CoinEnv
import cv2
import time
import math
import importlib
import random
import torch
import json
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

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(state_space, 256),            
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_space)
        )
    def forward(self, x):        
        y = self.layers(x)
        return y    

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
    parser.add_argument("--batch_size", type=int, default = 64)
    parser.add_argument("--learning_rate", type=float, default = 1e-3) 
    parser.add_argument("--buffer_len", type=int, default = 100000)
    parser.add_argument("--min_buffer_len", type=int, default = 64*100)
    parser.add_argument("--episodes", type=int, default=5000 )
    parser.add_argument("--print_per_iter", type=int, default=500)
    parser.add_argument("--eps_start", type=float, default=0.99)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay", type=float, default=0.001)
    parser.add_argument("--max_step", type=int, default=300)
    parser.add_argument("--target_update_period", type=int, default=10000) 
    parser.add_argument("--tau", type=float, default=1e-2)
    parser.add_argument("--player", type=str, default="player")
    args = parser.parse_args()

    # Create Output Dir
    exp_name = time.strftime("%Y%m%d-%H%M%S") + "_exp"
    output_dir = Path("./output").joinpath(exp_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Environment
    env = CoinEnv()


    playermodule = importlib.import_module(f"env.{args.player}")
    player = playermodule.Player()    

    
    

    writer = SummaryWriter(log_dir=output_dir, comment=exp_name)
    with open(output_dir.joinpath("args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)


    # Create Q functions
    Q = Q_net(state_space=player.state_space,  action_space=4).to(device)

    if args.resume:
        print("Load Checkpoint : ", args.resume)
        Q.load_state_dict(torch.load(args.resume))

    Q_target = Q_net(state_space=player.state_space,  action_space=4).to(device)
    
    Q_target.load_state_dict(Q.state_dict())
    optimizer = torch.optim.Adam(Q.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[1000, 1500, 2500], gamma=0.1)

    # Create Replay buffer
    replay_buffer = ReplayBuffer(player.state_space, size=args.buffer_len, batch_size=args.batch_size)

    # Start Training
    epsilon = args.eps_start

    writer = SummaryWriter(log_dir=output_dir)
    
    step = 0
    for i in range(args.episodes):
        done = False
        
        state = env.reset(player=player)

        Q.train()
        Q_target.train()

        loss = 0
        for t in range(args.max_step):
            step += 1
            # Take Action
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action = Q.sample_action(state_tensor, epsilon)

            # Next Step
            state_prime, reward, done = env.step(action)            

            # Check Done State
            done = (t >= args.max_step) or done
            done_mask = 0.0 if done else 1.0
            
            replay_buffer.put(np.array(state), action, reward/100.0, np.array(state_prime), done_mask)

            # Update State
            state = state_prime
            
            if len(replay_buffer) >= args.min_buffer_len:
                loss += train(Q, Q_target, replay_buffer, device, optimizer=optimizer)
                # scheduler.step()

                if step % args.target_update_period == 0:                    
                    for target_param, local_param in zip(Q_target.parameters(), Q.parameters()): #<- soft update
                            target_param.data.copy_(args.tau*local_param.data + (1.0 - args.tau)*target_param.data)
                
            if done:
                break
        
            # Animation
            if args.render:
                screen = env.render()
                cv2.imshow("render", screen)
                cv2.waitKey(1)
            # time.sleep(0.1)

        # Output
        writer.add_scalar("output/Score", env.score, i)
        writer.add_scalar("output/Reward", env.reward, i)
        writer.add_scalar("train/Loss", loss, i)
        writer.add_scalar("train/Epsilon", epsilon, i)
        print(f"{exp_name} : episode {i}, score {env.score}, reward {env.reward}")

        epsilon = max(args.eps_end, epsilon * (1.0-args.eps_decay)) #Linear annealing

        # Save
        if i % (args.print_per_iter+1) == 0 and i!=0 or i == args.episodes-1:
            save_path = output_dir.joinpath(f"eps_{i+1}.pth")
            save_model(Q, save_path)

            