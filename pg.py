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

import models
from env.player import Player
from env.render import render
from utils import ReplayBuffer, train, save_model
import argparse
from torch.utils.tensorboard import SummaryWriter



device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")



def update_policy(policy, optimizer):
    R = 0
    rewards = []
    
    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0,R)


        
    # Scale rewards
    rewards = torch.FloatTensor(rewards)

    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    
    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, torch.autograd.Variable(rewards)).mul(-1), -1))
    
    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    #Save and intialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = torch.autograd.Variable(torch.Tensor())
    policy.reward_episode= []

    return loss.item()

if __name__ == "__main__":
    torch.manual_seed(0)

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch_size", type=int, default = 64)
    parser.add_argument("--learning_rate", type=float, default = 1e-2)     
    parser.add_argument("--episodes", type=int, default=5000 )
    parser.add_argument("--print_per_iter", type=int, default=500)    
    parser.add_argument("--max_step", type=int, default=300)    
    parser.add_argument("--tau", type=float, default=1e-2)
    parser.add_argument("--player", type=str, default="player_heatmap_nn")
    parser.add_argument("--model", type=str, default="Policy")
    args = parser.parse_args()

    # Create Output Dir
    exp_name = time.strftime("%Y%m%d-%H%M%S") + "_pg"
    output_dir = Path("./output").joinpath(exp_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create Environment
    env = CoinEnv()


    playermodule = importlib.import_module(f"env.{args.player}")
    player = playermodule.Player()    

    writer = SummaryWriter(log_dir=output_dir, comment=exp_name)
    with open(output_dir.joinpath("args.json"), "w") as f:
        argsjson = {}
        for key,val in vars(args).items():
            argsjson[key] = str(val)
        json.dump(argsjson, f, indent=4)


    # Create Q functions
    Policy = getattr(models, args.model)
    policy = Policy(state_space=player.state_space,  action_space=4).to(device)

    if args.resume:
        print("Load Checkpoint : ", args.resume)
        policy.load_state_dict(torch.load(args.resume))
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    

    writer = SummaryWriter(log_dir=output_dir)
    
    step = 0
    running_reward = 10

    for i in range(args.episodes):
        done = False
        
        state = env.reset(player=player)

        policy.train()        

        loss = 0
        for t in range(args.max_step):
            step += 1
            # Take Action
            state_tensor = torch.tensor(state, dtype=torch.float32)
            prop = policy(state_tensor)
            c = torch.distributions.Categorical(prop)
            action = c.sample()


            # Add log probability of our chosen action to our history    
            if policy.policy_history.dim() != 0:
                policy.policy_history = torch.cat([policy.policy_history, c.log_prob(action).unsqueeze(0)])
            else:
                policy.policy_history = (c.log_prob(action))

            # Next Step
            state, reward, done = env.step(action)            
            done = (t >= args.max_step) or done
            
            # Append Reward
            policy.reward_episode.append(reward)
            
            if done:
                break
        
            # Animation
            if args.render:
                screen = env.render()
                cv2.imshow("render", screen)
                cv2.waitKey(1)
            # time.sleep(0.1)

        # Used to determine when the environment is solved.
        running_reward = (running_reward * 0.99) + (t * 0.01)
        loss = update_policy(policy, optimizer)

        # Output
        writer.add_scalar("output/Score", env.score, i)
        writer.add_scalar("output/Reward", env.reward, i)
        writer.add_scalar("train/Loss", loss, i)        
        print(f"{exp_name} : episode {i}, score {env.score}, reward {env.reward}")

        # Save
        if i % (args.print_per_iter+1) == 0 and i!=0 or i == args.episodes-1:
            save_path = output_dir.joinpath(f"eps_{i+1}.pth")
            save_model(policy, save_path)

            