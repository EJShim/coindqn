import argparse
from env import CoinEnv
import numpy as np
from itertools import count
from collections import namedtuple
from env.player_heatmap_normalize import Player

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Cart Pole

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N', help='random seed (default: 543)')
parser.add_argument('--render', action='store_true', help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='interval between training status logs (default: 10)')
args = parser.parse_args()



player = Player()
env = CoinEnv()
env.reset(player=player)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self):
        super(Policy, self).__init__()

        state_space = 81
        action_space = 4

        self.layers = torch.nn.Sequential(
            torch.nn.Linear(state_space, 256),                    
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),            
            torch.nn.ReLU(),            
        )

        # actor's layer
        self.action_head = nn.Linear(128, 4)

        # critic's layer
        self.value_head = nn.Linear(128, 1) 

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        x = self.layers(x)

        # actor: choses action to take from state s_t
        # by returning probability of each action
        action_prob = F.softmax(self.action_head(x), dim=-1)

        # critic: evaluates being in the state s_t
        state_values = self.value_head(x)

        # return values for both actor and critic as a tuple of 2 values:
        # 1. a list with the probability of each action over the action space
        # 2. the value from state s_t
        return action_prob, state_values


model = Policy()
optimizer = optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999))
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.tensor(state).float()
    probs, state_value = model(state)

    # create a categorical distribution over the list of probabilities of actions
    m = Categorical(probs)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.item()


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = [] # list to save actor (policy) loss
    value_losses = [] # list to save critic (value) loss
    returns = [] # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # calculate the discounted value
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()

        # calculate actor (policy) loss
        policy_losses.append(-log_prob * advantage)

        # calculate critic (value) loss using L1 smooth loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    # reset gradients
    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]

    return loss.item()


def main():
    running_reward = 10

    # run infinitely many episodes
    for i_episode in count(1):

        # reset environment and episode reward
        state = env.reset(player=player)
        ep_reward = 0        

        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 100):

            # select action from policy
            action = select_action(state)

            # take the action
            state, reward, done = env.step(action)
            # if reward < 0 : done = True

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward            
            if done:
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        loss =finish_episode()

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\t Last Score : {:.2f}\t Loss: {:.2f}\tAverage reward: {:.2f}'.format( i_episode, env.score, loss, running_reward))

        # check if we have "solved" the cart pole problem
        if running_reward > 70000:
            print("Solved! Running reward is now {} and the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()