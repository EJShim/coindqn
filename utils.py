import numpy as np
import torch
import json
from pathlib import Path
from typing import Dict

def make_2d_input_map(input_map, row, column):

    # input to 2d amp
    result = [[0] * column for _ in range(row)]

    for r in range(row):
        result[r] = input_map[r*column:r*column+20 ]

    return result

def sample_pad_2d_input_map(map2d, sight=9, position=[0,3]):
    column = len(map2d[0]) 
    row = len(map2d)
    pad = sight // 2
    result = [[-1]*(column+(pad*2)) for _ in  range(row + (pad*2)) ]
    for r in range(row):
        result[r+pad][pad:-pad] = map2d[r]

    state = [result[i][position[1]:position[1]+sight] for i in range(position[0],position[0]+sight)]

    # state = result[0][position[0]:position[0]+sight]
    return state

def index_to_position(index, row, column):
    return [ index//column, index%column]


if __name__ == "__main__":

    input_map = [0, 0, 0, 0, 0, -1, 30, 30, 30, 100, 100, 30, 30, 30, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 30, 30, 30, 30, 30, 30, 30, 30, -1, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 10, 10, 10, 10, 10, 30, 30, 30, 30, 30, 30, 30, 30, 10, 10, 10, 10, 10, 0, 0, 10, 10, 10, 10, 10, 30, -1, 100, 200, 200, 100, -1, 30, 10, 10, 10, 10, 10, 0, 200, 10, -1, -1, 10, 10, 30, -1, 100, 100, 100, 100, -1, 30, 10, 10, -1, -1, 10, 200, 200, 10, -1, -1, 10, 10, 30, -1, 100, 100, 100, 100, -1, 30, 10, 10, -1, -1, 10, 200, 0, 10, 10, 10, 10, 10, 30, -1, 100, 200, 200, 100, -1, 30, 10, 10, 10, 10, 10, 0, 0, 10, 10, 10, 10, 10, 30, 30, 30, 30, 30, 30, 30, 30, 10, 10, 10, 10, 10, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, -1, 30, 30, 30, 30, 30, 30, 30, 30, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 30, 30, 30, 100, 100, 30, 30, 30, -1, 0, 0, 0, 0, 0]

    row = 12
    column = 20
    


    for position_index in range(row*column):
        
        position = index_to_position(position_index, row, column)        
        map2d = make_2d_input_map(input_map, row, column)
        padded = sample_pad_2d_input_map(map2d, position=position)

        # # print(padded)
        print(np.array(padded).shape)
        # print(np.array(padded).shape)
    

    position_index = 19
    position = index_to_position(position_index, row, column)        
    map2d = make_2d_input_map(input_map, row, column)
    padded = sample_pad_2d_input_map(map2d, position=position)

    # # print(padded)
    print(np.array(padded))



class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(self, obs_dim: int, size: int, batch_size: int = 32):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

    def put(
        self,
        obs: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

    def __len__(self) -> int:
        return self.size


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
    q_target_max = target_q_net(next_states).max(1)[0].unsqueeze(1).detach()
    targets = rewards + gamma*q_target_max*dones
    q_out = q_net(states)
    q_a = q_out.gather(1, actions)

    # Multiply Importance Sampling weights to loss        
    loss = torch.nn.functional.smooth_l1_loss(q_a, targets)

    # Update Network
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def save_model(model, save_path):

    state_dict = model.state_dict()
    torch.save(state_dict, save_path)

    # Save Json Also
    json_dict = {}
    for key, value in state_dict.items():
        json_dict[key] = value.detach().cpu().numpy().tolist()

    json_file = save_path.with_suffix(".json")
    with open(json_file, "w") as f:
        json.dump(json_dict, f)