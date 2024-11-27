import os
import socket
import json
import torch
import subprocess

import win32com.client
from train_test import ReplayBuffer, train
import win32com
import time
import random
from pathlib import Path
import argparse

wsh = win32com.client.Dispatch("WScript.Shell")

def save_model(model, path='default.pth'):

    save_path = Path(path)
    if not save_path.parent.exists():
        save_path.parent.mkdir(exist_ok=True)

    state_dict = model.state_dict()
    torch.save(state_dict, path)

    # Save Json Also
    json_dict = {}
    for key, value in state_dict.items():
        json_dict[key] = value.detach().cpu().numpy().tolist()

    json_file = path[:-4] + ".json"
    with open(json_file, "w") as f:
        json.dump(json_dict, f)

class Q_net(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super(Q_net, self).__init__()

        self.action_space = action_space


        self.Linear1 = torch.nn.Linear(state_space, 64)        
        self.Linear2 = torch.nn.Linear(64, 128)
        self.Linear3 = torch.nn.Linear(128, 64)
        self.Linear4 = torch.nn.Linear(64, action_space)

    def forward(self, x):
        x = torch.nn.functional.relu(self.Linear1(x))
        x = torch.nn.functional.relu(self.Linear2(x))
        x = torch.nn.functional.relu(self.Linear3(x))
        return self.Linear4(x)

    def sample_action(self, obs, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_space-1)
        else:
            return self.forward(obs).argmax().item()


device = torch.device("cpu")
class SocketAgent:
    def __init__(self, process = None, resume=None):
        
        # Training Parameters : 
        self.eps_start = 0.1
        self.eps_end = 0.001
        self.eps_decay = 0.995

        self.resume = resume


        self.epsilon = self.eps_start


        self.SIZE = 1024
        IP = ''
        PORT = 5050
        ADDR = (IP, PORT)

        print("run Serer")

        self.q_model = None
        self.q_target = None
        self.replay_buffer = None

        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.server_socket.bind(ADDR)
        self.server_socket.listen()

        # For Reward Calculation
        self.prev_state = None
        self.action = None
        self.total_rewards = 0


        self.t = 0
        self.batch_size=64
        self.target_update_period = 4
        self.tau = 1*1e-2

        self.episode = 0


        self.process = process

        seed = str(random.randint(1,4))
        time.sleep(5)
        wsh.AppActivate(self.process.pid)
        wsh.SendKeys(seed)
        print(f"Send Key {seed}")


    def reset(self):



        # For Reward Calculation
        self.t = 0
        self.prev_state = None
        self.action = None
        self.total_rewards = 0

        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay) 
    

    def run(self):        
        # 무한루프 진입
        while True:
            client_socket, client_addr = self.server_socket.accept()  
            msg = client_socket.recv(self.SIZE)  # 클라이언트가 보낸 메시지 
            data = json.loads(msg.decode())
            if data['type'] =='start':
                print("Start Process")

                time.sleep(3)
                wsh.AppActivate(self.process.pid)
                wsh.SendKeys("~")
                
                print("Enter Key Dnoe")

            elif data['type'] == 'init':
                sight = data['sight']

                self.reset()
                if self.q_model == None:
                    self.q_model = Q_net(state_space=sight*sight, action_space=4)
                    if self.resume is not None:
                        print("Load model : ", self.resume)
                        self.q_model.load_state_dict(torch.load(self.resume))

                    self.q_target = Q_net(state_space=sight*sight,  action_space=4)
                    self.q_target.load_state_dict(self.q_model.state_dict())
                    self.replay_buffer = ReplayBuffer(sight*sight,
                                                        size=100000,
                                                        batch_size=self.batch_size )
                    
                    self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=1e-3)


            elif data['type'] == 'state':

                state = torch.tensor(data['state'])
                # print(state.shape)
                state = torch.flatten(state).to(torch.float32)
                reward = data["reward"]


                # print(asdf)
                # exit()
                # state = torch.tensor( data['map'] , dtype=torch.float32)
                # position = torch.tensor( [data['position']]*col*row )             
                # state_tensor  = torch.cat([state, position])
                # reward = self.calculate_reward(state_tensor, data['position'] )
                
                done_mask = 0.0 if 1 else 1.0

                if self.prev_state is not None:
                    self.replay_buffer.put(self.prev_state, 
                                            self.action, 
                                            reward/100.0,
                                            state,
                                            done_mask )
                    
                    if len(self.replay_buffer) >= self.batch_size:
                        train(self.q_model, self.q_target, 
                            self.replay_buffer, device, 
                                optimizer=self.optimizer)

                        if (self.t+1) % self.target_update_period == 0:
                            # Q_target.load_state_dict(Q.state_dict()) <- naive update
                            for target_param, local_param in zip(self.q_target.parameters(), self.q_model.parameters()): #<- soft update
                                    target_param.data.copy_(self.tau*local_param.data + (1.0 - self.tau)*target_param.data)
                
                self.prev_state = state

                # Run Network
                self.action = self.q_model.sample_action( state, self.epsilon )
                self.total_rewards += reward

                self.t += 1

                client_socket.send( str(self.action).encode() )

                if data["done"]:     
                    print("Done", self.epsilon, self.total_rewards)
                    env_name = "V2"
                    if self.episode % 20 == 0:
                        save_path = f"output/{env_name}/{self.episode}.pth"
                        save_model(self.q_model, save_path )

                    self.episode += 1
                    self.reset()
                    
        


                    subprocess.call(['taskkill', '/F', '/T', '/PID',  str(self.process.pid)])


                    cwd = os.path.realpath("play_windows_20x12")
                    self.process = subprocess.Popen("CoinChallenger.exe ", cwd=cwd, shell=True, stdin=subprocess.PIPE)
                    time.sleep(5)
                    seed = str(random.randint(1,4))
                    wsh.AppActivate(self.process.pid)
                    wsh.SendKeys(seed)
                    print(f"Send Key {seed}")
            else:
                print(data)
                if data['type'] == "error" : break





                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    cwd = os.path.realpath("play_windows_20x12")
    p = subprocess.Popen("CoinChallenger.exe ", cwd=cwd, shell=True, stdin=subprocess.PIPE)
    
    
    agent = SocketAgent(p, args.resume)
    agent.run()


