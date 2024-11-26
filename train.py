import os
import socket
import json
import torch
import subprocess
from train_test import Q_net, ReplayBuffer, train

device = torch.device("cpu")
class SocketAgent:
    def __init__(self, process = None):
        self.process = process
        
        # Training Parameters : 
        self.eps_start = 0.9
        self.eps_end = 0.001
        self.eps_decay = 0.5


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


    def reset(self):
        # For Reward Calculation
        self.t = 0
        self.prev_state = None
        self.action = None
        self.total_rewards = 0

        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay) #Linear annealing1q




    def calculate_reward(self, state, position):
        if self.prev_state == None : return torch.tensor(0, dtype=torch.float32)

        if position == state[-1]:
            # print("not moved")
            return torch.tensor(0, dtype=torch.float32)

        reward = self.prev_state[position] - state[position]


        return reward
    

    def run(self):        
        # 무한루프 진입
        while True:
            client_socket, client_addr = self.server_socket.accept()  
            msg = client_socket.recv(self.SIZE)  # 클라이언트가 보낸 메시지 
            data = json.loads(msg.decode())
            if data['type'] =='start':
                print("Start Process", dir(self.process))

            elif data['type'] == 'init':
                col = data['col']
                row = data['row']

                self.reset()
                if self.q_model == None:
                    self.q_model = Q_net(state_space=col*row*2, action_space=4)
                    self.q_target = Q_net(state_space=col*row*2,  action_space=4)
                    self.q_target.load_state_dict(self.q_model.state_dict())
                    self.replay_buffer = ReplayBuffer(col*row*2,
                                                        size=100000,
                                                        batch_size=self.batch_size )
                    
                    self.optimizer = torch.optim.Adam(self.q_model.parameters(), lr=1e-3)


            elif data['type'] == 'state':
                state = torch.tensor( data['map'] , dtype=torch.float32)
                position = torch.tensor( [data['position']]*col*row )             
                state_tensor  = torch.cat([state, position])

                reward = self.calculate_reward(state_tensor, data['position'] )
                
                done_mask = 0.0 if 1 else 1.0

                if self.prev_state is not None:
                    self.replay_buffer.put(self.prev_state, 
                                            self.action, 
                                            reward/100.0,
                                            state_tensor,
                                            done_mask )
                    
                    if len(self.replay_buffer) >= self.batch_size:
                        train(self.q_model, self.q_target, 
                            self.replay_buffer, device, 
                                optimizer=self.optimizer)

                        if (self.t+1) % self.target_update_period == 0:
                            # Q_target.load_state_dict(Q.state_dict()) <- naive update
                            for target_param, local_param in zip(self.q_target.parameters(), self.q_model.parameters()): #<- soft update
                                    target_param.data.copy_(self.tau*local_param.data + (1.0 - self.tau)*target_param.data)
                
                self.prev_state = state_tensor

                
                # Run Network
                self.action = self.q_model.sample_action( state_tensor, self.epsilon )

                self.total_rewards += reward
                # print(self.t, self.action, self.total_rewards)

                self.t += 1

                client_socket.send( str(self.action).encode() )

                if data["done"]:
                    print("Done", self.epsilon)

                    subprocess.call(['taskkill', '/F', '/T', '/PID',  str(self.process.pid)])


                    cwd = os.path.realpath("play_windows_20x12")
                    self.process = subprocess.Popen("CoinChallenger.exe ", cwd=cwd, shell=True, stdin=subprocess.PIPE)





                
if __name__ == "__main__":

    cwd = os.path.realpath("play_windows_20x12")
    p = subprocess.Popen("CoinChallenger.exe ", cwd=cwd, shell=True, stdin=subprocess.PIPE)
    
    
    agent = SocketAgent(p)
    agent.run()


