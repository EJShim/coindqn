import socket
import json
import random
import torch
from train_test import Q_net



if __name__ == "__main__":

    # Training Parameters : 
    eps_start = 0.3
    eps_end = 0.001
    eps_decay = 0.995


    epsilon = eps_start

    IP = ''
    PORT = 5050
    SIZE = 1024
    ADDR = (IP, PORT)

    print("run Serer")

    model = None

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        server_socket.bind(ADDR)
        server_socket.listen()

        
        # 무한루프 진입
        while True:
            client_socket, client_addr = server_socket.accept()  
            msg = client_socket.recv(SIZE)  # 클라이언트가 보낸 메시지 

            data = json.loads(msg.decode())

            if data['type'] == 'init':
                col = data['col']
                row = data['row']

                model = Q_net(state_space=col*row+1, action_space=4)

                print(data, col, row)
            elif data['type'] == 'state':
                state = torch.tensor( data['map'] , dtype=torch.float32)
                position = torch.tensor( data['position'] ) 

                input_tensor  = torch.cat([state, position.unsqueeze(0)])

                # Normalize Input
                input_tensor /= 255.0
                input_tensor -= 0.5
                
                
                # Run Network
                sample_action = model.sample_action( input_tensor, epsilon )

                print(sample_action)
                # direction =  random.randint(0, 3)
                # print(score, direction)

                client_socket.send( str(sample_action).encode() )
            
