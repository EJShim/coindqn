import socket
import json
import random
import torch

if __name__ == "__main__":

    IP = ''
    PORT = 5050
    SIZE = 1024
    ADDR = (IP, PORT)

    print("run Serer")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        server_socket.bind(ADDR)
        server_socket.listen()

        
        # 무한루프 진입
        while True:
            client_socket, client_addr = server_socket.accept()  
            msg = client_socket.recv(SIZE)  # 클라이언트가 보낸 메시지 

            data = msg.decode()
            try:
                received = json.loads(data)
            except Exception as e:
                print(data)
                continue
            
            state = torch.tensor( received['map'] )
            position = torch.tensor( received['position'] ) 
            print(state.shape, position.shape)

            direction =  random.randint(0, 3)

            client_socket.send( str(direction).encode() )
