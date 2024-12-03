import random

import socket
import json
import time
import gc

SERVER_IP = 'localhost'
SERVER_PORT = 5050
SIZE = 1024
SERVER_ADDR = (SERVER_IP, SERVER_PORT)

class Player:
    def __init__(self):
        self._my_number = None
        self._column = None
        self._row = None
        self._sight = 9
        self.time_limit = 40


        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect(SERVER_ADDR)  # 서버에 접속

            init_data = json.dumps({
                'type' : 'start'
            })
            client_socket.send(init_data.encode())  # 

    def get_name(self) -> str:
        """
        플레이어의 이름을 반환합니다.
        """
        return "AI"

    def initialize(self, my_number: int, column: int, row: int):
        """
        플레이어의 초기화 메서드.
        :param my_number: 플레이어 번호
        :param column: 열 크기
        :param row: 행 크기
        """
        self._my_number = my_number
        self._column = column
        self._row = row
        self._sight = 9
        self._start = time.time()

        self.done = False
        self.reward = 0

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect(SERVER_ADDR)  # 서버에 접속

            init_data = json.dumps({
                'type' : 'init',
                'sight' : self._sight 
            })
            client_socket.send(init_data.encode())  # 

    def move_next(self, map: list[int], my_position: int) -> int:
        """
        플레이어의 다음 이동을 결정하는 메서드.
        :param map: 게임 맵 정보
        :param my_position: 플레이어의 현재 위치
        :return: 0(상), 1(우), 2(하), 3(좌) 중 랜덤 방향
        """
        # calculate elapsed time
        elapsed_time = time.time() - self._start
        if elapsed_time > self.time_limit  : self.done = True

        

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect(SERVER_ADDR)  # 서버에 접속
            

            try:
                #Calcualte Map
                position = self.index_to_position(my_position)
                map2d = self.make_2d_input_map(map)
                state = self.sample_pad_2d_input_map(map2d, position)


                send_data = json.dumps({
                    'type' : "state",                
                    'done' : self.done,
                    'reward' : self.reward,
                    'state' : state
                })
                client_socket.send(send_data.encode())  # 서버에 메시지 전송

                msg = client_socket.recv(SIZE)  
                direction = int( msg.decode() )
            except Exception as e:
                send_data = json.dumps({
                    'type' : "error",
                    'value' : str(e) + str(my_position)
                })
                client_socket.send(send_data.encode())  # 서버에 메시지 전송

        if self.done : 
            return -1
        # 0 : l
        # 1 : t
        # 2 : r
        # 3 : b
        self.reward = 0
        if direction == 0:
            self.reward = state[4][3]
        elif direction == 1:
            self.reward = state[3][4]
        elif direction == 2:
            self.reward = state[4][5]
        else:
            self.reward = state[5][4]
        # pad = self._sight // 2
        if self.reward < 0.0 :
            self.reward = -100.0 
            self.done = True

        return direction
    

    

        
    def make_2d_input_map(self, input_map):
        # input to 2d amp
        result = [[0] * self._column for _ in range(self._row)]

        for r in range(self._row):
            result[r] = input_map[r*self._column:r*self._column+20 ]

        return result

    def sample_pad_2d_input_map(self, map2d, position=[0,3]):
        column = len(map2d[0]) 
        row = len(map2d)
        pad = self._sight // 2
        result = [[-1]*(column+(pad*2)) for _ in  range(row + (pad*2)) ]
        for r in range(row):
            result[r+pad][pad:-pad] = map2d[r]

        state = [
            result[i][position[1]:position[1]+self._sight] 
            for i in range(position[0],position[0]+self._sight)]

        return state

    def index_to_position(self, index):
        return [ index//self._column, index%self._column]
