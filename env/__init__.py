import random
import numpy as np
import cv2
from .player import Player
np.set_printoptions(precision=5)

class CoinEnv:
    def __init__(self):
        self.row = 12
        self.column = 20

        # Initialize Players
        self.player = Player()
        self.position = [6, 10]


        # Map objects
        self.consts =     [-1, 0, 10, 100, 200, 500]
        self.percentage = np.array([ 3.5,5,4,3,2, 1])
        self.percentage = np.exp(self.percentage)/sum(np.exp(self.percentage))
        print(self.percentage)
        self.reset()


    def reset(self, seed=0):
        self.space = np.random.choice(self.consts, (self.row, self.column), p=self.percentage)


    def render(self):
        
        space_8bit = self.space.copy()
        space_8bit[space_8bit==-1.0] = 44.0
        space_8bit[space_8bit==500.0] = 255.0
        space_8bit = space_8bit.astype(np.uint8)

        # Set Player position   
        
        space_8bit[ *self.position ] = 1

        # Set Custom LUT
        # 44 : [0, 0, 0]
        # 0 : [256, 256, 256]
        # 10 : [10, 10, 0]
        # 100 : [200, 200, 200]
        # 200 : [200, 200, 10]
        # 255 : [ 50, 50, 50 ]

        lut = np.zeros((256, 1, 3), dtype=np.uint8)
        lut[44] = [[0,0,0]]
        lut[0] = [[255, 255, 255]]
        lut[10] =[[0, 100, 100]]
        lut[100] = [[200, 200, 200]]
        lut[200] = [[10, 200, 200]]
        lut[255] = [[50, 50, 50]]

        lut[1] = [[0, 0, 255]] # player1

        image = cv2.applyColorMap(space_8bit, lut)
        
        return image

    def step(self, action=None):

        action = self.player.move_next(self.get_state, self.convert_index( self.position ))

        pos = self.position
        if action == 0:
            pos = [pos[0]-1, pos[1]]
        elif action == 1:            
            pos = [pos[0], pos[1]+1]         
        elif action == 2:            
            pos = [pos[0], pos[1]+1]
        else:
            pos = [pos[0], pos[1]-1]            

        self.position = pos

    def get_state(self):
        return self.space.ravel().tolist()
    
    def convert_2d(self, index):
        return [index // self.column, index % self.column]
    
    def convert_index(self, pos):
        return pos[0] * self.column + pos[1]