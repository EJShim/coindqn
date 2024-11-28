import random
import numpy as np
import cv2
from .render import render
np.set_printoptions(precision=5)

class CoinEnv:
    def __init__(self):
        self.row = 12
        self.column = 20

        # Player Info
        self.position = [ random.randint(0,self.row), random.randint(0,self.column)  ]
        self.score = 0

        # Random Map Objects
        self.consts =     [-1, 0, 10, 100, 200, 500]
        self.percentage = np.array([ 3.5,5,4,3,2, 1])
        self.percentage = np.exp(self.percentage)/sum(np.exp(self.percentage))        


    def reset(self, seed=0):
        self.space = np.random.choice(self.consts, (self.row, self.column), p=self.percentage)
        self.position = [ random.randint(0,self.row-1), random.randint(0,self.column-1)  ]
        self.space[*self.position] = 0

        return self.get_state(), self.convert_index(self.position)


    def render(self):
        return render(self.space, self.position)

    def step(self, action):


        pos = self.position
        if action == 0:
            pos = [pos[0]-1, pos[1]]            
        elif action == 1:            
            pos = [pos[0], pos[1]+1]         
        elif action == 2:            
            pos = [pos[0]+1, pos[1]]
        else:
            pos = [pos[0], pos[1]-1]

        # Check Valid Position
        if self.invalid_position(pos):
            pos = self.position

        self.position = pos

        # Update MAp and Score
        self.score += self.space[ *self.position ]
        self.space[ *self.position ] = 0

        return self.get_state(), self.convert_index(self.position)
 


    def invalid_position(self, position):        
        if position[0] < 0 or position[0] > self.row-1:
            return True
        if position[1] < 0 or position[1] > self.column-1:
            return True
        
        # if wall        
        if self.space[*position] == -1:
            return True
        return False

    def get_state(self):
        return self.space.ravel().tolist()
    
    def convert_2d(self, index):
        return [index // self.column, index % self.column]
    
    def convert_index(self, pos):
        return pos[0] * self.column + pos[1]