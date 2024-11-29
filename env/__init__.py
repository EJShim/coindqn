import random
import numpy as np
import cv2
from .render import render
np.set_printoptions(precision=5)

class CoinEnv:
    def __init__(self, blank=76, wall=32):
        self.row = 12
        self.column = 20

        # Player Info
        self.position = [ random.randint(0,self.row), random.randint(0,self.column)  ]
        self.score = 0
        self.reward = 0

        # Random Map Objects
        # Wall = 32
        # Blank = 76
        # COPPER = 52
        # Silver = 52
        # Gold = 16
        # DIAMONDS = 8
        # Black thing = 4
        self.consts =     [-1, 0, 10, 30, 100, 200, 500]
        self.percentage = np.array([ wall, blank, 52, 52, 16, 8 ,4], dtype=np.float32)
        self.percentage /= self.percentage.sum()  # sum 1


    def reset(self, row=12, column=20):
        self.row = row
        self.column = column

        self.space = np.random.choice(self.consts, (self.row, self.column), p=self.percentage)
        self.position = [ random.randint(0,self.row-1), random.randint(0,self.column-1)  ]
        self.space[tuple(self.position)] = 0
        self.score = 0
        self.reward = 0
        return self.get_state(), self.convert_index(self.position)


    def render(self):
        return render(self.space, self.position)

    def step(self, action):
        
        done = False
        reward = 0 # reward for training
        score = 0 # just coin score

        pos = self.position
        if action == 0:
            pos = [pos[0], pos[1]-1]            
        elif action == 1:            
            pos = [pos[0]-1, pos[1]]         
        elif action == 2:            
            pos = [pos[0], pos[1]+1]
        else:
            pos = [pos[0]+1, pos[1]]


        # Check Valid Position
        if self.invalid_position(pos) :            
            reward = -10
        else:        
            # Update Position
            self.position = pos

            # Update Score
            score += self.space[ tuple(self.position) ] # coin reward      
            self.space[ tuple(self.position) ] = 0
            reward += score
        
        # reward -= self.time_step*0.1

        self.score += score
        self.reward += reward

        return self.get_state(), reward, done, self.convert_index(self.position)

    def invalid_position(self, position):        
        if position[0] < 0 or position[0] > self.row-1:
            return True
        if position[1] < 0 or position[1] > self.column-1:
            return True
        
        # if wall        
        if self.space[tuple(position)] == -1:
            return True
        return False

    def get_state(self):
        return self.space.ravel().tolist()
    
    def convert_2d(self, index):
        return [index // self.column, index % self.column]
    
    def convert_index(self, pos):
        return pos[0] * self.column + pos[1]