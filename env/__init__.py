import random
import numpy as np
import cv2
from .render import render
np.set_printoptions(precision=5)

class CoinEnv:
    def __init__(self):
        self.row = None
        self.column = None

        self.preset_map = {
            12 : np.array([0, 0, 0, 0, 0, -1, 30, 30, 30, 100, 100, 30, 30, 30, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 30, 30, 30, 30, 30, 30, 30, 30, -1, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 10, 10, 10, 10, 10, 30, 30, 30, 30, 30, 30, 30, 30, 10, 10, 10, 10, 10, 0, 0, 10, 10, 10, 10, 10, 30, -1, 100, 200, 200, 100, -1, 30, 10, 10, 10, 10, 10, 0, 200, 10, -1, -1, 10, 10, 30, -1, 100, 100, 100, 100, -1, 30, 10, 10, -1, -1, 10, 200, 200, 10, -1, -1, 10, 10, 30, -1, 100, 100, 100, 100, -1, 30, 10, 10, -1, -1, 10, 200, 0, 10, 10, 10, 10, 10, 30, -1, 100, 200, 200, 100, -1, 30, 10, 10, 10, 10, 10, 0, 0, 10, 10, 10, 10, 10, 30, 30, 30, 30, 30, 30, 30, 30, 10, 10, 10, 10, 10, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, -1, 30, 30, 30, 30, 30, 30, 30, 30, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 30, 30, 30, 100, 100, 30, 30, 30, -1, 0, 0, 0, 0, 0]).reshape(12,20),
            20 : np.array( [0,0,-1,10,10,10,10,0,0,0,0,0,0,100,100,100,100,0,0,0,0,0,0,10,10,10,10,-1,0,0,0,0,-1,10,10,10,10,0,0,0,-1,0,0,100,200,200,100,0,0,-1,0,0,0,10,10,10,10,-1,0,0,0,0,0,10,10,10,10,-1,0,0,-1,10,10,10,10,10,10,10,10,-1,0,0,-1,10,10,10,10,0,0,0,-1,-1,0,10,10,10,10,-1,0,0,-1,10,10,10,10,10,10,10,10,-1,0,0,-1,10,10,10,10,0,-1,-1,10,10,10,10,10,10,10,-1,10,10,10,10,10,10,10,10,10,10,10,10,10,10,-1,10,10,10,10,10,10,10,10,10,10,10,10,10,10,0,10,10,10,10,10,10,10,10,10,10,10,10,10,10,0,10,10,10,10,10,10,10,-1,-1,-1,-1,0,0,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,0,0,-1,-1,-1,-1,100,100,10,10,10,0,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,0,10,10,10,100,100,100,100,10,10,10,-1,30,30,30,30,30,30,-1,-1,0,0,-1,-1,30,30,30,30,30,30,-1,10,10,10,100,100,100,100,10,10,10,-1,30,30,30,30,30,30,-1,100,200,200,100,-1,30,30,30,30,30,30,-1,10,10,10,100,100,100,100,10,10,10,-1,30,30,30,30,30,30,-1,100,200,200,100,-1,30,30,30,30,30,30,-1,10,10,10,100,100,100,100,10,10,10,-1,30,30,30,30,30,30,-1,-1,0,0,-1,-1,30,30,30,30,30,30,-1,10,10,10,100,100,100,100,10,10,10,0,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,0,10,10,10,100,100,-1,-1,-1,-1,0,0,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,0,0,-1,-1,-1,-1,10,10,10,10,10,10,10,0,10,10,10,10,10,10,10,10,10,10,10,10,10,10,0,10,10,10,10,10,10,10,10,10,10,10,10,10,10,-1,10,10,10,10,10,10,10,10,10,10,10,10,10,10,-1,10,10,10,10,10,10,10,-1,-1,0,10,10,10,10,-1,0,0,-1,10,10,10,10,10,10,10,10,-1,0,0,-1,10,10,10,10,0,-1,-1,0,0,0,10,10,10,10,-1,0,0,-1,10,10,10,10,10,10,10,10,-1,0,0,-1,10,10,10,10,0,0,0,0,0,-1,10,10,10,10,0,0,0,-1,0,0,100,200,200,100,0,0,-1,0,0,0,10,10,10,10,-1,0,0,0,0,-1,10,10,10,10,0,0,0,0,0,0,100,100,100,100,0,0,0,0,0,0,10,10,10,10,-1,0,0]).reshape(20,30)
        }
                # Player Info
        self.position = None
        self.score = 0
        self.reward = 0

        
        self.consts =     [-1, 0, 10, 30, 100, 200, 500]
        


    def reset(self, player, row=12, column=20, preset=False, random_state=False ):
        self.row = row
        self.column = column
        preset_pos = [
            [0, 0],
            [0,self.column-1],
            [self.row-1,0],
            [self.row-1, self.column-1]
        ]

        self.position = [ random.randint(0,self.row-1), random.randint(0,self.column-1)  ]
        if preset:
            self.space = self.preset_map[self.row]
            self.position = preset_pos[ random.randint(0,3) ]            
        else:
            # percentage = np.array([ 32, 76, 52, 52, 16, 8 ,4], dtype=np.float32)
            unique, percentage = np.unique(self.preset_map[20], return_counts=True)
            percentage = list(percentage)
            percentage.append(random.randint(0, 10))            
            percentage = np.array(percentage, dtype=float)
            percentage /= percentage.sum()  # sum 1
            
            self.space = np.random.choice(self.consts, (self.row, self.column), p=percentage)            
            self.space[tuple(self.position)] = 0

        # TODO : Randomly add Zeros, Randomly add 500s, ranodmly change position
        if random_state:
            pass



        self.score = 0
        self.reward = 0

        self.total_score = np.sum(self.space[self.space>0])        

        # Initialize Player
        self.player = player
        self.player.initialize(0, self.column, self.row)

        self.state = self.player.preprocess(self.get_state(), self.convert_index(self.position))

        return self.state

    def render(self):
        return render(self.space, self.position)

    def step(self, action=None):
        if action==None:
            action=self.player.move_next(self.get_state(), self.convert_index(self.position))
        
        done = False
        reward = 0 # reward for training
        score = 0 # just coin score

        # Update Action Position
        pos = self.position
        if action == 0 and pos[1] > 0:
            pos = [pos[0], pos[1]-1]            
        elif action == 1 and pos[0] > 0:            
            pos = [pos[0]-1, pos[1]]         
        elif action == 2 and pos[1] < self.column-1:            
            pos = [pos[0], pos[1]+1]
        elif action == 3 and pos[0] < self.row-1:
            pos = [pos[0]+1, pos[1]]
        if self.space[pos[0]][pos[1]] == -1:
            pos = self.position
        self.position = pos

        score += self.space[ tuple(self.position) ] # coin reward      
        reward += self.player.get_reward(action)
        self.space[ tuple(self.position) ] = 0
        
        # reward -= self.time_step*0.1
        self.score += score
        self.reward += reward

        if score > self.total_score :            
            done=True

        self.state = self.player.preprocess(self.get_state(), self.convert_index(self.position))

        return self.state, reward, done

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