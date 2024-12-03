import math
import random


def linear_layer(mat1, mat2, bias, relu=False):
    rows = len(mat1)
    cols = len(mat1[0])
    result = [0] * rows    
    for i in range(rows):
        for j in range(cols):
            result[i] += mat1[i][j] * mat2[j]
        result[i] += bias[i]
        if relu:
            result[i] = max(result[i], 0)
    return result

def argmax(x):
    return max(range(len(x)), key=lambda i: x[i])

class DuelNeuralNetwork:
    def __init__(self, ckpt):        
        self.ckpt = ckpt

        self.num_layers = 0
        for key, val in ckpt.items():            
            names = key.split(".")
            if names[0] == "layers": self.num_layers = int(names[1]) // 2+1
        

    def __call__(self, x):
        y = x
        for i in range(self.num_layers):            
            y = linear_layer(self.ckpt[f"layers.{i*2}.weight"], y, self.ckpt[f"layers.{i*2}.bias"], relu=True)        
        Adv = linear_layer(self.ckpt["A.weight"], y, self.ckpt["A.bias"], relu=False)

        V = linear_layer(self.ckpt["V.weight"], y, self.ckpt["V.bias"], relu=False)
        V = V[0]

        Q = [V + (x - sum(Adv)/4) for x in Adv]
                
        score = [ abs(x) for x in Q ]
        score_sum = sum(score)
        score = [ x / score_sum for x in score ]

        return score
    
class Player:
    def __init__(self, sight=9):
        self._my_number = None
        self._column = None
        self._row = None
        self._eps = None
        self._sight = sight
        self._nn = DuelNeuralNetwork(ckpt)
        self.step:int = None
        self.explored = []
        self.prev_action = None
        self.state_space = self._sight * self._sight
        

    def get_name(self) -> str:

        return "Elmo V7.1-max100"

    def initialize(self, my_number: int, column: int, row: int):
        
        self._my_number = my_number
        self._column = column
        self._row = row 
        self._eps = 0

        # Check Exploration        
        self.step = 0
        self.explored = [0] * self._column * self._row

        # Prevent Prev
        self.prev_position_index = None

        # NN Input Size
        

    def move_next(self, map: list[int], my_position: int) -> int:
        if self.prev_position_index: map[self.prev_position_index] = -1
        self.prev_position_index = my_position


        input_data = self.preprocess(map, my_position)
        score = self._nn(input_data)# Score's alwyas positive, sum 1, because softmaxed

        # preprocess subgrid charactor index 
        move_candidates = self.get_move_candidates(input_data)
        
        valid_score = [bool(x+1)*y for x,y in zip(move_candidates, score)]        
        # valid_score = [(x+1)*y for x,y in zip(move_candidates, score)]        
        
        # Get Candidate
        index = sorted(range(len(valid_score)), key=lambda k: valid_score[k],reverse=True)


        if random.random() < self._eps:
            return random.randint(0, 3)
        else:
            index[0]        

        return index[0]
    
    def get_move_candidates(self, cropped_sight):
        center = (self._sight * self._sight) // 2
        return [ 
            cropped_sight[center - 1],
            cropped_sight[center - self._sight],
            cropped_sight[center + 1],
            cropped_sight[center + self._sight]
        ]
    
    def make_2d_input_map(self, input_map):
        # input to 2d amp
        result = [[0] * self._column for _ in range(self._row)]

        for r in range(self._row):
            result[r] = input_map[r*self._column:r*self._column+self._column ]

        return result

    def sample_pad_2d_input_map(self, map2d, position):
        column = len(map2d[0]) 
        row = len(map2d)
        pad = (self._sight//2)
        
        result = [[-1]*(column+(pad*2)) for _ in  range(row + (pad*2)) ]
        for r in range(row):
            result[r+pad][pad:-pad] = map2d[r]

        state = [
            result[i][position[1]:position[1]+self._sight] 
            for i in range(position[0],position[0]+self._sight)]

        return state

    def index_to_position(self, index):
        return [ index // self._column, index % self._column]

    def firstperson_view(self, state, index):
        position = self.index_to_position(index) # This is correct
        map2d = self.make_2d_input_map(state)
        sample_map = self.sample_pad_2d_input_map(map2d, position)

        return sample_map
    
    def preprocess(self, state, index):

        self.step += 1
        self.explored[index] = self.step

        state = [ x+self.step - self.explored[idx] if x != -1 else x for idx,x in enumerate( state) ]

        # # debug
        # import cv2
        # import numpy as np
        # score_map = np.array(state, dtype=np.float32).reshape(12,20)
        # score_map = ((score_map / 500.0)*255.0).astype(np.uint8)
        # score_map = cv2.applyColorMap(score_map, cv2.COLORMAP_RAINBOW)
        # score_map = cv2.resize(score_map, (200*2, 120*2))
        # cv2.imshow("score", score_map)        


        position = self.index_to_position(index) # This is correct
        map2d = self.make_2d_input_map(state)
        sample_map = self.sample_pad_2d_input_map(map2d, position)

        # Set Player Value -2
        sample_map[ self._sight//2][self._sight//2 ] = -2
        
        player_view = sum(sample_map,[])

        return player_view