import math
import random

ckpt = {}

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

class NeuralNetwork:
    def __init__(self, ckpt):        
        self.ckpt = ckpt
        self.num_layers = len(self.ckpt.keys()) // 2

    def __call__(self, x):
        y = x
        for i in range(self.num_layers):
            relu = not i==(self.num_layers-1)
            y = linear_layer(self.ckpt[f"layers.{i*2}.weight"], y, self.ckpt[f"layers.{i*2}.bias"], relu=relu)
        
        score = [ math.exp(x) for x in y ]
        score_sum = sum(score)
        score = [ x / score_sum for x in score ]

        return score
    
class Player:
    def __init__(self, sight=9):
        self._my_number = None
        self._column = None
        self._row = None
        self._eps = None
        self._nn = NeuralNetwork(ckpt)
        self.step:int = None
        self.explored = []
        self.prev_action = None
        self.state_space = 12 * 20 + 1
        
        
        

    def get_name(self) -> str:

        return "Baseline"

    def initialize(self, my_number: int, column: int, row: int):
        
        self._my_number = my_number
        self._column = column
        self._row = row 
        self._eps = 0.00

        # Check Exploration        
        self.step = 0

        # Prevent Prev
        self.prev_position_index = None
        

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
        
    def get_move_candidates(self, state):
        my_position = state[-1]
        
        candidate_indices = [
            my_position-1 if my_position%self._column > 0 else -1,
            my_position-self._column if my_position>self._column else -1,
            my_position+1 if not (my_position+1)%self._column==0 else -1,
            my_position+self._column if not my_position // self._column == (self._row-1) else -1
        ]
        
        output = [ state[idx] if idx != -1 else -1 for idx in candidate_indices ]

        return output

    
    def preprocess(self, state, index):        

        input_array = state + [index]
        return input_array
