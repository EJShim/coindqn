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

class NeuralNetwork:
    def __init__(self, ckpt):        

        self.ckpt = ckpt
        self.num_layers = len(self.ckpt.keys()) // 2


    def __call__(self, x):
        
        y = x
        for i in range(self.num_layers):
            relu = not i==(self.num_layers-1)
            y = linear_layer(self.ckpt[f"layer.{i*2}.weight"], y, self.ckpt[f"layer.{i*2}.bias"], relu=relu)
        
        return y
    
class Player:
    def __init__(self):
        self._my_number = None
        self._column = None
        self._row = None
        self._sight = None
        self._nn = None
        self.prev_position_index = None

    def get_name(self) -> str:

        return "Agent V1"

    def initialize(self, my_number: int, column: int, row: int):
        
        self._my_number = my_number
        self._column = column
        self._row = row
        self._sight = 9
        self.prev_position_index = None
        self._nn = NeuralNetwork(ckpt)

    def move_next(self, map: list[int], my_position: int) -> int:
        # Mask previous position
        if self.prev_position_index: map[self.prev_position_index] = -1

        input_data = self.preprocess(map, my_position)
        score = self._nn(input_data)

        self.prev_position_index = my_position

        # preprocess subgrid charactor index 
        sg_index = (self._sight * self._sight) // 2
        

        candidates = [
            input_data[sg_index - 1],
            input_data[sg_index - self._sight],
            input_data[sg_index + 1],
            input_data[sg_index + self._sight]
        ]

        # TODO : Get Candidate
        index = sorted(range(len(score)), key=lambda k: score[k],reverse=True)

        for valid in index:
            if candidates[valid] != -1:
                return valid

            
    def make_2d_input_map(self, input_map):
        # input to 2d amp
        result = [[0] * self._column for _ in range(self._row)]

        for r in range(self._row):
            result[r] = input_map[r*self._column:r*self._column+self._column ]

        return result

    def sample_pad_2d_input_map(self, map2d, position=[0,3]):
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
    
    def preprocess(self, state, index):
        position = self.index_to_position(index) # This is correct
        map2d = self.make_2d_input_map(state)
        sample_map = self.sample_pad_2d_input_map(map2d, position)
        player_view = [y for x in sample_map for y in x]

        return player_view