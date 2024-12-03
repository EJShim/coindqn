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
    def __init__(self):
        self._my_number = None
        self._column = None
        self._row = None
        self._eps = None
        self._sight = 9
        self._nn = NeuralNetwork(ckpt)
        self.step:int = None
        self.explored = []
        self.prev_action = None
        self.state_space = self._sight * self._sight + 4 # Quad score sum 
        
        
        

    def get_name(self) -> str:

        return "Quad"

    def initialize(self, my_number: int, column: int, row: int):
        
        self._my_number = my_number
        self._column = column
        self._row = row 
        self._eps = 0.05

        # Check Exploration        
        self.step = 0

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
    
    def calculate_quad_score(self, space, position):

        top_left = 0
        top_right = 0
        bottom_left = 0
        bottom_right = 0

        for i in range(self._row):
            for j in range(self._column):
                value = space[i][j]
                if value < 0 : continue
                if i<=position[0] and j<=position[1]:
                    top_left += value
                elif i<=position[0] and j>position[1]:
                    top_right += value
                elif i>position[0] and j<=position[1]:
                    bottom_left += value
                elif i>position[0] and j>position[1]:
                    bottom_right += value

        return [top_left/1000, top_right/1000, bottom_left/1000, bottom_right/1000]
    
    def preprocess(self, state, index):

        position = self.index_to_position(index) # This is correct
        map2d = self.make_2d_input_map(state)
        quad_vec = self.calculate_quad_score(map2d, position)
        sample_map = self.sample_pad_2d_input_map(map2d, position)

        # Set Player Value -2
        sample_map[ self._sight//2][self._sight//2 ] = -2
        
        player_view = sum(sample_map,[])
        out = player_view + quad_vec # 81 + 4

        return out