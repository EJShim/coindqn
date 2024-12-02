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
        self._sight = sight
        self._nn = NeuralNetwork(ckpt)
        self.step:int = None
        self.explored = []
        self.prev_action = None
        self.heatmap = None
        self.state_space = self._sight * self._sight
        self.alpha = 0.2
        
    

    def get_name(self) -> str:

        return "HeatmapFollower"

    def initialize(self, my_number: int, column: int, row: int):
        
        self._my_number = my_number
        self._column = column
        self._row = row 
        self._eps = 0.0

        # Check Exploration        
        self.step = 0

        # Prevent Prev
        self.prev_position_index = None

        # Score Heatmap
        self.heatmap = None
        self.prev_state = None
        

    def move_next(self, map: list[int], my_position: int) -> int:        

        if self.prev_position_index: map[self.prev_position_index] = -1
        self.prev_position_index = my_position


        input_data = self.preprocess(map, my_position)
        # score = self._nn(input_data)# Score's alwyas positive, sum 1, because softmaxed

        # This is heatmap score
        move_candidates = self.get_move_candidates(input_data)
                
        
        # valid_score = [bool(x+1)*y for x,y in zip(move_candidates, score)]        
        # valid_score = [(x+1)*y for x,y in zip(move_candidates, score)]        
        valid_score = move_candidates
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
    
    def to_2d_list(self, flat_list):
        """
        Convert a 1D list into a 2D list of size n x m.

        Args:
            flat_list (list): 1D list to be converted.
            n (int): Number of rows in the 2D list.
            m (int): Number of columns in the 2D list.

        Returns:
            list of list: 2D list with dimensions n x m.
        """

        n = self._row
        m = self._column

        if len(flat_list) != n * m:
            raise ValueError("The size of the 1D list does not match the specified dimensions.")

        # Create the 2D list
        return [flat_list[i * m:(i + 1) * m] for i in range(n)]

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
        map2d = self.to_2d_list(state)
        sample_map = self.sample_pad_2d_input_map(map2d, position)

        return sample_map
    
    def preprocess(self, state, index):        
        

        if state[index] == -1 : state[index] = 0
        

        position = self.index_to_position(index) # This is correct
        map2d = self.to_2d_list(state)

        # Calculate Heatmap
        if self.heatmap == None:
            self.heatmap = self.initialize_heatmap(map2d)            

        else:            
            changed = [ x-y for (x, y) in zip(state, self.prev_state) ]
            for idx, value, in enumerate(changed):
                if abs(value) >1 :                     
                    self.update_heatmap(self.heatmap, map2d, idx, value )
            
        heatmap_flatten = sum(self.heatmap, [])
        heatmap_min = min(heatmap_flatten)
        heatmap_max = max(heatmap_flatten)                
        # heatmap_normalized = [
        #     (x-heatmap_min)/(heatmap_max-heatmap_min) for x in heatmap_flatten
        # ]            
        heatmap_normalized = [
            x / 2000 for x in heatmap_flatten
        ]
        heatmap_added = [
            x + h*10  if x != -1 else x  for (x,h)  in zip(state, heatmap_normalized)
        ]
        
        self.prev_state = state

        import numpy as np
        import cv2        
        debug = np.array(heatmap_normalized, dtype=float).reshape(self._row, self._column)        
        debug = cv2.resize(debug, (200*2, 120*2))
        cv2.imshow("heatmap", debug)

        sample_map = self.sample_pad_2d_input_map(self.to_2d_list(heatmap_added), position)

        # Set Player Value -2
        sample_map[ self._sight//2][self._sight//2 ] = -2
        
        player_view = sum(sample_map,[])

        return player_view


    def initialize_heatmap(self, grid):
        """
        Initialize the global heatmap for the entire grid.

        Args:
            grid (list of list): n x m grid where each cell contains the coin value (0 if no coin).
            alpha (float): Decay factor for the distance.

        Returns:
            list of list: Heatmap as a 2D list of the same size as the input grid.
        """
        n = len(grid)
        m = len(grid[0])
        heatmap = [[0.0 for _ in range(m)] for _ in range(n)]

        for i in range(n):
            for j in range(m):
                if grid[i][j] > 0:  # Only process cells with coins
                    value = grid[i][j]
                    coin_index = i*self._column + j
                    self.update_heatmap(heatmap, grid, coin_index, value)        
        return heatmap



    def update_heatmap(self, heatmap, grid, coin_index, coin_value):
        """
        Update the global heatmap based on the change in a single coin.

        Args:
            heatmap (list of list): The current heatmap to be updated.
            grid (list of list): The grid with coin values.
            coin_x (int): X-coordinate of the coin.
            coin_y (int): Y-coordinate of the coin.
            coin_value (float): The value of the coin being added or removed.
            alpha (float): Decay factor for the distance.
            add (bool): If True, add the coin's contribution; if False, remove it.

        Returns:
            None: Updates the heatmap in place.
        """
        coin_x, coin_y = self.index_to_position(coin_index)

        n = len(grid)
        m = len(grid[0])


        
        # Threshold to limit the range of update
        max_distance = -math.log(0.01) / self.alpha

        for x in range(max(0, int(coin_x - max_distance)), min(n, int(coin_x + max_distance) + 1)):
            for y in range(max(0, int(coin_y - max_distance)), min(m, int(coin_y + max_distance) + 1)):
                distance = math.sqrt((x - coin_x) ** 2 + (y - coin_y) ** 2)
                if distance <= max_distance:
                    heatmap[x][y] += coin_value * math.exp(-self.alpha * distance)

    
