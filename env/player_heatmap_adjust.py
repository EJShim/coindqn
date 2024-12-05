import math
import random


ckpt ={}

    
class Player:
    def __init__(self):
        self._my_number = None
        self._column = None
        self._row = None        
        

        self.heatmap = None
        self.explored = []
        self.step = 0
        self.alpha = 0.1

    def get_name(self) -> str:

        return "heatmap_adjust"

    def initialize(self, my_number: int, column: int, row: int):
        
        self._my_number = my_number
        self._column = column
        self._row = row         

        # Prevent Prev
        self.prev_position_index = None

        # Score Heatmap
        self.heatmap = None
        self.input_data = None
        self.prev_state = None

        self.step = 1
        self.explored = [1] * column * row
        

    def move_next(self, map: list[int], my_position: int) -> int:        

        # Update Exloration info
        self.explored[my_position] += 1
        self.step += 1

        if self.prev_position_index: map[self.prev_position_index] = -1
        self.prev_position_index = my_position

        self.preprocess(map, my_position)

        # This is heatmap score
        heatmap_score = self.get_move_candidates()
        
        # Get Candidate
        index = sorted(range(len(heatmap_score)), key=lambda k: heatmap_score[k],reverse=True)


        return index[0]

    def get_reward(self, action):
        next_candidates = self.get_move_candidates()

        reward = next_candidates[action]
        if reward < 0:
            reward *= 10
        

        return reward
    
    def get_move_candidates(self):
        heatmap = self.input_data   

        score = [-1, -1, -1, -1]

        position = self.prev_position_index

        if position % self._column != 0:
            score[0] = heatmap[self.prev_position_index - 1] * (self.step / self.explored[self.prev_position_index-1] ) 
        if position > self._column:
            score[1] = heatmap[self.prev_position_index - self._column] * (self.step / self.explored[self.prev_position_index-self._column] ) 
        if position % self._column != self._column-1:
            score[2] = heatmap[self.prev_position_index + 1] * (self.step / self.explored[self.prev_position_index+1] ) 
        if position < self._column *(self._row -1):
            score[3] = heatmap[self.prev_position_index + self._column] * (self.step / self.explored[self.prev_position_index+self._column] ) 
        
        return score        
    
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

    def index_to_position(self, index):
        return [ index // self._column, index % self._column]
    
    def preprocess(self, state, index):        
        if state[index] == -1 : state[index] = 0
                
        map2d = self.to_2d_list(state)

        # Calculate Heatmap
        if self.heatmap == None:
            self.heatmap = self.initialize_heatmap(map2d)            

        else:            
            changed = [ x-y for (x, y) in zip(state, self.prev_state) ]
            for idx, value, in enumerate(changed):
                if abs(value) >1 :                     
                    self.update_heatmap(self.heatmap, map2d, idx, value )

        heatmap = sum(self.heatmap, [])        
                
        heatmap = [
            (x+h)  if x != -1 else x  for (x,h)  in zip(state, heatmap)
        ]

        self.debug = heatmap
        
        self.prev_state = state
        self.input_data = heatmap # sum(sample_map,[])

        return self.input_data


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

        
        alpha = self.alpha        
        # Threshold to limit the range of update
        max_distance = -math.log(0.01) / alpha

        for x in range(max(0, int(coin_x - max_distance)), min(n, int(coin_x + max_distance) + 1)):
            for y in range(max(0, int(coin_y - max_distance)), min(m, int(coin_y + max_distance) + 1)):
                distance = math.sqrt((x - coin_x) ** 2 + (y - coin_y) ** 2)
                if distance <= max_distance:
                    heatmap[x][y] += coin_value * math.exp(-alpha * distance)


    