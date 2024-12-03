import math
import random

    
class Player:
    def __init__(self):
        self._my_number = None
        self._column = None
        self._row = None
        self._eps = None
        self._sight = 9        
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
        
        map2d = self.to_2d_list(map)

        
        # Calculate Heatmap
        if self.heatmap == None:
            self.heatmap = self.initialize_heatmap(map2d)            

        else:            
            changed = [ x-y for (x, y) in zip(map, self.prev_state) ]
            for idx, value, in enumerate(changed):
                if abs(value) >1 :                     
                    self.update_heatmap(self.heatmap, map2d, idx, value )            
        heatmap = sum(self.heatmap, [])
        heatmap_min = min(heatmap)
        heatmap_max = max(heatmap)                
        heatmap = [ # normalize
            (x-heatmap_min)/(heatmap_max-heatmap_min) for x in heatmap
        ]                    
        heatmap = [ # Add Adjecent
            x + h*10  if x != -1 else x  for (x,h)  in zip(map, heatmap)
        ]
    
        candidate_indices = [
            my_position-1 if my_position%self._column > 0 else -1,
            my_position-self._column if my_position>self._column else -1,
            my_position+1 if not (my_position+1)%self._column==0 else -1,
            my_position+self._column if not my_position // self._column == (self._row-1) else -1
        ]

        # print(candidate_indices)
        scores = [heatmap[idx] if idx != -1 and map[idx] != -1 else -1 for idx in candidate_indices ]        

        # Get Candidate
        index = sorted(range(len(scores)), key=lambda k: scores[k],reverse=True)

        return index[0]
    
    def get_reward(self, action):
        return 0
    
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