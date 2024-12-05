import math
from collections import deque

ckpt ={}

    
class Player:
    def __init__(self):
        self._my_number = None
        self._column = None
        self._row = None        
        

        self.heatmap = None        
        self.alpha = 0.2

    def get_name(self) -> str:

        return "HeatmapFollower_Normalize"

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
        

    def move_next(self, map: list[int], my_position: int) -> int:        

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
            score[0] = heatmap[self.prev_position_index - 1]
        if position > self._column:
            score[1] = heatmap[self.prev_position_index - self._column]
        if position % self._column != self._column-1:
            score[2] = heatmap[self.prev_position_index + 1]
        if position < self._column *(self._row -1):
            score[3] = heatmap[self.prev_position_index + self._column]
        
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
                
        self.grid = self.to_2d_list(state)

        # Calculate Heatmap
        if self.heatmap == None:
            self.heatmap = [[0.0 for _ in range(self._column)] for _ in range(self._row)]
            self.initialize_heatmap()           
        else:               
            changed = [ x-y for (x, y) in zip(state, self.prev_state) ]
            for idx, value, in enumerate(changed):
                if abs(value) >1 :                     
                    self._spread_heat(idx, -value)        

        heatmap = sum(self.heatmap, [])        
        
        heatmap = [
            x + h  if x != -1 else x  for (x,h)  in zip(state, heatmap)
        ]

        self.debug = heatmap
        
        self.prev_state = state
        self.input_data = heatmap # sum(sample_map,[])

        return self.input_data


    def initialize_heatmap(self):
        """
        Initialize the global heatmap for the entire grid.

        Args:
            grid (list of list): n x m grid where each cell contains the coin value (0 if no coin).
            alpha (float): Decay factor for the distance.

        Returns:
            list of list: Heatmap as a 2D list of the same size as the input grid.
        """
        n = len(self.grid)
        m = len(self.grid[0])
        
        for i in range(n):
            for j in range(m):
                if self.grid[i][j] > 0:  # Only process cells with coins
                    value = self.grid[i][j]
                    coin_index = i*self._column + j
                    self._spread_heat(coin_index, value)
                    # self.update_heatmap(heatmap, grid, coin_index, value)                

    def _spread_heat(self, coin_index, value):        
        """Spread heat from a coin using BFS."""


        y, x = self.index_to_position(coin_index)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited = [[False] * self._column for _ in range(self._row)]
        queue = deque([(x, y, 0)])  # (x, y, remaining_value)        
        while queue:
            cx, cy, distance = queue.popleft()            
            if visited[cy][cx]:
                continue
            visited[cy][cx] = True
            
            score = value * math.exp(-distance * self.alpha)

            self.heatmap[cy][cx] += score
            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy                
                
                if 0 <= nx < self._column and 0 <= ny < self._row:
                    # print(nx, ny, len(self.grid), len(self.grid[0]))                    
                    if self.grid[ny][nx] != -1 and not visited[ny][nx]:
                        queue.append((nx, ny, distance+1))  # Decrease value as it spreads