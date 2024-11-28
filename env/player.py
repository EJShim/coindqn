import random


class Player:
    def __init__(self):
        self._my_number = None
        self._column = None
        self._row = None
        self._sight = None


    def get_name(self) -> str:

        return "python player"

    def initialize(self, my_number: int, column: int, row: int):
        
        self._my_number = my_number
        self._column = column
        self._row = row
        self._sight = 9

    def move_next(self, map: list[int], my_position: int) -> int:

        direction = random.randint(0, 3)
        return direction
    

            
    def make_2d_input_map(self, input_map):
        # input to 2d amp
        result = [[0] * self._column for _ in range(self._row)]

        for r in range(self._row):
            result[r] = input_map[r*self._column:r*self._column+20 ]

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
