import numpy as np


def make_2d_input_map(input_map, row, column):

    # input to 2d amp
    result = [[0] * column for _ in range(row)]

    for r in range(row):
        result[r] = input_map[r*column:r*column+20 ]

    return result

def sample_pad_2d_input_map(map2d, sight=9, position=[0,3]):
    column = len(map2d[0]) 
    row = len(map2d)
    pad = sight // 2
    result = [[-1]*(column+(pad*2)) for _ in  range(row + (pad*2)) ]
    for r in range(row):
        result[r+pad][pad:-pad] = map2d[r]

    state = [result[i][position[1]:position[1]+sight] for i in range(position[0],position[0]+sight)]

    # state = result[0][position[0]:position[0]+sight]
    return state

def index_to_position(index, row, column):
    return [ index//column, index%column]


if __name__ == "__main__":

    input_map = [0, 0, 0, 0, 0, -1, 30, 30, 30, 100, 100, 30, 30, 30, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 30, 30, 30, 30, 30, 30, 30, 30, -1, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 10, 10, 10, 10, 10, 30, 30, 30, 30, 30, 30, 30, 30, 10, 10, 10, 10, 10, 0, 0, 10, 10, 10, 10, 10, 30, -1, 100, 200, 200, 100, -1, 30, 10, 10, 10, 10, 10, 0, 200, 10, -1, -1, 10, 10, 30, -1, 100, 100, 100, 100, -1, 30, 10, 10, -1, -1, 10, 200, 200, 10, -1, -1, 10, 10, 30, -1, 100, 100, 100, 100, -1, 30, 10, 10, -1, -1, 10, 200, 0, 10, 10, 10, 10, 10, 30, -1, 100, 200, 200, 100, -1, 30, 10, 10, 10, 10, 10, 0, 0, 10, 10, 10, 10, 10, 30, 30, 30, 30, 30, 30, 30, 30, 10, 10, 10, 10, 10, 0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0, -1, 30, 30, 30, 30, 30, 30, 30, 30, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 30, 30, 30, 100, 100, 30, 30, 30, -1, 0, 0, 0, 0, 0]

    row = 12
    column = 20
    


    for position_index in range(row*column):
        
        position = index_to_position(position_index, row, column)        
        map2d = make_2d_input_map(input_map, row, column)
        padded = sample_pad_2d_input_map(map2d, position=position)

        # # print(padded)
        print(np.array(padded).shape)
        # print(np.array(padded).shape)
    

    position_index = 19
    position = index_to_position(position_index, row, column)        
    map2d = make_2d_input_map(input_map, row, column)
    padded = sample_pad_2d_input_map(map2d, position=position)

    # # print(padded)
    print(np.array(padded))