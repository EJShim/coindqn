
import cv2
import numpy as np

def render(state, position=None):
    space_8bit = state.copy()
    space_8bit[space_8bit==-1.0] = 44.0
    space_8bit[space_8bit==500.0] = 255.0
    space_8bit = space_8bit.astype(np.uint8)

    # Set Player position   
    if position:
        space_8bit[ *position ] = 1

    # Set Custom LUT
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    lut[44] = [[0,0,0]]
    lut[0] = [[255, 255, 255]]
    lut[10] =[[51, 115, 184]]
    lut[100] = [[192, 192, 192]]
    lut[200] = [[0, 215, 255]]
    lut[255] = [[255, 242, 185]]

    lut[1] = [[0, 0, 255]] # player1

    image = cv2.applyColorMap(space_8bit, lut)

    return image