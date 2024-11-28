from typing import List

def preprocess_append_position(view, pidx) -> List:
    return view + [pidx]

def preprocess_onehot_position_12_20(view, pidx) -> List:
    position_onehot = [0] * 12 * 20
    position_onehot[pidx] = 1

    return view + position_onehot

def preprocess_large_position_12_20(view, pidx):
    position_tensors = [pidx] * 12 * 20

    return view + position_tensors

