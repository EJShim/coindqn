import cv2
import time
from env import CoinEnv
# from env import player
# from env.render import render
import argparse
import importlib
import os
# from agents.v1 import Player


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--wall", type=int, default=15)
    parser.add_argument("--player", type=str, default="player")
    args  = parser.parse_args()

    env = CoinEnv(wall=args.wall)


    playermodule = importlib.import_module(f"env.{args.player}")
    player = playermodule.Player()

    player.initialize(0, env.column, env.row)

    
    # cv2.namedWindow("player", cv2.WINDOW_NORMAL)


    space, position_index = env.reset()        

    for t in range(1000):

        action = player.move_next(space, position_index)        
        space, reward, done, position_index = env.step(action) 

        screen = env.render()
        # charactor_view = render( np.array(player.firstperson_view(space, position_index)), [player._sight//2, player._sight//2])
        # cv2.namedWindow("render", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("render", )

        screen = cv2.resize(screen, (int(env.column*20), int(env.row*20)), interpolation=cv2.INTER_NEAREST_EXACT)
        # cv2.resizeWindow("player", 256, 256)

        cv2.imshow("render", screen)
        # cv2.imshow("player", charactor_view)
        cv2.waitKey(1)
        time.sleep(0.00001)