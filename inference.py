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
    parser.add_argument("--preset", action='store_false')
    parser.add_argument("--player", type=str, default="player_stepscore")
    args  = parser.parse_args()


    env = CoinEnv()


    playermodule = importlib.import_module(f"env.{args.player}")
    player = playermodule.Player()    
    
    env.reset(player=player, row=20, column=30, preset=args.preset)        


    for t in range(500):        

        # action = player.move_next(space, position_index)        
        env.step() 

        screen = env.render()
        screen = cv2.resize(screen, (int(env.column*20), int(env.row*20)), interpolation=cv2.INTER_NEAREST_EXACT)        

        cv2.imshow("render", screen)
        # cv2.imshow("player", charactor_view)
        cv2.waitKey(int(bool(t)))
        time.sleep(0.01)