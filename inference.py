import cv2
import time
from env import CoinEnv
# from env import player
# from env.render import render
import argparse
import importlib
import numpy as np


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--wall", type=int, default=15)
    parser.add_argument("--preset", action='store_false')
    parser.add_argument("--size", nargs=2, type=int, default=[20,30])
    parser.add_argument("--player", type=str, default="player_stepscore")
    args  = parser.parse_args()

    if args.size[0] != 20 or args.size[0] != 20:
        args.preset = False
    

    env = CoinEnv()


    playermodule = importlib.import_module(f"env.{args.player}")
    player = playermodule.Player()    
    
    print(args.preset)
    env.reset(player=player, row=args.size[0], column=args.size[1], preset=args.preset)        


    for t in range(5000):        

        # action = player.move_next(space, position_index)        
        env.step() 

        screen = env.render()
        screen = cv2.resize(screen, (int(env.column*20), int(env.row*20)), interpolation=cv2.INTER_NEAREST_EXACT)        

        if hasattr(player, "heatmap"):
            heatmap = np.array(player.heatmap)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255
            heatmap = cv2.resize(heatmap, (args.size[1]*20, args.size[0]*20) )
            heatmap = heatmap.astype(np.uint8)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TWILIGHT)
            cv2.imshow("heatmap", heatmap)
            

        cv2.imshow("render", screen)
        # cv2.imshow("player", charactor_view)
        cv2.waitKey(int(bool(t)))
        time.sleep(0.01)