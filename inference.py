import cv2
import time
from env import CoinEnv
from env.player import Player


if __name__ == "__main__":
    env = CoinEnv()

    player = Player()
    player.initialize(0, env.column, env.row)

    cv2.namedWindow("render", cv2.WINDOW_NORMAL)


    space, position_index = env.reset()        

    for t in range(2000):

        

                
        action = player.move_next(space, position_index)        
        space, reward, done, position_index = env.step(action) 


        screen = env.render()

        cv2.resizeWindow("render", int(env.column*20), int(env.row*20))
        cv2.imshow("render", screen)
        cv2.waitKey(1)
        time.sleep(0.00001)