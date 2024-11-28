from env import CoinEnv
import cv2
import time
import numpy as np

from env.player import Player
from env.render import render

if __name__ == "__main__":

    # Create Environment
    env = CoinEnv()


    cv2.namedWindow("render", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("render", env.column*20, env.row*20)

    # CLO Player
    cv2.namedWindow("player", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("player", 128, 128)

    player = Player()
    player.initialize(0, env.column, env.row)

    state, position = env.reset()


    for i in range(100):
        # Show Status
        screen = env.render()
        cv2.imshow("render", screen)

        # Show Player
        view = player.preprocess(state, position)
        view = np.array(view).reshape(player._sight, player._sight)
        player_view = render(view, [4,4])
        cv2.imshow("player", player_view)

        # Take Action
        action = player.move_next( state, position )
        state, position = env.step(action)


        # Animation
        cv2.waitKey(1)
        time.sleep(0.1)
        