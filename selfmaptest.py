from env import CoinEnv
import cv2
import time

if __name__ == "__main__":

    env = CoinEnv()

    
    screen = env.render()
    cv2.namedWindow("render", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("render", env.column*20, env.row*20)
    cv2.imshow("render", screen)
    # cv2.waitKey(0)

    for i in range(20):
        env.step()

        screen = env.render()
        cv2.imshow("render", screen)
        cv2.waitKey(1)
        time.sleep(0.1)
        