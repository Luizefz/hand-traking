import mouse
import cv2
from hand_traking import hand_detector

def mouse_position(finger_x, finger_y, smooth=20):
    percent_x = (finger_x / 600) * 1920
    percent_y = (finger_y / 400) * 1080

    current_x, current_y = mouse.get_position()

    delta_x = (percent_x - current_x) / smooth
    delta_y = (percent_y - current_y) / smooth

    for step in range(1, smooth + 1):
        x = current_x + step * delta_x
        y = current_y + step * delta_y
        mouse.move(int(x), int(y))


    return mouse.get_position()


def main():
    video_capture = cv2.VideoCapture(0)
    traker = hand_detector() #utilizando o construtor da classe hand_detector

    while True:
        sucess, img = video_capture.read()
        img = traker.find_hands(img)
        lm_list = traker.find_hands_position(img)
        if len(lm_list) != 0:
            print(lm_list[8])
            mouse_position(lm_list[8][1], lm_list[8][2])
        
        cv2.imshow("I see this: ", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()