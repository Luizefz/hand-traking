import cv2
import mediapipe as mp

class hand_detector():
    def __init__(self, static_mode=False, max_amount_hands=2, model_complexity=1, min_precise_detection=0.5, min_precise_tracking=0.5):
        self.static_mode = static_mode
        self.max_amount_hands = max_amount_hands
        self.model_complexity = model_complexity
        self.min_precise_detection = min_precise_detection
        self.min_precise_tracking = min_precise_tracking

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.static_mode, self.max_amount_hands, self.model_complexity, self.min_precise_detection, self.min_precise_tracking)
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, img, draw=True):
        img_colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_colors)

        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hand, self.mp_hands.HAND_CONNECTIONS)
    
        return img

    def find_hands_position(self, img, hand_number=0, draw=True):
        lm_list = []

        if self.results.multi_hand_landmarks:
            hand_detected = self.results.multi_hand_landmarks[hand_number]
            lm_list = [(id, int(lm.x * img.shape[1]), int(lm.y * img.shape[0])) 
                       for id, lm in enumerate(hand_detected.landmark)]

        if draw:
            for id, cx, cy in lm_list:
                if id == 8:
                    cv2.circle(img, (cx, cy), 8, (19, 246, 223), cv2.FILLED)

        return lm_list

def main():
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1024.0 )
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 768.0)
    traker = hand_detector() #utilizando o construtor da classe hand_detector

    while True:
        sucess, img = video_capture.read()
        img = traker.find_hands(img)
        lm_list = traker.find_hands_position(img)
        if len(lm_list) != 0:
            print(lm_list[8])

        cv2.imshow("I see this: ", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()