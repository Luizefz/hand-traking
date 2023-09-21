import cv2
import mediapipe as mp

video_capture = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = video_capture.read()
    imgColors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgColors)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            for id, lm in enumerate(hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                if (id == 8):
                    cv2.circle(img, (cx, cy), 25, (19, 246, 223), cv2.FILLED)
            mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)

    cv2.imshow("I see this: ", img)
    cv2.waitKey(1)
