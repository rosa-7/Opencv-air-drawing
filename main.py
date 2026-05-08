import cv2 as cv
import mediapipe as mp
import numpy as np

mpHands=mp.solutions.hands
drawing = mp.solutions.drawing_utils

hands=mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5

)

cam=cv.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0

while True:
    success,frame = cam.read()

    if not success:
        print("Camera not detected")

    frame=cv.flip(frame,1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    frameRGB=cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    results=hands.process(frameRGB)
    

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            h, w, c = frame.shape

            # index finger tip = landmark 8
            x = int(hand.landmark[8].x * w)
            y = int(hand.landmark[8].y * h)

            # draw only if finger is up
            # (tip y < joint y → finger up)
            if hand.landmark[8].y < hand.landmark[6].y:
                cv.circle(frame, (x, y), 8, (0, 255, 0), -1)

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv.line(canvas, (prev_x, prev_y), (x, y), (0, 0, 255), 5)

                prev_x, prev_y = x, y
            else:
                prev_x, prev_y = 0, 0

            drawing.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)


    frame = cv.add(frame, canvas)

    cv.imshow("Air Drawing", frame)

    key = cv.waitKey(1)

    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)
cam.release()
cv.destroyAllWindows()