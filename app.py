import cv2 as cv
import mediapipe as mp

mpHands=mp.solutions.hands
drawing = mp.solutions.drawing_utils

hands=mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5

)

cam=cv.VideoCapture(0)
while True:
    success,frame = cam.read()

    if not success:
        print("Camera not detected")

    frame=cv.flip(frame,1)
    frameRGB=cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    handsDetected=hands.process(frameRGB)


    if handsDetected.multi_hand_landmarks:
        for landmark in handsDetected.multi_hand_landmarks:
            print(landmark)
             
            drawing.draw_landmarks(
                image=frame,
                landmark_list=landmark,
                connections=mpHands.HAND_CONNECTIONS
            )

    cv.imshow("Hand Landmark ",frame)

    if cv.waitKey(1)==ord('q'):
        break

cam.release()
cv.destroyAllWindows()