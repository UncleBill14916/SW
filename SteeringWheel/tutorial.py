

import cv2
import imutils
from imutils.video import VideoStream
import numpy as np
from directkeys import A, D, Space, ReleaseKey, PressKey

cam = VideoStream(src=0).start()
currentKey = list()

while True:
    key = False

    img = cam.read()
    img = np.flip(img, axis=1)
    img = np.array(img)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    blurred = cv2.GaussianBlur(hsv, (11, 11), 0)

    colourLower = np.array([53, 55, 209])
    colourUpper = np.array([180, 255, 255])

    mask = cv2.inRange(blurred, colourLower, colourUpper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    width = img.shape[1]
    height = img.shape[0]

    upContour = mask[0:height // 2, 0:width]
    downContour = mask[3 * height // 4:height, 2 * width // 5:3 * width // 5]

    cnts_up = cv2.findContours(upContour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_up = imutils.grab_contours(cnts_up)

    cnts_down = cv2.findContours(downContour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts_down = imutils.grab_contours(cnts_down)

    if len(cnts_up) > 0:
        c = max(cnts_up, key=cv2.contourArea)
        M = cv2.moments(c)

        # Check if M["m00"] is not zero to avoid division by zero
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])

            if cX < (width // 2 - 35):
                PressKey(A)
                key = True
                currentKey.append(A)

            elif cX > (width // 2 + 35):
                PressKey(D)
                key = True
                currentKey.append(D)
        else:
            # Handle the case where no significant contour is found
            cX = width // 2  # Default behavior or set to a neutral position

    if len(cnts_down) > 0:
        PressKey(Space)
        key = True
        currentKey.append(Space)

    img = cv2.rectangle(img, (0, 0), (width // 2 - 35, height // 2), (0, 255, 0), 1)
    cv2.putText(img, 'LEFT', (110, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (139, 0, 0))

    img = cv2.rectangle(img, (width // 2 + 35, 0), (width, height // 2), (0, 255, 0), 1)
    cv2.putText(img, 'RIGHT', (440, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (139, 0, 0))

    img = cv2.rectangle(img, (2 * (width // 5), 3 * height // 4), (3 * width // 5,height), (0,255,0),1)
    cv2.putText(img,'NITRO',(2*(width//5) +20,height-10),cv2.FONT_HERSHEY_DUPLEX ,1,(139 ,0 ,0))

    cv2.imshow("Steering", img)

    if not key and len(currentKey) != 0:
        for current in currentKey:
            ReleaseKey(current)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()