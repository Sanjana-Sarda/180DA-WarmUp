#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#thresholding
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-features


import numpy as np 
import cv2

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    #Capture frame-by-frame
    ret, frame = cap.read()

    if ret==True:
        #Our operations on the frame come here
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        #define range of blue color in HSV
        lower_blue = np.array([90, 100, 0])
        upper_blue = np.array([120, 255, 255])

        #Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask = mask)

        gray_res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        uh, thresh = cv2.threshold(gray_res, 0, 255, 0)
        th2 = cv2.adaptiveThreshold(gray_res, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2) 

        contours, hierarchy =  cv2.findContours(th2, 1, 2)

        if len(contours)>0:
            for uh1, contour in enumerate(contours): 
                area = cv2.contourArea(contour) 
                if(area > 300): 
                    x, y, w, h = cv2.boundingRect(contour)
                    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


        #Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

    