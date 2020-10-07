#https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html#converting-colorspaces
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html#thresholding
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html#contour-features
#https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097

import numpy as np 
import cv2
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0
    length = 0
    count = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        if endX - startX > length:
            length = endX-startX
            index = count

        startX = endX
        count = count+1
    cv2.rectangle(bar, (int(0), 0), (int(endX), 50), centroids[index].astype("uint8").tolist(), -1)
    # return the bar chart
    return bar


cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    #Capture frame-by-frame
    ret, frame = cap.read()

    if ret==True:        
        x, y, w, h = 250, 150, 180, 140
        frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        roi = frame[140:340,220:420].copy()
        
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
        clt = KMeans(n_clusters=4) #cluster number
        clt.fit(img)

        hist = find_histogram(clt)
        bar = plot_colors2(hist, clt.cluster_centers_)

        bar = cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)

        #Display the resulting frame
        cv2.imshow('frame', frame)
        cv2.imshow('plot', bar)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

    