import cv2
import imutils
# HOG feature descriptor is used in computer vision popularly for object detection
# detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
  
cap = cv2.VideoCapture('pedestrain-crossing-video.mp4') #recorded video path copy
  
while cap.isOpened():
    # Read the video stream
    res, image = cap.read()
    if res:
        image = imutils.resize(image, width=min(400, image.shape[1]))
        (regions, _) = hog.detectMultiScale(image, winStride=(4, 4),  padding=(4, 4),scale=1.05)
                                            
        for (x, y, w, h) in regions:
            cv2.rectangle(image, (x, y),(x+w,y+h),(0,0,255), 2)        #draw rectangle(coundary box) around the detected person
        cv2.imshow("Image", image)                     #output video
        if(cv2.waitKey(25) & 0xFF == ord('q')):        #to stop the video press key 'q'
            break
    else:
        break
 
cap.release()
cv2.destroyAllWindows()
