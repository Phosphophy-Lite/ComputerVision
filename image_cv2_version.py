import numpy as np
import cv2

#Load the pre-trained classifiers XML files
# refer to doc : https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
face_cascade = cv2.CascadeClassifier('./Models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./Models/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('./Models/haarcascade_mcs_nose.xml')

def detect_face(input):
    #conversion of the image input to a grayscale (because detection works better this way)
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

    #detection of objects in a picture
    #1.3 : scale in which each frame is resized. 1.3 is common for accuracy and speed
    # 5 : k-nearest neighbor method, here the minimum of positive neighbors needed to confirm if class is face or not
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # no face detected
    if len(faces) == 0:
        return input

    #Create the rectangles around the detected areas
    # (x,y) : upper left corner coordinates
    # w, h : width, height of rectangle
    for (x,y,w,h) in faces:

        #draw a rectangle on the input pic
        #rectangle(pic, tuple upper_left_coordinates, tupple bottom_right_coordinates, tupple RGB_color, line_weight)
        cv2.rectangle(input,(x,y),(x+w,y+h),(255,0,0),2)

        ### EYE DETECTION : ####

        #ROI : Region Of Interest
        roi_gray = gray[y:y+h, x:x+w] #make one output image in grayscale for the detection
        roi_color = input[y:y+h, x:x+w] #make one output image for the output visual

        nose = nose_cascade.detectMultiScale(roi_gray)
        print(nose)
        for (nx,ny,nw,nh) in nose:
            cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,0,255),2)

        eyes = eye_cascade.detectMultiScale(roi_gray)
        if len(eyes):
            for (ex,ey,ew,eh) in eyes:
                for (nx,ny,nw,nh) in nose: #prevents the model to mistakenly detect nose holes as eyes, because it often does
                    if (ny > ey):
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    return input

#Load the input image/video in grayscale
filename = "ai_generated_face.jpg" #change with the filename you want to test the model with
img = cv2.imread('./Img/' + filename)
img = detect_face(img)
cv2.imshow('img',img)


#free ressources
cv2.waitKey(0)
cv2.destroyAllWindows()


