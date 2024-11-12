import numpy as np
import cv2
import pygame


pygame.mixer.init()
sound = pygame.mixer.Sound("bip.ogg")

#Load the pre-trained classifiers XML files
# refer to doc : https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
face_cascade = cv2.CascadeClassifier('./Models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./Models/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('./Models/haarcascade_mcs_nose.xml')

#Capture the feed of the webcam. 
# 0 : default value of webcam. 
# If multiple webcams, can change the value depending on which webcam to use
cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Can't access webcam.")
    exit()

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

        #ROI : Region Of Interest
        roi_gray = gray[y:y+h, x:x+w] #make one output image in grayscale for the detection
        roi_color = input[y:y+h, x:x+w] #make one output image for the output visual


        ## NOSE DETECTION : ###

        nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.075,minNeighbors=5)
        if len(nose) < 2:
            for (nx,ny,nw,nh) in nose:
                cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(0,0,255),2)

        # ### EYE DETECTION : ####

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.075,minNeighbors=5,minSize=(50,50))
        if len(eyes) < 2:
            sound.play()
        elif len(eyes) == 2:
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    return input


#Read frame by frame the camera feed (saved in a frame)
while True:
    check, frame = cam.read() #captures a frame of the camera

    #check : boolean to check if catpure succeeded
    if not check :
        print("Error : Can't read the frame.")
        break

    # Optionnal : Resize frame for faster processing
    #frame = cv2.resize(frame, (640, 480))

    #detect face
    frame = detect_face(frame)

    #Display image
    cv2.imshow('video', frame)

    #Quit with keyboard : ESCAPE KEY
    key = cv2.waitKey(1)
    if key == 27:
        break

#free ressources
cam.release()
cv2.destroyAllWindows()

