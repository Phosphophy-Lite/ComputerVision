import numpy as np
import cv2
import urllib.request

#Load the pre-trained classifiers XML files
# refer to doc : https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
face_cascade = cv2.CascadeClassifier('./Models/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./Models/haarcascade_eye.xml')
nose_cascade = cv2.CascadeClassifier('./Models/haarcascade_mcs_nose.xml')

# URL to IPWebcam stream (to use Android phone camera)
url='http://192.168.1.12:8080/shot.jpg'

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
            print("Blinked!\n")
        elif len(eyes) == 2:
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    return input

while True:
    # Retrieve image from phone camera using urllib to send HTTP request to IP Webcam
    imgResp = urllib.request.urlopen(url)

    # Convert stream with read() into an array of bytes and then into an array of 8 bytes unsigned integers with numpy
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)

    # Decode the array to OpenCV format. Reads image in its original color format.
    img = cv2.imdecode(imgNp,cv2.IMREAD_COLOR)

    #detect face
    img = detect_face(img)

    # show image on screen
    cv2.imshow('IPWebcam', img)

    #Quit with keyboard : ESCAPE KEY
    key = cv2.waitKey(1)
    if key == 27:
        break

#free ressources
cv2.destroyAllWindows()
