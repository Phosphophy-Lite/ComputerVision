import numpy as np
import dlib
import cv2
from imutils import face_utils
import urllib.request
from gpiozero import LED
import threading

# to calculate distance between the eye landmarks 
from scipy.spatial import distance as dist 

# URL to IPWebcam stream (to use Android phone camera)
url='http://192.168.1.12:8080/shot.jpg'

# LED pin (GPIO 17)
led = LED(17)

# Initializing the Models for Landmark and face Detection 
detector = dlib.get_frontal_face_detector()

#Check 68 Facial Landmarks blueprint
# shape_predictor -> return <class 'dlib.shape_predictor'>
# see documentation (object shape_predictor : __call__(self, image_rect))
# image : image in which to detect the face
# rect : rectangular detection zone
landmark_predict = dlib.shape_predictor('../Models/shape_predictor_68_face_landmarks.dat') 

# Eye landmarks 
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"] 
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye'] 

### EYE ASPECT RATIO = "EAR" ###
# check https://www.sciencedirect.com/science/article/pii/S2667241322000039 for picture and points
# landmarks points : https://www.researchgate.net/profile/Fabrizio-Falchi/publication/338048224/figure/fig1/AS:837860722741255@1576772971540/68-facial-landmarks.jpg
# EAR = (||p2-p6|| + ||p3-p5||)/2*||p1-p4||)
def calculate_ear(eye):

    #||p2-p6||
    y1 = dist.euclidean(eye[1], eye[5]) #distance between point 38 and point 42
    #||p3-p5||
    y2 = dist.euclidean(eye[2], eye[4]) #distance between point 39 and point 41
    #||p1-p4||
    x1 = dist.euclidean(eye[0], eye[3])

    ear = (y1+y2)/(2*x1)
    return ear

def blink_detected(delay):
    print("Blinked\n")
    led.on()
    timer = threading.Timer(delay, led.off)
    timer.start()

def detect_face(input):
    #conversion of the image input to a grayscale (because detection works better this way)
    gray = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)

    # no face detected
    if len(faces) == 0:
        return input

    #Create the rectangles around the detected areas
    # (x,y) : upper left corner coordinates
    # w, h : width, height of rectangle
    for face in faces: #face is a rectangle of 4 values :
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        #draw a rectangle on the input pic
        #rectangle(pic, tuple upper_left_coordinates, tupple bottom_right_coordinates, tupple RGB_color, line_weight)
        cv2.rectangle(input,(x,y),(x+w,y+h),(255,0,0),2)

        ## LANDMARKS + EYE DETECTION ###

        #landmark detection
        #method __call__ of shape_predictor : returns <class 'dlib.full_object_detection'>
        shape = landmark_predict(gray, face)

        #full_object_detection class :
        # rect : type dlib.rectangle of detection (left,top,right,bottom attributes)

        #converting shape class to a list of (x,y) coord
        #shape_to_np : converts full_object_detection to numpy.ndarray (num_points, 2)
        #num_points : number of landmarks points (here 68)
        #2 : two dimensions for the points (x,y)
        shape = face_utils.shape_to_np(shape)

        #now shape is a numpy array of point coordinates lists [x, y]

        #extract lefteye and righteye landmarks
        lefteye = shape[L_start:L_end] # [37, 38, 39, 40, 41, 42] where each number is a [x, y] point (see 68 landmarks)
        righteye = shape[R_start:R_end] # [43, 44, 45, 46, 47, 48]

        leftEyeHull = cv2.convexHull(lefteye)
        rightEyeHull = cv2.convexHull(righteye)
        
        #calculate EAR
        left_EAR = calculate_ear(lefteye)
        right_EAR = calculate_ear(righteye)

        cv2.drawContours(input, [leftEyeHull], -1, (255, 0, 0), 2)
        cv2.drawContours(input, [rightEyeHull], -1, (255, 0, 0), 2)

        average = (left_EAR+right_EAR)/2
        if average < 0.2: #threshold
            blink_detected(2)

    return input

def main_loop():
    while True:
        try:
            # Retrieve image from phone camera using urllib to send HTTP request to IP Webcam
            imgResp = urllib.request.urlopen(url)

            # Convert stream with read() into an array of bytes and then into an array of 8 bytes unsigned integers with numpy
            imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)

            # Decode the array to OpenCV format. Reads image in its original color format.
            img = cv2.imdecode(imgNp,cv2.IMREAD_COLOR)

            #failed to load frame, skipping it (happens when HTTP requests to webcam failed sometimes)
            if img is None or img.size == 0:
                continue
        
            # Optionnal : Resize frame for faster processing
            img = cv2.resize(img, (1080,608))

            #detect face
            img = detect_face(img)

            # show image on screen
            cv2.imshow('IPWebcam', img)

            #Quit with keyboard : ESCAPE KEY
            key = cv2.waitKey(1)
            if key == 27:
                break

        except Exception as e:
             print(f"Error in main loop: {e}")
             continue
 
    #free ressources
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()





