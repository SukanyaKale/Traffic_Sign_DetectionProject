import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_model
from pytorch_model import NeuralNet
from torchvision import transforms
import pyttsx3 as sp


engine = sp.init('sapi5')

voices = engine.getProperty('voices')
# print(voices[0].id)



engine.setProperty('voice', voices[1].id)

def speak(audio):
    print(f"Sign: {audio}")
    engine.say(audio)
    engine.runAndWait()


def crop_minAreaRect(img,rect):

    #rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    #rotate bounding box
    rect0 = (rect[0],rect[1],0.0)
    box = cv2.boxPoints(rect)
    pts = np.int0(cv2.transform(np.array([box]),M))[0]
    pts[pts < 0] = 0 

    #crop 
    img_crop = img_rot[pts[1][1]:pts[0][1], pts[1][0]:pts[2][0]]

    #rotate cropped image
    angle=rect[2]
    rows,cols = img_crop.shape[0], img_crop.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),-angle,1)
    img_final = cv2.warpAffine(img_crop,M,(cols,rows))


    return img_final

def area_check(contours):
    cnts = list()
    index = list()
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) >= 1024:
            cnts.append(contours[i])
            index.append(i)

    return cnts,index

def shape_check(contours,index):
    cnts = list()
    index = list()
    for cnt in contours:
        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        if len(approx)==3 or 9<=len(approx)<=12:
            cnts.append(cnt)
            index.append(contours.index(cnt))
    return cnts, index

def draw_rect(contours,image):
    for cnt in contours:
        x,y,w,h, = cv2.boundingRect(cnt)
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    return image

def hierarchy_check(hierarchy,index):
    result = index
    for i in index:
        if hierarchy[0][i][1] in index and hierarchy[0][i][3] ==-1:
            result.remove(i)

    return result

def detect_traffic_sign(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 100, 150])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv_image, lower_red, upper_red)
    
    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Area filter for the contours
    area_contours, index = area_check(contours)

    # Shape filtering contours
    shape_contours, index = shape_check(area_contours, index)

    # Draw bounding rectangle for contours
    image = draw_rect(shape_contours, image)

    # Crop detected sign and resize to 32x32 pixels
    images = []
    for cnt in shape_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        images.append(cv2.resize(image[y:y+h, x:x+h], (96, 96), interpolation=cv2.INTER_AREA))
    return images


def recognize_sign(image, model, device):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform_validation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    tensor = transform_validation(image).to(device).unsqueeze(0)
    output = model(tensor)
    _, pred = torch.max(output, 1)
    classes = ('Left Turn', 'No Entry', 'No Horn', 'No Left Turn', 'No Stopping', 'No U-Turn', 'Pedestrian Crossing', 'Right Turn', 'Speed Breaker', 'Junction Ahead')
    return classes[pred.item()]

# Initialize and load model onto CPU
device = torch.device("cpu")
model = NeuralNet()
model = torch.load('model_31Aug.pt', map_location=torch.device('cpu'))

model.eval()


# Real-time video capturing
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    images = detect_traffic_sign(frame)

    if len(images) != 0:
        for image in images:
            sign = recognize_sign(image, model, device)
            cv2.putText(frame, sign, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
            speak(sign)
    
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#engine.stop
cap.release()
cv2.destroyAllWindows()

