# Remove warning message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Required library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob
import sklearn
from os import listdir
from os.path import isfile, join

# Load a given neural network a the path
def load_model(path):
    try:
        path = splitext(path)[0]

        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()

        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)

        print("[INFO] Loading model successfully")

        return model
    except Exception as e:
        print(e)

# Load the labels for the neural network
def load_labels():
    labels = LabelEncoder()
    labels.classes_ = np.load('./models/classes.npy')

    print("[INFO] Labels loaded successfully")

    return labels

# Load image at the given path and return it
def load_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255

    if resize:
        img = cv2.resize(img, (224,224))
    
    return img

# Detect the license plate in the given image and crop the image to this license plate
def get_plate(img, Dmax=608, Dmin = 608):
    ratio = float(max(img.shape[:2])) / min(img.shape[:2])
    side = int(ratio * Dmin)
    
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(model_wpod, img, bound_dim, lp_threshold=0.5)

    return LpImg, cor

# Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
                                        
    return cnts

# Get the characters of the given contours of characters and return them as a string
def get_characters_of_contours(contours, thre_mor, img):
    cropped_characters = crop_characters(contours, thre_mor, img)
    final_string = predict_characters(cropped_characters)

    return final_string

# Crop the images of characters around their characters
def crop_characters(contours, thre_mor, img):
    crop_characters = []

    # define standard width and height of character
    digit_w, digit_h = 30, 60

    for c in sort_contours(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 1<=ratio<=3.5: # Only select contour with defined ratio
            if h/img.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
                # Sperate number and give prediction
                curr_num = thre_mor[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

    return crop_characters

# Predict the characters of the given contour of characters
def predict_characters(crop_characters):
    final_string = ''

    for i,character in enumerate(crop_characters):
        letter = predict_character(character,model,labels)
        final_string += letter[0]

    return final_string

# Predict the character of the given contour of a single character using a neural network
def predict_character(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)

    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    
    return prediction

# Get te contours of a given image
def get_contours_of_image(img):
    img = cv2.convertScaleAbs(img, alpha=(255.0))

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image
    blur = cv2.GaussianBlur(gray,(7,7),0)
    blur = cv2.GaussianBlur(gray,(7,7),0)

    # Applied inversed thresh_binary 
    binary = cv2.threshold(blur, 180, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    
    cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return cont, thre_mor

# Get the license plate number of the image at the given path
def get_license_plate_of_image(img_path):
    img = load_image(img_path)
    img_license_plate, cor = get_plate(img)

    img_license_plate = img_license_plate[0]

    contours, thre_mor = get_contours_of_image(img_license_plate)
    license_plate = get_characters_of_contours(contours, thre_mor, img_license_plate)

    return license_plate

# Get the license plate numbers of all images at the given path
def get_license_plates_of_images(mypath):
    platesL = []
    images = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    for img in images:
        license_plate = get_license_plate_of_image(f"{mypath}/{img}")
        platesL.append(license_plate)

        print(f"Image: {mypath}/{img}       -      License plate: {license_plate}")
        
    return platesL

#loading pretrained models to detect license plate and detect and extract characters from the license plates
model = load_model("./models/mobile_nets/model.json")
model_wpod = load_model("./models/wpod/model.json")

labels = load_labels()

#get all predicted license plates from images at the given path
PredictedPlateNumber=get_license_plates_of_images("./Plate_examples")

#print list of all predicted license plates
print(PredictedPlateNumber)

#example extract license plate in image 3 
PredictedPlateNumber[3]

#extract correct license plates from .xml (available for trainingdata)
import xml.etree.ElementTree as ET

import os

path, dirs, files = next(os.walk("./Plate_examples_xml"))
file_count = len(files)

#create list of correct license plates for every image in given path
actuallPlateNumber=[]
for Image in range(file_count):
    image_paths_xml = glob.glob("Plate_examples_xml/*.xml")
    test_image_xml = image_paths_xml[Image]
    mytree = ET.parse(test_image_xml)
    myroot=mytree.getroot()


    for x in myroot[1]:      
            #print(x.text)
            val=x.text
            actuallPlateNumber.append(val)

print( actuallPlateNumber)

#example actual/correct license plate for image number 3
actuallPlateNumber[3]

#use fuzzywuzzy to compare arrays of correct license plates and predicted license plates
import fuzzywuzzy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
SumRatio=0
for i in range(file_count):
    SumRatio += fuzz.ratio(PredictedPlateNumber[i].lower(), actuallPlateNumber[i].lower())

  #calculate mean over all used images to evaluate the success of the model  
FinalScore=SumRatio/file_count
print(FinalScore)