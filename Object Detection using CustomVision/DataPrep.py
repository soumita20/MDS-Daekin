#The region of each tagged object needs to be specified using normalized co-ordinates. Let us use computer vision to generate the co-ordinates of each image and store them in json format.
import os
import json 

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image,ImageDraw, ImageFont

def BuildCoordinates():
    #Adding subscription key and endpoints for computer vision
    subscription_key = "d1991707813b4e3bad77eb82abd2b18e"
    endpoint = "https://cvsc154point2d.cognitiveservices.azure.com/"
    computervision_client = ComputerVisionClient(endpoint,CognitiveServicesCredentials(subscription_key))
    folder = "C:/Applications/Machine Learning/MDS-Data Science/mds-deakin/Object Detection using CustomVision/Dataset/mini_vehicle_dataset/train/"

    bus_dict = {}
    car_dict = {}
    bike_dict = {}
    autorickshaw_dict = {}
    truck_dict = {}
    #iterating through images of each category and passing them through the "detect_objects_in_stream" API of the computer vision client
    train_folder = "C:/Applications/Machine Learning/MDS-Data Science/mds-deakin/Object Detection using CustomVision/Dataset/mini_vehicle_dataset/train"

    for subdir, dirs, files in os.walk(train_folder):
        for file in files:
            with open(os.path.join(subdir, file),mode='rb') as imgstream:
                detect_objects_results = computervision_client.detect_objects_in_stream(imgstream)
                #Here I am looping through each of the predictions and getting the coordinates of the bounding boxes of each object in the image and saving it to a dictionary.
                for object in detect_objects_results.objects:
                    #print detected objects results with bounding boxes
                    print("Listing co-ordinates of file:",file)
                    if len(detect_objects_results.objects) == 0:
                        print("No objects detected.")
                    else:
                        for object in detect_objects_results.objects:
                            if "bike" in file:
                                bike_dict.update({os.path.splitext(os.path.basename(file))[0]:[object.rectangle.x,object.rectangle.y,object.rectangle.w,object.rectangle.h]})
                            elif "bus" in file:
                                bus_dict.update({os.path.splitext(os.path.basename(file))[0]:[object.rectangle.x,object.rectangle.y,object.rectangle.w,object.rectangle.h]})
                            elif "car" in file:
                                car_dict.update({os.path.splitext(os.path.basename(file))[0]:[object.rectangle.x,object.rectangle.y,object.rectangle.w,object.rectangle.h]})
                            elif "auto" in file:
                                autorickshaw_dict.update({os.path.splitext(os.path.basename(file))[0]:[object.rectangle.x,object.rectangle.y,object.rectangle.w,object.rectangle.h]})
                            elif "truck" in file:
                                truck_dict.update({os.path.splitext(os.path.basename(file))[0]:[object.rectangle.x,object.rectangle.y,object.rectangle.w,object.rectangle.h]})
#Here, the dictionaries are finally being written into json files
    with open("bike.json", "w") as write_file:
        json.dump(bike_dict,write_file,indent=2)

    with open("bus.json", "w") as write_file:
        json.dump(bus_dict,write_file,indent=2)

    with open("car.json", "w") as write_file:
        json.dump(car_dict,write_file,indent=2)

    with open("autorickshaw.json", "w") as write_file:
        json.dump(autorickshaw_dict,write_file,indent=2)

    with open("truck.json", "w") as write_file:
        json.dump(truck_dict,write_file,indent=2)
    




