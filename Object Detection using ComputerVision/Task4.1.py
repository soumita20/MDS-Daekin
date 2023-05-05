import PIL
import requests
import numpy as np
import os

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image,ImageDraw, ImageFont

from array import array
from PIL import Image

subscription_key = "d1991707813b4e3bad77eb82abd2b18e"
endpoint = "https://cvsc154point2d.cognitiveservices.azure.com/"
computervision_client = ComputerVisionClient(endpoint,CognitiveServicesCredentials(subscription_key))

remote_image_url = "https://cdn.pixabay.com/photo/2020/06/03/17/51/hummingbird-5255827_960_720.jpg"


#image description using computer vision - provides a brief description of any remote image along with confidence scores
description_results = computervision_client.describe_image(remote_image_url)

font = ImageFont.truetype('arial.ttf',16)

print("Description of remote image:")
if(len(description_results.captions) == 0):
    print("No description detected.")
else:
    for caption in description_results.captions:
        print("'{}' with confidence {:.2f}%".format(caption.text,caption.confidence * 100))

#image category using computer vision - extracts categories from a remote image along with a confidence score
print("Categorizing an image - remote")
remote_image_features = ["categories"]
categorize_results_remote = computervision_client.analyze_image(remote_image_url,remote_image_features)

#printing results with confidence scores
print("Categories from remote image:")
if(len(categorize_results_remote.categories) == 0):
    print("No categories detected.")
else:
    for category in categorize_results_remote.categories:
        print("'{}' with confidence {:.2f}%".format(category.name,category.score*100))

#image tagging using computer vision - It returs a tag for each object of the image
print("Tag a remote image")
tags_result_remote = computervision_client.tag_image(remote_image_url)

#printing results with confidence scores
if(len(tags_result_remote.tags)==0):
    print("No tags detected.")
else:
    for tag in tags_result_remote.tags:
        print("'{}' with confidence {:.2f}%".format(tag.name,tag.confidence * 100))


#object detection using computer vision - This API returns bounding box co-ordinates in pixels for each object found
print("Detecting objects in local image:")
img = Image.open(requests.get(remote_image_url, stream = True).raw)
folder="C:/Applications/Machine Learning/MDS-Data Science/mds-deakin/Object Detection using ComputerVision/Dataset/"
filepath = folder + "/inputs/" + os.path.basename(remote_image_url)

img.save(filepath)


for file in os.listdir(folder + "/inputs/"):
    path_of_file = os.path.join(folder + "/inputs/",file)
    img = Image.open(path_of_file)
    image_draw = ImageDraw.Draw(img)

    with open(path_of_file,mode='rb') as imgstream:
        #print(path_of_file)
        detect_objects_results = computervision_client.detect_objects_in_stream(imgstream)
        for object in detect_objects_results.objects:
            #print detected objects results with bounding boxes
            print("Detecting objects in remote image:")
            

            if len(detect_objects_results.objects) == 0:
                print("No objects detected.")
            else:
                for object in detect_objects_results.objects:
                    
                    print("object name {} with confidence {:.2f}% at location {}, {}, {},{}".format(object.object_property, 
                                                                                                    object.confidence * 100, 
                                                                                                    object.rectangle.x, 
                                                                                                    object.rectangle.x + object.rectangle.w, 
                                                                                                    object.rectangle.y, 
                                                                                                    object.rectangle.y + object.rectangle.h))
                    left = object.rectangle.x
                    top = object.rectangle.y
                    width = object.rectangle.w 
                    height = object.rectangle.h

                    shape = [(left,top),(left+width,top+height)]
                    image_draw.rectangle(shape,outline="green",width=5)
                    text = f'{object.object_property } ({object.confidence *100}%)'
                    image_draw.text((left+5-1,top+height-30+1),text,(0,0,255),font)
                    image_draw.text((left+5,top+height-30),text,(0,0,255),font)
        img.save(folder  + "/outputs/" + os.path.basename(remote_image_url) + "_boundingboxes.jpeg")



        









