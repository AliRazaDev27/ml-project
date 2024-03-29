import os
import numpy as np
#import matplotlib.pyplot as plt
#%matplotlib inline


import sys
import caffe

#plt.rcParams['figure.figsize'] = (10, 10)
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

mean_filename='mean.binaryproto'
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0]

age_net_pretrained='age_net.caffemodel'
age_net_model_file='age_deploy.prototxt'
age_net = caffe.Classifier(age_net_model_file, age_net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

gender_net_pretrained='gender_net.caffemodel'
gender_net_model_file='gender_deploy.prototxt'
gender_net = caffe.Classifier(gender_net_model_file, gender_net_pretrained,
                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))

age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list=['Male','Female']

example_image = 'example_image.jpg'
input_image = caffe.io.load_image(example_image)
#_ = plt.imshow(input_image)

prediction = age_net.predict([input_image]) 

print ("predicted age:", age_list[prediction[0].argmax()])

prediction = gender_net.predict([input_image]) 

print ("predicted gender:", gender_list[prediction[0].argmax()])




