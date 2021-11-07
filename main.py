# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 18:44:25 2021

@author: Rohit Badugu
"""
import cv2
import numpy as np
import os
import glob

## import the handfeature extractor class
from handshape_feature_extractor import HandShapeFeatureExtractor
from frameextractor import frameExtractor
from numpy import genfromtxt
from scipy import spatial


def find_gesture_number(vect, penul_layer):
    cost_dist = []

    for p in penul_layer:
        cost_dist.append(spatial.distance.cosine(vect,p))
        ges = cost_dist.index(min(cost_dist))+1
    
    return ges


def getPenultimateLayer(frames_path,file_name):
    files_list = []

    path = os.path.join(frames_path,"*.png")
    frames = glob.glob(path)
    frames.sort()
    files_list = frames

    prediction_vector = get_vectors_for_frames(files_list)
    np.savetxt(file_name, prediction_vector, delimiter=",")



def get_vectors_for_frames(files_list):

    vectors = []
    prediction_model = HandShapeFeatureExtractor.get_instance()

    for frame in files_list:
        img = cv2.imread(frame, 0)

        results = prediction_model.extract_feature(img)
        results = np.squeeze(results)

        vectors.append(results)
    
    return vectors


# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

train_video_location = os.path.join('traindata')
train_path_to_videos = os.path.join(train_video_location,"*.mp4")
training_videos = glob.glob(train_path_to_videos)							#load all videos

count = 1
train_path_to_frames = os.path.join(train_video_location, "frames")

for video in training_videos:
	#extracting frames
	frameExtractor(video, train_path_to_frames, count)
	count+=1
	if count > 17:
		break
	
train_data_file = 'training_vector.csv'
getPenultimateLayer(train_path_to_frames, train_data_file)



# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video

test_video_location = os.path.join('test')
test_path_to_videos = os.path.join(test_video_location,"*.mp4")
testing_videos = glob.glob(test_path_to_videos)							#load all videos

count = 1
test_path_to_frames = os.path.join(test_video_location, "frames")

for video in testing_videos:
	#extracting frames
	frameExtractor(video, test_path_to_frames, count)
	count+=1
	
test_data_file = 'testing_vector.csv'
getPenultimateLayer(test_path_to_frames, test_data_file)



# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================

training_data = genfromtxt(train_data_file, delimiter=',')
test_data = genfromtxt(test_data_file, delimiter=',')

results = []

for data in test_data:
    results.append(find_gesture_number(data, training_data))

np.savetxt('Results.csv', results, delimiter=",", fmt='% d')
