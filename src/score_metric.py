from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from curses import baudrate

import tensorflow as tf
import argparse
# from asyncio.timeouts import timeout
import facenet
import os
import sys
import math
import pickle
import datetime
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
from math import sqrt
from glob import glob
import time
import shutil

def update_model():
    src = "facemodel.pkl" # source file path
    dst = "Models/facemodel.pkl" # destination file path
    if os.path.isfile(src):
        if os.path.isfile (dst): # check if destination file exists
            os.remove (dst) # delete the existing file
        shutil.move(src, dst) # move the source file to the destination
        print ("File is moved and overwritten successfully.")
    else:
        print ("New model not found.")


def pass_score(accuracy_current):
    if accuracy_current is not None:
        with open("model_score.txt", "r+") as f:
            lines = f.readlines()
            # accuracy_previous = float(lines[-1])
            accuracy_threshold = 0.9

            if accuracy_current > accuracy_threshold:
                f.write(f"\n{accuracy_current}")
                update_model()
                # serial_port = serial.Serial(port = "COM3", baudrate = 9600, timeout = 0.5)
                # serial_port.close()
                # serial_port.open()
                # serial_port.write('m'.encode())
                print("mo cua")
            else: 
                os.remove("facemodel.pkl")
                raise ValueError(f"Current accuracy {accuracy_current} is less than previous accuracy {accuracy_threshold}")
            f.close()
    else:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=0)
    args = parser.parse_args()
    
    # Cai dat cac tham so can thiet
    MINSIZE = 20
    THRESHOLD = 0.5
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'Models/facemodel.pkl'
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

    # Load model da train de nhan dien khuon mat - thuc chat la classifier
    with open(CLASSIFIER_PATH, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")

    with tf.Graph().as_default():

        # Cai dat GPU neu co
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load model MTCNN phat hien khuon mat
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Lay tensor input va output
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Cai dat cac mang con
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "src/align")

            people_detected = set()
            person_detected = collections.Counter()
            img_paths = []
            for ext in ["png", "jpg", "jpeg"]:
                img_paths += glob(args.path + r"\*\*." + f"{ext}")
            print(img_paths)
            true_positive = 0
            if len(img_paths) > 0:
                for img_path in img_paths:
                    label = img_path.split("\\")[-2]
                    # Lay hinh anh tu file video
                    frame = cv2.imread(img_path)
                    scaled = cv2.resize(frame, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                        interpolation=cv2.INTER_CUBIC)
                    scaled = facenet.prewhiten(scaled)
                    scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                    feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                    emb_array = sess.run(embeddings, feed_dict=feed_dict)
                    
                    # Dua vao model de classifier
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[
                        np.arange(len(best_class_indices)), best_class_indices]
                    
                    # Lay ra ten va ty le % cua class co ty le cao nhat
                    best_name = class_names[best_class_indices[0]]
                    print(best_name)
                    if best_name.lower() == label.lower() and best_class_probabilities[0] >= THRESHOLD:
                        true_positive += 1    
                        print(true_positive)
                    print("Label: {},   Name: {},   Probability: {},    img: {}".format(label, best_name, best_class_probabilities, img_path.split("\\")[-1]))
                
                accuracy = true_positive/len(img_paths)
            else:
                accuracy = None

            print("=======> Accuracy:", accuracy)
            pass_score(accuracy)

main()
