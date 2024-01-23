from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
import datetime
import facenet
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
from math import sqrt
import time
import csv
import mysql.connector
from mysql.connector import Error

timeSent = None

def import_mysql(user_id):

    current_date = datetime.date.today().strftime("%Y-%m-%d")
    current_time_ = datetime.datetime.now().strftime("%H:%M:%S")
    try:
        # Establish a connection to MySQL
        # connection = mysql.connector.connect(**connection_config)
        connection = mysql.connector.connect(
        host="localhost",
        user = "root",
        passwd = "123456789",
        database = "face_recorgnize",
        )
        cursor = connection.cursor()
        cursor.execute("SHOW TABLES")
        for db in cursor:
            print(db)
        data_test = [(user_id, current_date, current_time_)]

        insert_query = f"INSERT INTO students (name, date, time) VALUES (%s, %s, %s)"
        cursor.executemany(insert_query, data_test)

        # # Commit the changes
        connection.commit()

    except Error as e:
        print(f"Error: {e}")

    finally:
        # Close the connection
        if connection.is_connected():
            cursor.close()
            connection.close()

def import_csv(user_id):
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    current_time_ = datetime.datetime.now().strftime("%H:%M:%S")
    array = []
    array.append(user_id)
    array.append(current_date)
    array.append(current_time_)
    file_name_csv = "giam_sat_hoc_sinh_" + current_date + ".csv"
    folder_name = "worklog"
    file_path_csv = os.path.join(folder_name, file_name_csv)
    with open(file_path_csv, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(array)
    
    
    
def register_log(user_id, current_time):
    global timeSent
    current_date = datetime.date.today().strftime("%Y-%m-%d")
    current_time_ = datetime.datetime.now().strftime("%H:%M:%S")
    # print(current_time_)
    file_name = current_date + ".txt"
    folder_name = "worklog"
    file_path = os.path.join(folder_name, file_name)
    with open(file_path, "a") as f:
        # row_count_txt = sum(1 for line in f)
        f.write(f"\n{user_id}\t{current_date}\t{current_time_}")
        f.close()
    
    # Replace these values with your specific details
    file_path = file_path

    mysql_table = 'students'

    # Call the function to import data
    import_mysql(user_id)
    import_csv(user_id)

    
    if timeSent == None or time.time()*1000 - timeSent > 7000:
        timeSent = time.time()*1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path of the video you want to test on.', default=1)
    args = parser.parse_args()
    
    # Cai dat cac tham so can thiet
    MINSIZE = 20
    THRESHOLD = [0.8, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160
    CLASSIFIER_PATH = 'Models/facemodel.pkl'
    VIDEO_PATH = args.path
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

            cap = cv2.VideoCapture(0)
            # cap = cv2.VideoCapture("http://192.168.1.7:8080/video")


            while (cap.isOpened()):
                start = time.time()
                # Doc tung frame
                ret, frame = cap.read()
                h, w, c = frame.shape

                # Phat hien khuon mat, tra ve vi tri trong bounding_boxes
                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                try:
                    # Neu co it nhat 1 khuon mat trong frame
                    if faces_found > 1:
                        cv2.putText(frame, "Please recognize each person.", (0, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (50, 50, 255), thickness=2)
                    elif faces_found == 1:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]
                            # Loai bo nhung object o qua xa
                            if sqrt((bb[i][1]-bb[i][3])**2+(bb[i][0]-bb[i][2])**2) < sqrt(h**2+w**2)*0.2:
                                continue
                            # Cat phan khuon mat tim duoc
                            cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
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
                            print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))

                            # Ve khung mau xanh quanh khuon mat
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20

                            # Neu ty le nhan dang > 0.5 thi hien thi ten
                            if best_class_probabilities > 0.7:
                                name = class_names[best_class_indices[0]]
                                current_time = datetime.datetime.now()
                                register_log(user_id=name, current_time=current_time)
                            else:
                                # Con neu <=0.5 thi hien thi Unknown
                                name = "Unknown"
                                
                            # Viet text len tren frame    
                            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                            cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (255, 255, 255), thickness=1, lineType=2)
                            person_detected[best_name] += 1
                except:
                    pass
                end = time.time()
                fps = round(1/(end - start))
                cv2.putText(frame, f"fps: {fps}", (0, 450), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (50, 50, 255), thickness=2)
                # Hien thi frame len man hinh
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


main()
