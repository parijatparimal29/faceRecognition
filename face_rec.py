import argparse
import time
import numpy as np
import face_recognition
from numpy import asarray
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
from PIL import Image
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import tensorflow as tf
import cv2
import os

tf.get_logger().setLevel('ERROR')

def draw_faces(filename, result_list):
    
    # load the image
    data = pyplot.imread(filename)
    
    # plot each face as a subplot
    for i in range(len(result_list)):
        
        # get coordinates
        x1, y1, width, height = result_list[i]['box']
        x2, y2 = x1 + width, y1 + height
        
        # define subplot
        pyplot.subplot(1, len(result_list), i+1)
        pyplot.axis('off')
        
        # plot face
        pyplot.imshow(data[y1:y2, x1:x2])
        
    # show the plot
    pyplot.show()

def extract_faces(detector, filename, required_size=(224, 224)):
    
    # load the image
    data = pyplot.imread(filename)
    faces = []
    frames = detector.detect_faces(data)
    
    # plot each face as a subplot
    for i in range(len(frames)):
        
        # get coordinates
        x1, y1, width, height = frames[i]['box']
        x2, y2 = x1 + width, y1 + height
        
        face = data[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        
        faces.append(face_array)
    
    return faces

def get_prediction(model, face):
    
    in_features = asarray(face, 'float32')
    in_features = preprocess_input(in_features, version=2)
    
    return model.predict(in_features)

def is_match(known_embedding, candidate_embedding, verbose=1, thresh=0.5):
    
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    
    if score <= thresh:
        if(verbose):
            print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
            
        return True
    else:
        if(verbose):
            print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))
        
        return False

def model_1(model, known_dir, unknown_dir, verbose):
    
    detector = MTCNN()
    start_time = time.time()
    
    known_names = []
    known_preds = []
    for file in os.listdir(known_dir):
        known_file = known_dir + '/' + file
        known_face = extract_faces(detector, known_file)
        known_pred = get_prediction(model, known_face)
        known_preds.append(known_pred)
        known_names.append(file.split('.')[0])
    
    matching_faces = []
    match = []
    time_to_run = []
    
    if(verbose):
        print("\nModel 1: - Detected faces\n")
    for file in os.listdir(unknown_dir):
        unknown_file = unknown_dir + '/' + file
        faces = extract_faces(detector, unknown_file)
        
        for face in faces:
            if(verbose):
                pyplot.imshow(face)
                pyplot.show()
            pred = get_prediction(model, [face])
            for i in range(len(known_preds)):
                if(is_match(known_preds[i], pred, verbose)):
                    matching_faces.append(face)
                    match.append(known_names[i])
            time_to_run.append(time.time() - start_time)
    
            
    return (matching_faces, match, time_to_run)

def get_outputs_names(net):
    # Get the names of all the layers in the network
    layers_names = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layers_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

    right = right + margin

    return left, top, right, bottom

def post_process(frame, outs, conf_threshold, nms_threshold, required_size=(224, 224)):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only
    # the ones with high confidence scores. Assign the box's class label as the
    # class with the highest score.
    confidences = []
    boxes = []
    final_boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        
        left, top, right, bottom = refined_box(left, top, width, height)
        
        face = frame[top:bottom, left:right]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        final_boxes.append(face_array)
        
    return final_boxes

def extract_faces_with_YOLO(net, filename):
    
    CONF_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    
    cap = cv2.VideoCapture(filename)
    has_frame, frame = cap.read()
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),[0, 0, 0], 1, crop=False)
    net.setInput(blob)
    outs = net.forward(get_outputs_names(net))
    faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
    
    return faces

def model_2(model, known_dir, unknown_dir, verbose):
    
    cfg = 'yolov3-face.cfg'
    wts = 'model-weight/yolov3-wider_16000.weights'
    net = cv2.dnn.readNetFromDarknet(cfg, wts)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    start_time = time.time()
    
    known_names = []
    known_preds = []
    for file in os.listdir(known_dir):
        known_file = known_dir + '/' + file
        known_face = extract_faces_with_YOLO(net, known_file)
        known_pred = get_prediction(model, known_face)
        known_preds.append(known_pred)
        known_names.append(file.split('.')[0])
    
    matching_faces = []
    match = []
    time_to_run = []
    
    if(verbose):
        print("\nModel 2: - Detected faces\n")
    for file in os.listdir(unknown_dir):
        unknown_file = unknown_dir + '/' + file
        faces = extract_faces_with_YOLO(net, unknown_file)
        
        for face in faces:
            if(verbose):
                pyplot.imshow(face)
                pyplot.show()
            pred = get_prediction(model, [face])
            for i in range(len(known_preds)):
                if(is_match(known_preds[i], pred, verbose)):
                    matching_faces.append(face)
                    match.append(known_names[i])
            time_to_run.append(time.time() - start_time)
            
    return (matching_faces, match, time_to_run)

def read_img(path):
    img = cv2.imread(path)
    (h, w) = img.shape[:2]
    width = 500
    ratio = width / float(w)
    height = int(h * ratio)
    return cv2.resize(img, (width, height))

def model_3(model, known_dir, unknown_dir, verbose=0):
    
    start_time = time.time()

    known_encodings = []
    known_names = []
    for file in os.listdir(known_dir):
        img = read_img(known_dir + '/' + file)
        img_enc = face_recognition.face_encodings(img)[0]
        known_encodings.append(img_enc)
        known_names.append(file.split('.')[0])
    
    matching_faces = []
    match = []
    time_to_run = []
        
    for file in os.listdir(unknown_dir):
        img = read_img(unknown_dir + '/' + file)
        enc = face_recognition.face_encodings(img)
        if(len(enc)<1):
            continue
        img_enc = enc[0]
    
        results = face_recognition.compare_faces(known_encodings, img_enc)
        for i in range(len(results)):
            if results[i]:
                name = known_names[i]
                (top, right, bottom, left) = face_recognition.face_locations(img)[0]
                matching_faces.append(img[top:bottom, left:right])
                match.append(known_names[i])
        
        time_to_run.append(time.time() - start_time)
            
    return (matching_faces, match, time_to_run)
    
def run_model(known_dir, unknown_dir, verbose=0):
    
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    strt = time.time()
    faces_1, match_1, time_1 = model_1(model, known_dir, unknown_dir, verbose)
    print("Model 1 completed in time", time.time() - strt)
    strt = time.time()
    faces_2, match_2, time_2 = model_2(model, known_dir, unknown_dir, verbose)
    print("Model 2 completed in time", time.time() - strt)
    strt = time.time()
    faces_3, match_3, time_3 = model_3(model, known_dir, unknown_dir, verbose)
    print("Model 3 completed in time", time.time() - strt)
    
    pyplot.figure(figsize=(10, 10))
    pyplot.plot(time_1, label="Model 1")
    pyplot.plot(time_2, label="Model 2")
    pyplot.plot(time_3, label="Model 3")
    pyplot.xlabel("Number of faces")
    pyplot.ylabel("Time (in seconds)")
    pyplot.legend()
    pyplot.show()
    
    print("\nModel 1: - MTCNN\n")
    fig, ax = pyplot.subplots(1,len(faces_1),figsize=(20, 20*len(faces_1)))
    for i in range(len(faces_1)):
        ax[i].imshow(faces_1[i])
        ax[i].set_title(match_1[i])
    pyplot.show()
    
    print("\nModel 2: - YOLO\n")
    fig, ax = pyplot.subplots(1,len(faces_2),figsize=(20, 20*len(faces_2)))
    for i in range(len(faces_2)):
        ax[i].imshow(faces_2[i])
        ax[i].set_title(match_2[i])
    pyplot.show()
    
    print("\nModel 3: - Face recognition implemented library\n")
    fig, ax = pyplot.subplots(1,len(faces_3),figsize=(20, 20*len(faces_3)))
    for i in range(len(faces_3)):
        ax[i].imshow(faces_3[i])
        ax[i].set_title(match_3[i])
    pyplot.show()
    
    tpi_1 = time_1[-1] / 8
    tpf_1 = time_1[-1] / 36
    print("Model 1: Time per file = {}, Time per face = {}".format(tpi_1, tpf_1))
    
    tpi_2 = time_2[-1] / 8
    tpf_2 = time_2[-1] / 36
    print("Model 2: Time per file = {}, Time per face = {}".format(tpi_2, tpf_2))
    
    tpi_3 = time_3[-1] / 6
    tpf_3 = time_3[-1] / 6
    print("Model 3: Time per file = {}, Time per face = {}".format(tpi_3, tpf_3))
            

if __name__ == '__main__':
    '''
        Main function => Handles inputs, runs chosen models and outputs the times and accuracies obtained with the chosen model
    '''
    
    parser = argparse.ArgumentParser()
    
    # Parse input from command line
    parser.add_argument('--infile', help='Enter path of known files', default='test1/known')
    parser.add_argument('--files', help='Enter path of unknown files', default='test1/unknown')
    parser.add_argument('--v', help='Verbose output - 0:False 1:True', default=0)
    args = parser.parse_args()
    
    # Use arguments to perform requested action
    verbose = int(args.v)
    
    # Run the model based on the inputs
    run_model(args.infile, args.files, verbose)
        
