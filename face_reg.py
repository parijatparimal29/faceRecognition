import argparse
import time
from numpy import asarray
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
from PIL import Image
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import tensorflow as tf

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

def extract_faces(filename, required_size=(224, 224)):
    
    # load the image
    data = pyplot.imread(filename)
    faces = []
    detector = MTCNN()
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

def is_match(known_embedding, candidate_embedding, verbose=0, thresh=0.5):
    
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

def model_1(model, known_file, known_name, test_files):
    
    start_time = time.time()
    
    known_face = extract_faces(known_file)
    known_pred = get_prediction(model, known_face)
    
    matching_faces = []
    match = []
    time_to_run = []
    
    for file in test_files:
        faces = extract_faces(file)
        
        for face in faces:
            pred = get_prediction(model, [face])
            matched = is_match(known_pred, pred)
            if(matched):
                matching_faces.append(face)
            match.append(matched)
            time_to_run.append(time.time() - start_time)
            
    return (matching_faces, match, time_to_run)


def model_2(model, known_file, known_name, test_files):
    
    start_time = time.time()
    
    known_face = extract_faces(known_file)
    known_pred = get_prediction(model, known_face)
    
    matching_faces = []
    match = []
    time_to_run = []
    
    for file in test_files:
        faces = extract_faces(file)
        
        for face in faces:
            pred = get_prediction(model, [face])
            matched = is_match(known_pred, pred)
            if(matched):
                matching_faces.append(face)
            match.append(matched)
            time_to_run.append(time.time() - start_time)
            
    return (matching_faces, match, time_to_run)

def model_3(model, known_file, known_name, test_files):
    
    start_time = time.time()
    
    known_face = extract_faces(known_file)
    known_pred = get_prediction(model, known_face)
    
    matching_faces = []
    match = []
    time_to_run = []
    
    for file in test_files:
        faces = extract_faces(file)
        
        for face in faces:
            pred = get_prediction(model, [face])
            matched = is_match(known_pred, pred)
            if(matched):
                matching_faces.append(face)
            match.append(matched)
            time_to_run.append(time.time() - start_time)
            
    return (matching_faces, match, time_to_run)
    
def run_model(known_file, known_name, test_files):
    
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    faces_1, match_1, time_1 = model_1(model, known_file, known_name, test_files)
    faces_2, match_2, time_2 = model_2(model, known_file, known_name, test_files)
    faces_3, match_3, time_3 = model_3(model, known_file, known_name, test_files)
    
    pyplot.plot(time_1, label="Model 1")
    pyplot.plot(time_2, label="Model 2")
    pyplot.plot(time_3, label="Model 3")
    pyplot.legend()
    pyplot.show()
    
    model_2_acc = 0
    for i in range(len(match_1)):
        if(match_1[i] == match_2[i]):
            model_2_acc += 1
    
    model_2_acc /= len(match_1)
    print("Model 2 accuracy =", model_2_acc)
    
    model_3_acc = 0
    for i in range(len(match_1)):
        if(match_1[i] == match_3[i]):
            model_3_acc += 1
            
    model_3_acc /= len(match_1)
    print("Model 3 accuracy =", model_3_acc)
            

if __name__ == '__main__':
    '''
        Main function => Handles inputs, runs chosen models and outputs the times and accuracies obtained with the chosen model
    '''
    
    parser = argparse.ArgumentParser()
    
    # Parse input from command line
    parser.add_argument('--infile', help='Enter filename of known face', default='test.jpeg')
    parser.add_argument('--inlabel', help='Enter label of known face', default='Test')
    parser.add_argument('--file', help='Enter filename of single test file', default='test.jpeg')
    parser.add_argument('--files', help='Enter filenames of multiple test file separated by commas (no spaces)', default=None)
    parser.add_argument('--v', help='Verbose output - 0:False 1:True', default=0)
    args = parser.parse_args()
    
    # Use arguments to perform requested action
    known_file = args.infile
    known_name = args.inlabel
    
    global verbose
    verbose = int(args.v)
    
    test_files = None
    if(args.files == None):
        test_files = []
        test_files.append(args.file)
    else:
        test_files = args.files.split(',')
    
    # Run the model based on the inputs
    run_model(known_file, known_name, test_files)
        