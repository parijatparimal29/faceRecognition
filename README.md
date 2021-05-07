# faceRecognition

The objective of the project is to compare the inference times of face recognition models with and without using YOLO to finding frames labeled as persons and use this frame for face recognition instead of the entire image.

Instructions to run:

Requirement:
All requirements can be satisfied by installing the below libraries using 'pip' command.

```bash
pip install numpy
pip install pillow
pip install face_recognition
pip install matplotlib
pip install mtcnn
pip install scipy
pip install keras_vggface
pip install cv2
pip install tensorflow
```

First we need to load pretrained weights.
```bash
chmod +x weights.sh
bash weights.sh
```
To run the the face recognition models, use the below command on terminal.
```bash
python face_rec.py --infile <dir_of_known> --files <dir_of_unknown> --v <0 or 1 for verbose>
```
