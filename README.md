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
The difference in runtimes can be noted as shown below:
![image](https://user-images.githubusercontent.com/54210698/117391915-fb6a5080-aebe-11eb-949a-843c421f838a.png)

Model 1 uses MTCNN to detect faces and VGGface to recognize them. <br>
Model 2 uses YOLO to detect faces and VGGface to recognize them.<br>
Model 3 uses face_recognition library to detect faces recognize them. [Unsuccessful for files with multiple images - hence very small number of faces detected]

<br><br>
Below are the outputs of MTCNN and YOLO models. We can see that MTCNN misclassifies 2 images, whereas the model with YOLO has 100% accuracy.
![image](https://user-images.githubusercontent.com/54210698/117392192-9105e000-aebf-11eb-90ed-ea69312c4812.png)
![image](https://user-images.githubusercontent.com/54210698/117392288-c27eab80-aebf-11eb-8a9b-966a300b71b9.png)

![image](https://user-images.githubusercontent.com/54210698/117392235-a3801980-aebf-11eb-9cbe-38be5b42bad8.png)


