# Webcam-based-Face-Recognition-using-Deep-Learning

Face Detection and landmark detection : It is done using Multi-task Cascaded Convolutional Networks(MTCNN) model. Used a pretrained model of MTCNN to detect face, to find the bounding box and landmark detection.
Reference : https://arxiv.org/pdf/1604.02878.pdf
            https://github.com/ipazc/mtcnn

Face Recognition : The face Recognition is done using Facenet model. Used a pretrained facenet model to compare the captured image/Input image with all images in database to recognize the correct face using clustering algorithm.
Reference : https://arxiv.org/pdf/1503.03832.pdf

MTCNN Pretrained Model Installation: Follow link -https://github.com/ipazc/mtcnn
FaceNet Pretrained Model Installation: Use Pretrained model in the link - https://github.com/arunmandal53/facematch
 
Dependency :
 1. Python 3.6
 2. tensorflow r1.12 or above 
 3. OpenCV 4.1.1 or above

How to Run the Code :
 1. Download and extract the code into a folder.
 2. Download the pretrained model from the link given and place the files in the extracted folder in step 1.
 3. Create or Update the database by inserting images of the people among among which face recognition is to be done. After inserting all images, select all images and rename the images as :Person<Number>
    example :Person1,Person2 etc. Your database can be of any size, but speed of the code depends on size of database. 
 4. Update the xls sheet with Person number and name of corresponding person.
 5. Now execute the Face_Rec.py script.

Execution and Outputs:
 1. The user can use either image captured throuh webcam automatically or the image from Input folder. The user can choose method of input during execution.
 2. After executing the program name of the person is displayed, if the input is a person from the database. 
 3. The image captured can be seen inside captured folder.
 4. The detected face with bounding box, cropped part of image and land mark detected images can be seen inside folder check.

Note :
1. Use the port number for the webcam according to Device Configration of your sytem.
   (Edit :Face_Rec.py,  Line 14 , camera = cv2.VideoCapture(0)) 
2. Use webcam with resolution greater than 640x480 for better accuracy.
         
