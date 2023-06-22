########################################################################
#     Mosquito_main.py
#
#       Locate mosquito images on the file image
#           mark the bounding box,
#               Determine the type of mosquito
#
#       Mike Pastor  June 20, 2023

#  python -m pip install  opencv-python

import os
from datetime import datetime
import numpy as np
import pandas as pd
import random

import cv2
import tensorflow as tf

import sklearn
from keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.activations import linear, relu, sigmoid


#  Images to process while testing
#    0 = all records
#
MAX_IMAGES=0

# Number of EPOCHS to train the model
#
EPOCH_COUNT=2

# Base size for the images
#
# IMAGE_X_SIZE=1024
# IMAGE_Y_SIZE=768
IMAGE_X_SIZE=1000
IMAGE_Y_SIZE=750

# Filesystem path for Images
IMAGE_PATH = './Images/'
IMAGE_PATH_TEST = './Images-TEST/'



##############################################################################
#   Functions

###########################################################
#       Display the image and wait for a keypress
def viewImage( myImage ):
    cv2.imshow( 'My Title', myImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



###############################################################################
#  Process starts here...
#
global_now = datetime.now()
global_current_time = global_now.strftime("%H:%M:%S")
print("##############  Mosquito-main   Starting up...  - Current Time =",
      global_current_time)


############################################################################
#    Load the meta datasets
#
print('Loading meta data...')

trainDF = pd.read_csv( 'train.csv' )
print( trainDF.head() )
testDF = pd.read_csv( 'test_phase1_v2.csv' )
print( testDF.head() )

print( '###   trainDF  size= ', len( trainDF ), 'testDF  size= ', len( testDF ))

##############################################
#     Dictionary for classification
#
possibleLabels = trainDF.class_label.unique()
labelDictionary = {}

# Loop over the labels
#  #  {'albopictus': 0, 'culex': 1, 'anopheles': 2, 'culiseta': 3, 'japonicus/koreicus': 4, 'aegypti': 5}
for index, possible_label in enumerate(possibleLabels):
    labelDictionary[possible_label] = index
print( labelDictionary )

# Reverse lookup also
id2LabelDictionary = {y:x for x,y in labelDictionary.items()}

# Create a new field for the label number
#   TEST is unlabeled
#
trainDF['class_label_number'] = trainDF.class_label.replace( labelDictionary )
#  testDF['class_label_number'] = testDF.class_label.replace( labelDictionary )
print( 'Class label distribution avg/total: ', trainDF['class_label_number'].mean(), len(trainDF) )

###########################################################################
#      Process the images into a clean array
#           this becomes our X independent data
#
#      For all images in the Train dataset...
#
x = 0
imageList=[]
classList=[]

for idx, data in trainDF.iterrows():

    # Get the path to the Image
    filePath = IMAGE_PATH + data.img_fName
    # print( idx, filePath, data.class_label, data.bbx_xtl, data.bbx_ytl )

    # Read the Image data
    img = cv2.imread( filePath, cv2.IMREAD_COLOR )
    # print('Img shape= ', img.shape, type(img) )

    # draw bounding box
    color=(255, 0, 0)  # blue
    thickness=4   # 2 pixels
    startPt = ( data.bbx_xtl, data.bbx_ytl )
    endPt = (data.bbx_xbr, data.bbx_xbr)

    cv2.rectangle(img,  startPt,  endPt, color, thickness)
    # print('BBX coords= ',  startPt, endPt )

    # Convert to gray scale
    #
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize
    # width= int( img.shape[0] / 10 )
    # height = int( img.shape[1] / 10)
    width = int(IMAGE_X_SIZE)
    height = int(IMAGE_Y_SIZE)

    img = cv2.resize(img, (width, height))

    # Set our x, y lists
    imageList.append( img )
    classList.append( data.class_label_number )


    #########################################
    x += 1
    if ( x % 100 == 0):
        print('Row --> ', x)

    if ( MAX_IMAGES != 0 and x > MAX_IMAGES):
        break;


print( 'Image lists processed -   X length= ', len( imageList ), ' Y Length= ', len( classList ) )


########################################################
#     Similar treatment for our TEST dataset
#
x = 0
imageListTEST=[]

for idx, data in testDF.iterrows():

    # Get the path to the Image
    filePath = IMAGE_PATH_TEST + data.img_fName

    # Read the Image data
    img = cv2.imread( filePath, cv2.IMREAD_COLOR )
    # print('Img shape= ', img.shape, type(img) )

    # Convert to gray scale
    #
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize
    #
    img = cv2.resize(img, (IMAGE_X_SIZE, IMAGE_Y_SIZE))

    # Set our x  list
    imageListTEST.append( img )
    #  classListTEST.append( data.class_label_number )


    #########################################
    x += 1
    if ( x % 100 == 0):
        print('Row --> ', x)

    if ( MAX_IMAGES != 0 and x > MAX_IMAGES):
        break;


print( '##### TEST X length= ', len( imageListTEST )  )


#########################################################################################
#    Setup our datasets
#

#  Set our Y dataset
numpyY = np.array( classList )
numpyY = numpyY.reshape( (len(numpyY), 1) )

# Set our X dataset
numpyX = np.array( imageList )
numpyX = numpyX.reshape( (len(numpyX), (IMAGE_X_SIZE * IMAGE_Y_SIZE), 1) )

# Same for TEST X datset
numpyXTEST = np.array( imageListTEST )
numpyXTEST = numpyXTEST.reshape( (len(numpyXTEST), (IMAGE_X_SIZE * IMAGE_Y_SIZE), 1) )


print( '########  X shape= ', numpyX.shape, ' Y shape= ', numpyY.shape )
print( '######## TEST  X shape= ', numpyXTEST.shape )



##########################################################################################
#  Model construction


#######################################################
# Summary of steps
#     Get more Training examples  -> Fixes High Variance
#       Try smaller set of features  ->  Fixes High Variance
#       Try Additional features ->  Fixes High Bias
#       Try adding polynomial features (x squared, etc) -> Fixes High Bias
#       Decrease Lambda ->  Fixes High Bias
#       Increase Lambda ->   Fixes High Variance

#  Decrease Regularizer  Lambda to fight  BIAS  - increase IT to fight VARIANCE
#
regLambda = 0.01    # Lambda for Regularization equation

model = Sequential(
    [

        tf.keras.Input(shape=( (IMAGE_X_SIZE * IMAGE_Y_SIZE), )),
        Dense(512, activation='relu', name='layer1' ),
        Dense(256, activation='relu', name='layer2' ),
        Dense(64, activation='relu', name='layer3' ),
        Dense( len(labelDictionary), activation='linear', name='layer4' )

        # Dense(len(labelDictionary), activation='softmax')

        # tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(784, 1) ),

        # tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        # tf.keras.layers.MaxPooling2D(2, 2),
        # tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        # tf.keras.layers.MaxPooling2D(2, 2),
        #
        # tf.keras.layers.Flatten(),
        # tf.keras.layers.Dense(512, activation='relu'),
        # tf.keras.layers.Dense(10, activation='softmax')

        ### END CODE HERE ###
    ], name="pennwick_model"
)

model.summary()


#   Add the Loss function  and the Gradient Descent Learning Rate
#
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy( from_logits=True ),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)


# Fit the model to the X and Y data
#    imageList, classList,

print( 'About to FIT Model: ', numpyX.shape, numpyY.shape )
history = model.fit(
    numpyX, numpyY,
    epochs=EPOCH_COUNT
)


print( "Model is Fit and ready to predict...")

####################################################################


################################################
# Predict using the test X dataset
#
#  prediction = model.predict(numpyX.reshape(len(numpyX), (IMAGE_X_SIZE * IMAGE_Y_SIZE)))
prediction = model.predict(numpyXTEST.reshape(len(numpyXTEST), (IMAGE_X_SIZE * IMAGE_Y_SIZE)))
# prediction = model.predict(numpyX.reshape(numpyXTEST))

prediction_p = tf.nn.softmax(prediction)
#  yhat = np.argmax(prediction_p)

print( "prediction = ", prediction )
# print( "prediction_p = ", prediction_p )

predictions_flat = np.argmax(prediction_p, axis=1).flatten()

print( f"We have {len(prediction_p)} predictions on the TEST Dataset yhat= ",  predictions_flat)

print( 'Average prediction:  ',  np.average ( predictions_flat ) )

#############################################################################
#  Setup the submission file
#
def prepareSubmissionFile( predictions_list ):

    keyList = list(labelDictionary.keys())

    # Add necessary columns to testDF
    #
    testDF['bbx_xtl'] = 0
    testDF['bbx_ytl'] = 0
    testDF['bbx_xbr'] = 0
    testDF['bbx_ybr'] = 0

    print( 'Predictions List= ',  len(predictions_list) )
    print('testDF= ', len(testDF))

    # For all records in the testDF dataset...
    #
    x = 0
    maxPreds = len( predictions_list )

    for idx, data in testDF.iterrows():

        # print( data.img_fName, data.img_w, data.img_h )

        if idx >= maxPreds:
            break

        pred = predictions_list[idx]
        testDF.loc[idx, 'class_label'] = id2LabelDictionary[pred]

        # Mock up the bounding box for now
        width = data.img_w / 2
        testDF.loc[idx, 'bbx_xtl'] = width
        height = data.img_h / 2
        testDF.loc[idx, 'bbx_ytl'] = height

        width2 = (data.img_w / 2) + 100
        testDF.loc[idx, 'bbx_xbr'] = width2
        height2 = (data.img_h / 2) + 100
        testDF.loc[idx, 'bbx_ybr'] = height2


    testDF.to_csv( 'MosquitoAlert-submission.csv', index=False )
    print('Wrote submission file successfully')

prepareSubmissionFile( predictions_flat )


# Mission Complete!
##################################################################################
global_later = datetime.now()
print("#####  Mosquito-main    - Total EXECUTION Time =", (global_later - global_now), global_later )

exit(-9)













##############################################################################################
#
# items = os.listdir('./Images')
# #  print (items)
# for each_image in items:
#     if each_image.endswith(".jpeg"):
#         print (each_image)
#         full_path = "./Images/" + each_image
#         print (full_path)
#         img = cv2.imread(full_path)
#         viewImage( img )
#         break;



# img = cv2.imread(
#     './Images/0000c8c4-e87a-44b8-84d4-8bebcf75645c.jpeg',
#     cv2.IMREAD_COLOR )

img = cv2.imread(
    './Images/subway.jpg',
#    './Images/0000c8c4-e87a-44b8-84d4-8bebcf75645c.jpeg',
    cv2.IMREAD_COLOR )


viewImage( img )

print( img.shape )
#  Blue, green, red  0-255
#
print( img[0,0,:])

# Convert to gray scale
#
img_gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY)

print( img_gray.shape )
print( img_gray[0,0])
viewImage( img_gray )

# Now get the gradients
#
gradientX = cv2.Sobel( img_gray, cv2.CV_64F, 1, 0 )
gradientX = np.absolute( gradientX )
viewImage(  gradientX/np.max( gradientX ))
# print('X Gradient= ', gradientX)

# change scan direction for 'y'
gradientY = cv2.Sobel( img_gray, cv2.CV_64F, 0, 1 )
gradientY = np.absolute( gradientY )
viewImage(  gradientY/np.max( gradientY ))

# magnitude of gradient vector
#     combine X and Y
#
magnitudeGradientVector = \
    np.sqrt( gradientX**2  + gradientY**2)

viewImage( magnitudeGradientVector
           / np.max( magnitudeGradientVector))

# Detect the edges using Canny detection
#   Play with range to highlight objects
#
myEdges = cv2.Canny( img_gray, 50, 150 )
viewImage(myEdges)

# Hough transform for lines
#
lines = cv2.HoughLinesP(
    myEdges,
    rho=1,
    theta=1. * np.pi/180.0,
    threshold=20,
    minLineLength=25,
    maxLineGap=5,
)

myLines = img.copy()

for l in lines:

    x1, y1, x2, y2 = l[0]
    cv2.line(
        myLines, (x1, y1), (x2,y2),
        (0,0,255), thickness=3 )

viewImage(myLines)


circles = cv2.HoughCircles(
    img_gray,
    method=cv2.HOUGH_GRADIENT,
    dp=2,
    minDist=35,
    param1=150,
    param2=40,
    minRadius=15,
    maxRadius=35
)

print('Got circles= ', len( circles ))

myCircles = img.copy()

for x,y,r  in circles[0]:

    print('CIRCLE= ', x,y,r)
    cv2.circle(
        myCircles,
        (int(x),int(y)),
        int(int(r)),
        (0,0,255),
        thickness=3
    )

viewImage(myCircles)

img_blurred = cv2.GaussianBlur(
    img_gray,
    ksize=(21, 21),
    sigmaX=0,
)
viewImage(img_blurred)

circles = cv2.HoughCircles(
    img_blurred,
    method=cv2.HOUGH_GRADIENT,
    dp=2,
    minDist=35,
    param1=150,
    param2=40,
    minRadius=15,
    maxRadius=35
)

print('Got circles= ', len( circles ))

myCircles = img.copy()

for x,y,r  in circles[0]:

    #  print('CIRCLE= ', x,y,r)
    cv2.circle(
        myCircles,
        (int(x),int(y)),
        int(int(r)),
        (0,0,255),
        thickness=3
    )

viewImage(myCircles)

exit(-1)





# Start capturing video input from the camera
# cap = cv2.VideoCapture(0)

# from the file
##################################################################
#  Basic  read write functions

#  cv2.IMREAD_COLOR,  cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
#
img = cv2.imread( './Images/0000c8c4-e87a-44b8-84d4-8bebcf75645c.jpeg', cv2.IMREAD_COLOR )

#  Height, Width
print('IMG = Rows, Cols, Channels  ', img.shape )



while True:
    #  Read an image from the camera
    # success, frame = cap.read()
    # success, frame = cap.read()
    frame = cv2.imread('./Images/0000c8c4-e87a-44b8-84d4-8bebcf75645c.jpeg',
                                cv2.IMREAD_COLOR)
    #
    # if not success:
    #     sys.exit(
    #         'ERROR: Unable to read from webcam. Please verify your webcam settings.'
    #     )

    # Width and height by cv2
    # width = int( cap.get(3) )
    # height = int( cap.get(4) )

    width = img.shape[0]
    height = img.shape[1]

    # Call tensorflow for predictions
    #
    # Convert the image from BGR to RGB as required by the TFLite model.
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Create a TensorImage object from the RGB image.
    #  input_tensor = vision.TensorImage.create_from_array(rgb_image)
    input_tensor = tf.convert_to_tensor(rgb_image, dtype=tf.float32)

    print('Tensorflow converted image')

    #  canvas
    # newImage = np.zeros( frame.shape, np.uint8 )
    # smallerFrame = cv2.resize( frame, (0,0), fx=0.5, fy=0.5 )

    #   Split the img into a canvas
    # newImage[  :height//2, :width//2 ] = smallerFrame
    # newImage[height // 2:, :width // 2] = smallerFrame
    # newImage[:height // 2, width // 2:] = cv2.rotate( smallerFrame, cv2.cv2.ROTATE_180)
    # newImage[height // 2:, width // 2:] = smallerFrame

    #  Draw line on image
    # img = cv2.line( frame, (0,0), (width, height), (255,0,0), 10 )
    # img = cv2.line(img, (0, height), (width, 0), (0, 0, 255), 5 )

    # Draw rectangle
    # img = cv2.rectangle(frame, (100, 100), (200,200), (128,128,128), 5 )
    # img = cv2.circle( img, (300, 300), 60, (0,0,255), 5 )

    #  Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText( frame, 'Pennwick Rover', (10, 100), font, 1, (0,0,255), 5, cv2.LINE_AA )

    #  cv2.imshow( 'This Frame', img )

    #  img = cv2.resize(img, (640, 480) )
    #  Set the image to half the size
    img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    #  img = cv2.rotate( img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)



    # Draw bounding_box
    # bbox = detection.bounding_box
    # start_point = bbox.origin_x, bbox.origin_y
    # end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
    # cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 3)

    img = cv2.rectangle(frame, (100, 100), (200, 200), (128, 128, 128), 5)

    # write the file
    cv2.imwrite('new.jpg', img)


    cv2.imshow('Mikes window', img)

    cv2.waitKey(0)

    break


#  Cleanup the windows
#   img.release()
cv2.destroyAllWindows()


print( 'Setup Ok')





