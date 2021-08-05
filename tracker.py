import h5py
import numpy as np
import cv2
import math
import statistics
import os
import argparse
import sys

from tensorflow.python.keras.models import load_model

model = load_model('model')



input_vid = 'output.mp4'
imageCounter = 1040

def realtimeClassifier(x_test):
    if len(x_test)<1:
        return []
    predicted_labels = []
    intensity_list = []
    for image in x_test:
        int_sum = 0
        G = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        G = G.ravel()
        for i in G:
            int_sum += i
        intensity_list.append(int_sum)
    threshold = statistics.median(intensity_list)
    for intensity_value in intensity_list:
        if intensity_value < threshold:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
    return predicted_labels

def imageClassifier(x_test):
    if x_test.ndim != 4:
        return []
    x_test = x_test.astype('float32') / 255
    #print(x_test.shape)
    predicted_labels = []
    prediction = model.predict(x_test)
    for i in range(len(prediction)):
        predicted_labels.append(np.argmax(prediction[i]))
    return predicted_labels

def imageCropper(Image,x,y,w,h):
    global imageCounter
    imageCounter += 1
    print(imageCounter)
    cropped = Image[y:h, x:w]
    x = str(imageCounter)
    path = 'D:/University/3992/Computer Vision/Project/Sample1/dataset'
    if cropped.size != 0 :
        cv2.imwrite(os.path.join(path, "data%s.jpg"%x), cropped)

def circleSpots(Image,color):
    Image = Image.max(axis=2)
    if color == "Red":
        color = (0, 0, 255)
    elif color == "Blue":
        color = (255, 0, 0)
    elif color == "Yellow":
        color = (0, 255, 255)




    n, C, stats, centroids = cv2.connectedComponentsWithStats(Image);
    K = np.zeros((700, 1050, 3), dtype=dotsImage[0].dtype)

    for i in range(1, n):
        cv2.circle(K, (int(centroids[i][0]), int(centroids[i][1])), 5, color, 5)


    return K

## In This Function We Detect The Players USing BackGround Subtraction And Binary Morphology

## KNN ALgorithm Is Chosen For BackGround Subtraction.
def calculateDifference(Image):

    global gatherData
    # Defining The ForeGround Mask
    fgMask = backSub.apply(Image)

    ## Thresholding The Result
    threshold = 190
    ret, T = cv2.threshold(fgMask, threshold, 255, cv2.THRESH_BINARY)

    ## Applying Binary Morphology With Opening And Closing On The Binary Image

    ## opening
    kernel = np.ones((3, 3), np.uint8)
    T = cv2.morphologyEx(T, cv2.MORPH_OPEN, kernel)

    ## closing
    kernel = np.ones((3, 3), np.uint8)
    T = cv2.morphologyEx(T, cv2.MORPH_CLOSE, kernel)

    ## The Players Are Detected AS Connected Components With Specific Stats
    n, C, stats, centroids = cv2.connectedComponentsWithStats(T , 8, cv2.CV_32S);

    ## Storing The Players Count And Position In Lists
    players = []
    for i in range(n):
            if stats[i][1] > 100 and stats[i][3] > 1.1 * stats[i][2] and stats[i][4] > 65 :
                players.append(i)
    players_pos = []
    for i in range(n):
            if stats[i][1] > 100 and stats[i][3] > 1.1 * stats[i][2] and stats[i][4] > 65 :
                players_pos.append([centroids[i][0],centroids[i][1]])

    players_bound = []
    for i in range(n):
        if stats[i][1] > 100 and stats[i][3] > 1.1 * stats[i][2] and stats[i][4] > 65:
            players_bound.append([stats[i][0], stats[i][1], stats[i][2], stats[i][3]])

    x_test = []
    y_test = []
    ## Identifying What Can Be Detected As A Player Using Stats And Drawing A Bounding Box Around Them

    for i in players:
        if not math.isnan(centroids[i][0]) and not math.isnan(centroids[i][0]):
            x = stats[i][0]
            y = stats[i][1]
            w = stats[i][0] + stats[i][2]
            h = stats[i][1] + stats[i][3]
            ## For Part 1
            #cv2.rectangle(Image, (x, y), (w, h), (0, 0, 255), 1)

    ### Data Collection :
            ### Cropping Images Segments Containing Players And Sending Them To Be Cropped And Saved

            #if gatherData:
            #    imageCropper(Image,x,y,w,h)

    ### Data Classification
            image = Image[y:h, x:w]
            temp = cv2.resize(image, (70, 70), interpolation=cv2.INTER_AREA)
            x_test.append(temp)





    ### Realtime Classification
    #y_test = realtimeClassifier(x_test)

    ### NN Classification
    x_test = np.array(x_test)
    y_test = imageClassifier(x_test)





    ## Defining Images That Only Have Circles On Players Locations (This Will Later Be Transformed Using Homography)
    locImageBlue = np.zeros(Image.shape, Image.dtype)
    locImageRed = np.zeros(Image.shape, Image.dtype)
    locImageWhite = np.zeros(Image.shape, Image.dtype)

    ## Define A BoundingBox Variable
    bboxes = []
    labels = []
    for i in range(len(players_pos)):
        player_pos = players_pos[i]
        player_bound = players_bound[i]
        if y_test[i] == 0:
            cv2.circle(locImageRed, (int(player_pos[0]), int(player_pos[1])), 5, (0, 0, 255), 5)
            bboxes.append([player_bound[0], player_bound[1], player_bound[2], player_bound[3]])
            labels.append(0)
        elif y_test[i] == 1:
            cv2.circle(locImageBlue, (int(player_pos[0]), int(player_pos[1])), 5, (255, 0, 0), 5)
            bboxes.append([player_bound[0], player_bound[1], player_bound[2], player_bound[3]])
            labels.append(1)
        elif y_test[i] == 2:
            cv2.circle(locImageWhite, (int(player_pos[0]), int(player_pos[1])), 5, (255, 255, 255), 5)
            bboxes.append([player_bound[0], player_bound[1], player_bound[2], player_bound[3]])
            labels.append(2)
        else:
            pass


    colored_images = []
    colored_images.append(locImageBlue)
    colored_images.append(locImageRed)
    colored_images.append(locImageWhite)

    return [bboxes,labels]


def computeHomography(Image):

    p1 = (639, 107)
    p2 = (867, 777)
    p3 = (143, 166)
    p4 = (1135, 115)



    points1 = np.array([p1, p2, p3, p4], dtype=np.float32)
    #for i in range(4):
    #    cv2.circle(I, (int(points1[i, 0]), int(points1[i, 1])), 5, [0, 0, 255], 2)
    #cv2.imshow('corners', I)
    #cv2.waitKey(0)


    p1 = (525, 0)
    p2 = (525, 700)
    p3 = (164, 159)
    p4 = (886, 159)

    points2 = np.array([p1, p2, p3, p4], dtype=np.float32)

    H = cv2.getPerspectiveTransform(points1, points2)

    return H


def detectFieldPoints(Image):

    G = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
    G = np.float32(G)
    window_size = 2
    soble_kernel_size = 3  # kernel size for gradients
    alpha = 0.04
    H = cv2.cornerHarris(G, window_size, soble_kernel_size, alpha)
    H = H / H.max()

    C = np.uint8(H > 0.01) * 255
    nC, CC, stats, centroids = cv2.connectedComponentsWithStats(C)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    corners = cv2.cornerSubPix(G, np.float32(centroids), (5, 5), (-1, -1), criteria)
    J = I.copy()
    for i in range(1, nC):
        cv2.circle(J, (int(corners[i, 0]), int(corners[i, 1])), 3, (0, 0, 255))
        print(corners[i])
        cv2.imshow('corners', J)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break


## STEP 1: READING THE VIDEO




parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str,  default=input_vid)
parser.add_argument('--algo', type=str, default='MOG2')
args = parser.parse_args()

rawMap = cv2.imread('2D_field.png')

if args.algo == 'MOG2':
    backSub = cv2.createBackgroundSubtractorKNN()

cap = cv2.VideoCapture(input_vid)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
ret, I = cap.read()

## Initialize Tracking By Defining A MultiTracker
multi_tracker = cv2.legacy.MultiTracker_create()

## Find 4 Corners in the Video In Order To Calculate Homography(This Was Done Manually)
#detectFieldPoints(I)

## Compute The Homography Using The 4 Point Correspondences
H = computeHomography(I)

## The Output Size Based On the Given 2D Map Size
n = 1280
m = 960
output_size = (n, m)

out = cv2.VideoWriter('Tracked.avi', fourcc, 30.0, (n, m))
frames = []


gatherData = False
p = 0
frame_num = 1
prev_frame = I
while True:
    #print("ASDA:",frame_num)
    ret, I = cap.read()

    ## The Code Below Is Written To Collect Data Every N frames (In Order To Prevent Collecting Redundant Data)
    if ret == False:
        break
    if p < 16 :
        gatherData = False
        p += 1
    elif  p == 16:
        gatherData = True
        p = 0
    dotsImage = []

    ## Calls Detection Every 10 frames


    if frame_num > 7:
        multi_tracker = cv2.legacy.MultiTracker_create()
        frame_num = 0
        box_label = calculateDifference(prev_frame)
        bboxes = box_label[0]
        #print(box_label)
        labels = box_label[1]

        if bboxes != [] or len(bboxes)>6:
            print("Box_added: ",len(bboxes))
            for bbox in bboxes:

                    # Add tracker to the multi-object tracker
                    multi_tracker.add(cv2.legacy.TrackerKCF_create(), prev_frame, bbox)

        else:
            frame_num = 17

    ret, bboxes = multi_tracker.update(I)
    #print(bboxes)
    for i, bbox in enumerate(bboxes):
        if labels[i] == 0:
            color = (0, 0, 255)
        elif labels[i] == 1:
            color = (255, 0, 0)
        else:
            color = (0, 255, 255)
        point_1 = (int(bbox[0]), int(bbox[1]))
        point_2 = (int(bbox[0] + bbox[2]),int(bbox[1] + bbox[3]))
        cv2.rectangle(I, point_1, point_2, color, 1)
        cv2.imshow('dots', I)
        cv2.waitKey(5)




        #for i in range(len(dotsImage)):
        #    dotsImage[i] = cv2.warpPerspective(dotsImage[i], H, output_size)

        #J =np.zeros(dotsImage[0].shape)

        #J += circleSpots(dotsImage[0], 'Blue')
        #J += circleSpots(dotsImage[1], 'Red')
        #J += circleSpots(dotsImage[2], 'Yellow')
        #cv2.imshow('dots', dotsImage)
        #cv2.waitKey(27)
        #K = rawMap.copy()
        #x,y,z = np.where(J > 0)
        #K[(x,y,z)] = J[(x,y,z)]


    frames.append(I)
    frame_num += 1
    prev_frame = I
for i in range(len(frames)):
    out.write(frames[i])

cap.release()
out.release()





def Good():
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)



    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2);

