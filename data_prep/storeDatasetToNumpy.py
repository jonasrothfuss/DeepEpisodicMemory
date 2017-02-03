"""Displays reconstruction and future predictions for trained models."""
import numpy as np
import cv2
from data_handler import *
from math import floor


# --- Declarations and I/O ---
cap = cv2.VideoCapture()
equallyDistributed = True
allFiles = []
numFrames = 20 # number of frames to capture
frameSize = 64
numberOfVideos = 3000
pathToDataDump = '/common/homes/students/ferreira/Desktop/localhome/ferreira/unsupervised-videos-master/datasets/'
pathToDataset = '/common/homes/students/ferreira/Downloads/UCF-101/'
filenames = np.array(['/common/homes/students/ferreira/Downloads/ucfTrainTestlist/testlist01.txt'])
#filenames = np.array(['/common/homes/students/ferreira/Downloads/ucfTrainTestlist/trainlist01.txt'])#,
                #'/common/homes/students/ferreira/Downloads/ucfTrainTestlist/trainlist02.txt',
                #'/common/homes/students/ferreira/Downloads/ucfTrainTestlist/trainlist03.txt'])


def getVideoCapture(path):
    cap = None
    if path:
        cap = cv2.VideoCapture(path)
        # set capture settings here:
        cap.set(0, 0)  # (0,x) POS_MSEC, (1,x)

    return cap;

def getNextFrame(cap):
    ret, frame = cap.read()
    if ret == True:
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #convert to RGB (standard is bgr in cv)
        return frame
    else:
        return None


# --- Parse description files ---
for i in np.nditer(filenames):
    f = open(i, 'r').readlines()
    N = len(f) - 1
    for i in range(0, N):
        w = f[i].split()
        allFiles.append(w[0])

print(str(len(allFiles))+' videos available')

assert len(allFiles) >= numberOfVideos
#without cropping
#data = np.zeros((numberOfVideos, 20, 3, 240, 320))
#image = np.zeros((3,240,320))
#video = np.zeros((20,3,240,320))
#data = np.zeros((numberOfVideos, 3, 240, 320)) #DataHandler.Crop receives only 4 of 5 dimensions

#let's do the resizing here:
image = np.zeros((3,frameSize,frameSize))
video = np.zeros((numFrames,3,frameSize,frameSize))
data = np.zeros((numberOfVideos, numFrames, 3, frameSize, frameSize))

# --- Sample the frames from each video ---

# non-equally distributed solution
if equallyDistributed == False:
    for i in range(numberOfVideos):
        print(i)
        cap = getVideoCapture(pathToDataset + allFiles[i])

        for j in range(numFrames):
            ret, frame = cap.read()


            for k in range(3): #3 depth channels
                resizedImage = cv2.resize(frame[:,:,k], (frameSize, frameSize))
                data[i,j,k,:,:] = resizedImage
                #image[k,:,:] = resizedImage
                #video[j,:,:,:] = image;

            #for l in range(1):
            #    cap.grab() #jump x frames ahead, to be improved as too short videos in dataset exist

        #video = x.Crop(video);
        #data[i,:,:,:,:] = video

# equally distributed solution
else:
    for i in range(numberOfVideos):
        print str(i) + " of " + str(numberOfVideos) + " videos processed"
        cap = getVideoCapture(pathToDataset + allFiles[i])

        # compute meta data of video
        frameCount = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        # returns nan, if fps needed a measurement must be implemented
        #frameRate = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        steps = frameCount / numFrames
        j = 0

        #print "taking every " + str(steps) + "th frame of " + str(int(frameCount)) + " overall frames"
        restart = True
        while restart:
            for f in range(int(frameCount)):
                # get next frame after 'steps' iterations:
                # floor used after modulo operation because rounding module before leads to
                # unhandy partition of data (big gab in the end)
                if floor(f % steps) == 0:
                    frame = getNextFrame(cap)
                    # special case: opencv's frame count != real frame count, reiterate over same video
                    if frame == None and j < numFrames:
                        print "reducing step size due to error"
                        steps -= 1 # repeat it with smaller step size
                        j = 0
                        cap.release()
                        cap = getVideoCapture(pathToDataset + allFiles[i])
                        video.fill(0)
                        break
                    else:
                        if j >= numFrames:
                            restart = False
                            break
                        # 3 depth channels
                        for k in range(3):
                            resizedImage = cv2.resize(frame[:,:,k], (frameSize, frameSize))
                            image[k,:,:] = resizedImage

                        video[j, :, :, :] = image
                        j += 1 # image counter
                        #print str(j) + " " + str(f)
                else:
                    getNextFrame(cap)

        data[i,:,:,:,:] = video
        cap.release()


#np.save(pathToDataDump+'UCF-101Videos1-9000Frames1-20_64x64.npy', data);
np.save(pathToDataDump+'UCF-101Videos1-3000Frames1-20_valid_64x64.npy', data);
