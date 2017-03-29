"""Stores a dataset into numpy ndarray format"""
import numpy as np
import cv2
import warnings
import glob
from math import floor


# --- Flags/Declarations and I/O ---
train = False
cap = cv2.VideoCapture()
equallyDistributed = True #do you want to have equally distributed step sizes between images? -> True
allFiles = []
numFrames = 20 #number of frames to capture
numChannels = 3 #depth of images
if numChannels == 1: warnings.warn("using gray scale images")
frameSize = 128
numberOfVideos = 3
fileType = "*.avi"


withDescriptionFiles = False #is there a file with video file names? -> True
filename = 'blobs-equally-3_videos_20_frames_1_channel_128x128.npy'

outputPath = '../data/blobs/'
inputPath = '../data/blobs/'

#if 'withDescriptionFiles = True, define:
descriptionFiles = np.array(['/common/homes/students/ferreira/Downloads/ucfTrainTestlist/testlist01.txt'])


def getVideoCapture(path):
    cap = None
    if path:
        cap = cv2.VideoCapture(path)
        #set capture settings here:
        cap.set(0, 0)  # (0,x) POS_MSEC, (1,x)
    return cap;

def getNextFrame(cap):
    ret, frame = cap.read()
    if ret == False:
        return None

   # if numChannels == 1:
            #convert to gray image
    #        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return np.asarray(frame)

if withDescriptionFiles:
# --- Parse description files ---
    for i in np.nditer(descriptionFiles):
        f = open(i, 'r').readlines()
        N = len(f) - 1
        for i in range(0, N):
            w = f[i].split()
            allFiles.append(w[0])
else:
    allFiles = glob.glob(inputPath + "/" + fileType)
    #allFiles = dircache.listdir(inputPath)



print(str(len(allFiles))+' videos available')

assert len(allFiles) >= numberOfVideos
#without cropping
#data = np.zeros((numberOfVideos, 20, 3, 240, 320))
#image = np.zeros((3,240,320))
#video = np.zeros((20,3,240,320))
#data = np.zeros((numberOfVideos, 3, 240, 320)) #DataHandler.Crop receives only 4 of 5 dimensions

#let's do the resizing here:

image = np.zeros((frameSize, frameSize, numChannels), dtype=np.uint8)
video = np.zeros((numFrames, frameSize, frameSize, numChannels), dtype=np.uint16)
data = np.zeros((numberOfVideos, numFrames, frameSize, frameSize, numChannels), dtype=np.uint16)

# --- Sample the frames from each video ---
if not equallyDistributed:
    for i in range(numberOfVideos):
        print(i)
        cap = getVideoCapture(inputPath + allFiles[i])

        for j in range(numFrames):
            ret, frame = cap.read()

            for k in range(numChannels):
                resizedImage = cv2.resize(frame[:,:,k], (frameSize, frameSize))
                data[i,j,:,:,k] = resizedImage


#algorithm chooses frame step size automatically for a equal separation distribution
else:
    for i in range(numberOfVideos):
        cap = getVideoCapture(allFiles[i])

        #compute meta data of video
        frameCount = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)

        # returns nan, if fps needed a measurement must be implemented
        #frameRate = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        steps = frameCount / numFrames
        j = 0

        restart = True
        while restart:
            for f in range(int(frameCount)):
                # get next frame after 'steps' iterations:
                # floor used after modulo operation because rounding module before leads to
                # unhandy partition of data (big gab in the end)
                if floor(f % steps) == 0:
                    frame = getNextFrame(cap)
                    #special case handling: opencv's frame count != real frame count, reiterate over same video
                    if frame is None and j < numFrames:
                        warnings.warn("reducing step size due to error")
                        #repeat with smaller step size
                        steps -= 1
                        j = 0
                        cap.release()
                        cap = getVideoCapture(allFiles[i])
                        video.fill(0)
                        break
                    else:
                        if j >= numFrames:
                            restart = False
                            break

                        # iterate over channels
                        for k in range(numChannels):
                            #cv returns 2 dim array if gray
                            if frame.ndim == 2:
                                resizedImage = cv2.resize(frame[:,:], (frameSize, frameSize))
                            else:
                                resizedImage = cv2.resize(frame[:, :, k], (frameSize, frameSize))
                            image[:,:,k] = resizedImage

                        video[j, :, :, :] = image
                        #image counter
                        j += 1
                        print('total frames: ' + str(j) + " frame in video: " + str(f))
                else:
                    getNextFrame(cap)

        print (str(i+1) + " of " + str(numberOfVideos) + " videos processed")
        data[i,:,:,:,:] = video
        cap.release()

    #non-equally distributed solution
    if train:
        np.save(outputPath + filename, data)
    else:
        np.save(outputPath + filename, data)

