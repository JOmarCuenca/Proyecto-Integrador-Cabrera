# imports
from .configs import config
from .configs.detection import detect_people, detect_faces
from scipy.spatial import distance as dist 
import numpy as np
from argparse import ArgumentParser
import imutils
import cv2
import os

__net           = None
__face_net      = None
__LABELS        = None
__FACE_LABELS   = None
__ln            = None
__face_ln       = None

class Args:

    def __init__(self, input, output, display) -> None:
        self.input      = input
        self.output     = output
        self.display    = display

class FrameArgs:

    def __init__(self, input : str, path : str) -> None:
        self.input  = input
        self.path   = path

class BoundingBox:

    def __init__(self,x1, y1, x2, y2) -> None:
        self.x1 = x1 
        self.y1 = y1
        self.x2 = x2 
        self.y2 = y2

    @staticmethod
    def fromT(t : tuple):
        return BoundingBox(t[0],t[1],t[2],t[3])

    def __str__(self):
        return f"[{self.x1}, {self.y1}, {self.x2}, {self.y2}]"

def getArgs() -> Args:
    # construct the argument parser and parse the arguments
    ap = ArgumentParser()
    ap.add_argument("-i", "--input", type=str, default="", help="path to (optional) input video file")
    ap.add_argument("-o", "--output", type=str, default="", help="path to (optional) output video file")
    ap.add_argument("-d", "--display", type=int, default=1, help="whether or not output frame should be displayed")

    temp = ap.parse_args()
    return Args(temp.input, temp.output, temp.display)

def setUpNN():
    global __net, __face_net, __LABELS, __FACE_LABELS, __ln, __face_ln
    # load the COCO class labels the YOLO model was trained on
    labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
    # labelsPath = os.path.sep.join([config.MODEL_PATH, "face_mask.names"])
    __LABELS = open(labelsPath).read().strip().split("\n")

    FacelabelsPath = os.path.sep.join([config.FACE_MODEL_PATH, "face_mask.names"])
    __FACE_LABELS = open(FacelabelsPath).read().strip().split("\n")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
    configPath  = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

    # load the YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    __net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)



    # Face Mask Yolo load from disk
    weightsPath = os.path.sep.join([config.FACE_MODEL_PATH, "face_mask.weights"])
    configPath  = os.path.sep.join([config.FACE_MODEL_PATH, "face_mask.cfg"])

    print("[INFO] loading YOLO Fine Tuned Mask from disk...")
    __face_net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

    

    # check if GPU is to be used or not
    if config.USE_GPU:
        # set CUDA s the preferable backend and target
        print("[INFO] setting preferable backend and target to CUDA...")
        __net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # print(f"cv2.dnn.DNN_BACKEND_CUDA {cv2.dnn.DNN_BACKEND_CUDA}")
        __net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # print(f"cv2.dnn.DNN_TARGET_CUDA {cv2.dnn.DNN_TARGET_CUDA}")
        __face_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # print(f"cv2.dnn.DNN_BACKEND_CUDA {cv2.dnn.DNN_BACKEND_CUDA}")
        __face_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # print(f"cv2.dnn.DNN_TARGET_CUDA {cv2.dnn.DNN_TARGET_CUDA}")

    # determine only the "output" layer names that we need from YOLO
    ln = __net.getLayerNames()
    __ln = [ln[i[0] - 1] for i in __net.getUnconnectedOutLayers()]

    # determine only the "output" layer names that we need from YOLO
    Faceln      = __face_net.getLayerNames()
    __face_ln   = [Faceln[i[0] - 1] for i in __face_net.getUnconnectedOutLayers()]

def get_iou(bb1 : BoundingBox, bb2 : BoundingBox):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1.x1 < bb1.x2
    assert bb1.y1 < bb1.y2
    assert bb2.x1 < bb2.x2
    assert bb2.y1 < bb2.y2

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1.x1, bb2.x1)
    y_top = max(bb1.y1, bb2.y1)
    x_right = min(bb1.x2, bb2.x2)
    y_bottom = min(bb1.y2, bb2.y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1.x2 - bb1.x1) * (bb1.y2 - bb1.y1)
    bb2_area = (bb2.x2 - bb2.x1) * (bb2.y2 - bb2.y1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def main(args : Args):
    # initialize the video stream and pointer to output video file
    print("[INFO] accessing video stream...")
    # open input video if available else webcam stream
    vs = cv2.VideoCapture(args.input if args.input else 0)
    writer = None

    # loop over the frames from the video stream
    while True:
        # read the next frame from the input video
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then that's the end fo the stream 
        if not grabbed:
            break

        # resize the frame and then detect people (only people) in it
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, __net, __ln, personIdx=__LABELS.index("person"))
        face_results = detect_faces(frame, __face_net, __face_ln, godIdx=__FACE_LABELS.index("Good"), badIdx=__FACE_LABELS.index("Bad"))

        # initialize the set of indexes that violate the minimum social distance
        violate = set()

        # ensure there are at least two people detections (required in order to compute the
        # the pairwise distance maps)
        if len(results) >= 2:
            # extract all centroids from the results and compute the Euclidean distances
            # between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i+1, D.shape[1]):
                    # check to see if the distance between any two centroid pairs is less
                    # than the configured number of pixels
                    if D[i, j] < config.MIN_DISTANCE:
                        # update the violation set with the indexes of the centroid pairs
                        violate.add(i)
                        violate.add(j)

        violating_BB = set()
        
        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract teh bounding box and centroid coordinates, then initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            # (cX, cY) = centroid
            color = (0, 255, 0)
            #result_text='Bajo riesgo'

            # if the index pair exists within the violation set, then update the color
            if i in violate:
                color = (0, 0, 255)
                #result_text='Alto riesgo'
                violating_BB.add(BoundingBox.fromT(bbox))

            # draw (1) a bounding box around the person and (2) the centroid coordinates of the person
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            # cv2.circle(frame, (cX, cY), 5, color, 1)

        # # draw the total number of social distancing violations on the output frame
        # text = "Social Distancing Violations: {}".format(len(violate))
        # cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        for (i, (prob, bbox, centroid)) in enumerate(face_results):
            # extract the bounding box and centroid coordinates, then initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            result_text='Bajo riesgo'

            #if the index pair exists within the violation set, then update the color
            if i == 1:
                bb = BoundingBox.fromT(bbox)
                ious = [get_iou(vBB,bb) for vBB in violating_BB]
                if(len(ious) > 0 and max(ious) > .002):
                    result_text='Alto riesgo'

            # draw (1) a bounding box around the person and (2) the centroid coordinates of the person 
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)       
            cv2.putText(frame, result_text, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

       # cv2.circle(frame, (cX, cY), 5, color, 1)

    # draw the total number of social distancing violations on the output frame
    #text = "Social Distancing Violations: {}".format(len(violate))
    #cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        # check to see if the output frame should be displayed to the screen
        if args.display > 0:
            # show the output frame
            cv2.imshow("Output", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the 'q' key is pressed, break from the loop
            if key == ord("q"):
                break
        
        # if an output video file path has been supplied and the video writer ahs not been 
        # initialized, do so now
        if args.output != "" and writer is None:
            # initialize the video writer
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(args.output, fourcc, 25, (frame.shape[1], frame.shape[0]), True)

        # if the video writer is not None, write the frame to the output video file
        if writer is not None:
            # print("[INFO] writing stream to output")
            writer.write(frame)

def predictFrames(args : FrameArgs):
    # initialize the video stream and pointer to output video file
    print("[INFO] accessing video stream...")
    # open input video if available else webcam stream
    vs = cv2.VideoCapture(args.input if args.input else 0)
    writer = None

    count = 0

    # loop over the frames from the video stream
    while True:
        # read the next frame from the input video
        (grabbed, frame) = vs.read()

        # if the frame was not grabbed, then that's the end fo the stream 
        if not grabbed:
            break

        # resize the frame and then detect people (only people) in it
        frame = imutils.resize(frame, width=700)
        results = detect_people(frame, __net, __ln, personIdx=__LABELS.index("person"))
        face_results = detect_faces(frame, __face_net, __face_ln, godIdx=__FACE_LABELS.index("Good"), badIdx=__FACE_LABELS.index("Bad"))

        # initialize the set of indexes that violate the minimum social distance
        violate = set()

        # ensure there are at least two people detections (required in order to compute the
        # the pairwise distance maps)
        if len(results) >= 2:
            # extract all centroids from the results and compute the Euclidean distances
            # between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            D = dist.cdist(centroids, centroids, metric="euclidean")

            # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i+1, D.shape[1]):
                    # check to see if the distance between any two centroid pairs is less
                    # than the configured number of pixels
                    if D[i, j] < config.MIN_DISTANCE:
                        # update the violation set with the indexes of the centroid pairs
                        violate.add(i)
                        violate.add(j)

        violating_BB = set()
        
        # loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract teh bounding box and centroid coordinates, then initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            # (cX, cY) = centroid
            color = (0, 255, 0)
            #result_text='Bajo riesgo'

            # if the index pair exists within the violation set, then update the color
            if i in violate:
                color = (0, 0, 255)
                #result_text='Alto riesgo'
                violating_BB.add(BoundingBox.fromT(bbox))

            # draw (1) a bounding box around the person and (2) the centroid coordinates of the person
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            # cv2.circle(frame, (cX, cY), 5, color, 1)

        # # draw the total number of social distancing violations on the output frame
        # text = "Social Distancing Violations: {}".format(len(violate))
        # cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        for (i, (prob, bbox, centroid)) in enumerate(face_results):
            # extract the bounding box and centroid coordinates, then initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            result_text='Bajo riesgo'

            #if the index pair exists within the violation set, then update the color
            if i == 1:
                bb = BoundingBox.fromT(bbox)
                ious = [get_iou(vBB,bb) for vBB in violating_BB]
                if(len(ious) > 0 and max(ious) > .002):
                    result_text='Alto riesgo'

            # draw (1) a bounding box around the person and (2) the centroid coordinates of the person 
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)       
            cv2.putText(frame, result_text, (startX, startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

       # cv2.circle(frame, (cX, cY), 5, color, 1)

    # draw the total number of social distancing violations on the output frame
    #text = "Social Distancing Violations: {}".format(len(violate))
    #cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        cv2.imwrite(f"{args.path}frame{count}.jpg",frame)
        count += 1

setUpNN()

if __name__ == "__main__":
    main(getArgs())
    # predictFrames(FrameArgs("../../uploadedAssets/IMG_9841.mp4",
    #     "frames/"
    # ), targetLabel="Good")