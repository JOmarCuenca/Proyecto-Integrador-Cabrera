import cv2
from os import makedirs
from shutil import rmtree
from .social_distancing_detector.social_distancing_detector import FrameArgs, predictFrames

__DEMO_PATH__   = "./demoFrames/"
__DEMO_M_PATH__ = "./demoFramesModified/"

def cleanAssets():
  rmtree(__DEMO_PATH__, True)
  rmtree(__DEMO_M_PATH__, True)
  makedirs(__DEMO_PATH__, exist_ok=True)
  makedirs(__DEMO_M_PATH__, exist_ok=True)

def breakVideoIntoFrames(path_to_video : str):
  vidcap = cv2.VideoCapture(path_to_video)
  success,image = vidcap.read()
  count = 0
  while success:
    modifiedImg = cv2.flip(image, 0)
    # cv2.imwrite(__DEMO_PATH__+"frame%d.jpg" % count, image)
    cv2.imwrite(__DEMO_M_PATH__+"frame%d.jpg" % count, modifiedImg)
    success,image = vidcap.read()
    count += 1
  return count

def analyzeStream(path_to_video: str):
  print(" Begin analyzing ".capitalize().center(30,"="))
  cleanAssets()
  predictFrames(FrameArgs(path_to_video, __DEMO_M_PATH__))

def cleanAndBreak(path_to_video : str):
  cleanAssets()
  return breakVideoIntoFrames(path_to_video)

if __name__ == "__main__":
  cleanAssets()
  breakVideoIntoFrames('sampleVideo.mp4')