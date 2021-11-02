import time
from base_camera import BaseCamera
from os import listdir

__DEMO_PATH__ = "demoFrames"
__DEMO_M_PATH__ = "demoFramesModified"

PLACEHOLDER = open("./uploadedAssets/placeHolder.png")

class Camera(BaseCamera):

    imgs    = []
    index   = 0

    def __init__(self):
        super().__init__()
        ModifiedCamera.index = 0
        Camera.imgs = [open(f"{__DEMO_PATH__}/frame{f}.jpg", 'rb').read() for f in range(len(listdir(__DEMO_PATH__)))]


    @staticmethod
    def frames():
        while True:
            time.sleep(1/60)
            if(len(Camera.imgs) != 0):
                yield Camera.imgs[Camera.index]
                Camera.index += 1
                if(Camera.index >= len(Camera.imgs)):
                    Camera.index = 0
            else:
                yield PLACEHOLDER

class ModifiedCamera(BaseCamera):

    imgs    = []
    index   = 0

    def __init__(self):
        super().__init__()
        ModifiedCamera.index = 0
        ModifiedCamera.imgs = [open(f"{__DEMO_M_PATH__}/frame{f}.jpg", 'rb').read() for f in range(len(listdir(__DEMO_M_PATH__)))]

    @staticmethod
    def frames():
        while True:
            time.sleep(1/60)
            if(len(ModifiedCamera.imgs) != 0):
                yield ModifiedCamera.imgs[ModifiedCamera.index]
                ModifiedCamera.index += 1
                if(ModifiedCamera.index >= len(ModifiedCamera.imgs)):
                    ModifiedCamera.index = 0
            else:
                yield PLACEHOLDER
