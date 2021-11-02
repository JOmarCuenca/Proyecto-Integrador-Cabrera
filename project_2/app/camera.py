import time
from base_camera import BaseCamera
from os import listdir

__DEMO_PATH__ = "demoFrames"
__DEMO_M_PATH__ = "demoFramesModified"

PLACEHOLDER = open("./uploadedAssets/placeHolder.png", "rb").read()

class StreamedCamera(BaseCamera):

    def __init__(self):
        super().__init__()

    def currentImage(path : str, index : int):
        f = PLACEHOLDER
        nextIndex = index + 1
        try:
            f = open(f"{path}/frame{index}.jpg", 'rb').read()
        except:
            print("Frame not ready")
            nextIndex = index
        return f, nextIndex


    

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

class ModifiedCamera(StreamedCamera):

    index       = 0
    maxLength   = 0

    def __init__(self, maxLength : int):
        super().__init__()
        Camera.maxLength        = maxLength
        ModifiedCamera.index    = 0

    @staticmethod
    def frames():
        while True:
            time.sleep(1/60)
            if(not (ModifiedCamera.index < ModifiedCamera.maxLength)):
                ModifiedCamera.index = 0
            
            f, ModifiedCamera.index = super().currentImage(__DEMO_M_PATH__, ModifiedCamera.index)
            yield f
