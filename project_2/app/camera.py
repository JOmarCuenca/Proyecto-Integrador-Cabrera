import time
from base_camera import BaseCamera
from os import listdir

__DEMO_PATH__ = "demoFrames"
__DEMO_M_PATH__ = "demoFramesModified"

PLACEHOLDER = open("./uploadedAssets/placeHolder.png","rb").read()

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

    index   = -1
    lastSuccess = None

    def __init__(self):
        super().__init__()
        ModifiedCamera.index = -1
        # ModifiedCamera.imgs = [open(f"{__DEMO_M_PATH__}/frame{f}.jpg", 'rb').read() for f in range(len(listdir(__DEMO_M_PATH__)))]

    @staticmethod
    def get_current_frame(index : int):
        if(index == -1):
            ModifiedCamera.lastSuccess = None
            return PLACEHOLDER
        else:
            try:
                # print(f"Currently trying to print index {index}")
                f = open(f"{__DEMO_M_PATH__}/frame{index}.jpg", 'rb').read()
                if(index > ModifiedCamera.index):
                    ModifiedCamera.lastSuccess = time.time()
                ModifiedCamera.index = index
                return f
            except FileNotFoundError:
                # print(f"Error on index {ModifiedCamera.index}")
                notNone = (not ModifiedCamera.lastSuccess is None)
                # print(f"Last Success is not None {notNone}")
                lapsedTime = 0
                if(notNone):
                    lapsedTime = (time.time() - ModifiedCamera.lastSuccess)
                    # print(f"Lapsed Time {lapsedTime}")
                if(notNone and lapsedTime > 2):
                    ModifiedCamera.index = -1
                    ModifiedCamera.lastSuccess = None
                    return ModifiedCamera.get_current_frame(0)
                else:
                    return ModifiedCamera.get_current_frame(index - 1)


    @staticmethod
    def frames():
        while True:
            time.sleep(1/30)
            yield ModifiedCamera.get_current_frame(ModifiedCamera.index + 1)
