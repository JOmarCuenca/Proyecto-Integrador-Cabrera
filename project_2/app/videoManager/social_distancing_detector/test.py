from social_distancing_detector import FrameArgs, predictFrames

if __name__ == "__main__":
    predictFrames(FrameArgs("pedestrians.mp4","tempFrames/"))