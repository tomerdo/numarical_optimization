import scipy.io
import numpy as np
from matplotlib import pyplot





if __name__ == "__main__":
    GMMData = scipy.io.loadmat('GMMData.mat')
    PeaksData = scipy.io.loadmat('PeaksData.mat')
    SwissRollData = scipy.io.loadmat('SwissRollData.mat')

    SwissCt = SwissRollData['Ct']

    print(SwissCt.shape)

    PeaksCt = PeaksData['Yt']

    print(PeaksCt.shape)

    print(PeaksCt[0].shape)

    print(type(GMMData['Cv']))
    print(type(GMMData['Ct']))
    print(type(GMMData['Yv']))
    print(type(GMMData['Yt']))


