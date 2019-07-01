import scipy.io
import stochastic_gradient_descent
import numpy as np
from matplotlib import pyplot


if __name__ == "__main__":
    GMMData = scipy.io.loadmat('GMMData.mat')
    PeaksData = scipy.io.loadmat('PeaksData.mat')
    SwissRollData = scipy.io.loadmat('SwissRollData.mat')

    SwissCt = SwissRollData['Ct']
    SwissYt = SwissRollData['Yt']
    print(" swissCt.shape: " + str(SwissCt.shape))
    print(" swissYt.shape: " + str(SwissYt.shape))

    PeaksYt = PeaksData['Yt']
    PeaksCt = PeaksData['Ct']

    print(" peeksYt.shape: " + str(PeaksYt.shape))
    print(" peeksCt.shape: " + str(PeaksCt.shape))

    GMMYt = GMMData['Ct']
    GMMCt = GMMData['Yt']

    print(" gmmYt.shape: " + str(GMMYt.shape))
    print(" gmmCt.shape: " + str(GMMCt.shape))
    print(type(GMMData['Cv']))
    print(type(GMMData['Ct']))
    print(type(GMMData['Yv']))
    print(type(GMMData['Yt']))

    stochastic_gradient_descent.stochastic_gradient_descent(W=[0, 0, 0, 0], )


