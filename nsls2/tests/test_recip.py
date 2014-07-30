import numpy as np
import nsls2.recip as recip

 
def test_process_to_q():

    detPixSizeX, detPixSizeY, detSizeX, detSizeY, detX0, detY0, detDis, detAng = 0.0135*8, 0.0135*8, 256, 256, 256/2.0, 256/2.0, 355.0, 0.0

    energy = 640  # ( ev)
    # HC_OVER_E to convert from E to Lambda
    hc_over_e = 12398.4
    waveLen = hc_over_e/ energy # (Angstrom )
     
    UBmat= np.matrix([[-0.01231028454, 0.7405370482 , 0.06323870032],
                       [ 0.4450897473 , 0.04166852402,-0.9509449389 ],
                       [-0.7449130975 , 0.01265920962,-0.5692399963 ]])
     
    settingAngles = np.matrix([[40., 15., 30., 25., 10., 5.],
                              [90., 60., 0., 30., 10., 5.]])
    # delta=40, theta=15, chi = 90, phi = 30, mu = 10.0, gamma=5.0
    
    istack = 100*np.ones((settingAngles.shape[0], 256, 256))

    totSet = recip.process_to_q(settingAngles, detSizeX, detSizeY, detPixSizeX, detPixSizeY, detX0, detY0, detDis, waveLen, UBmat, istack)
    
    print totSet print "\n\n Six Angles "
    print settingAngles
    print "\n\n Known HKL values  "
    print " HKL = [[-0.15471196  0.19673939 -0.11440936]]"
    print " HKL = [[ 0.10205953  0.45624416 -0.27200778]]"
    
    print " \n\n HKL Values "
    print totSet
   

def test_process_grid():
    size = 5
    Qmax = array([1.0, 1.0, 1.0])
    Qmin = array([-1.0, -1.0, -1.0])
    dQN = array([size,size, size])
    sigma = 0.1
    grid = np.mgrid[0:dQN[0], 0:dQN[1], 0:dQN[2]]
    r = (Qmax - Qmin) / dQN

    X = grid[0] * r[0] + Qmin[0]
    Y = grid[1] * r[1] + Qmin[1]
    Z = grid[2] * r[2] + Qmin[2]

    out = np.zeros((size,size,size))
    
    out = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2))

    out = array([np.ravel(X),
                 np.ravel(Y),
                 np.ravel(Z),
                 np.ravel(out)])
    data = out.T

    gridData, gridOccu, gridStd, gridOut, emptNb, gridbins = recip.process_grid(data,
                                                array([-1.0, -1.0, -1.0]),
                                                array([1.0, 1.0, 1.0]),
                                                array([size, size, size]))
    print "\n\n gridData "
    print gridData
    
