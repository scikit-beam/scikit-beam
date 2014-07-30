from pylab import *
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
    # print istack
    #istack = 100*np.ones((settingAngles.shape[0], detPixSizeX, detPixSizeY))

    totSet = recip.process_to_q(settingAngles, detSizeX, detSizeY, detPixSizeX, detPixSizeY, detX0, detY0, detDis, waveLen, UBmat, istack)
    
    print " \n\n HKL Values "
    print totSet
    return totSet


"""def test_process_grid():
    size = 10
    data = test_process_to_q()
    Qmax = array([1.0, 1.0, 1.0])
    Qmin = array([-1.0, -1.0, -1.0])
    dQN = array([size,size, size])
    grid = np.mgrid[0:dQN[0], 0:dQN[1], 0:dQN[2]]
    r = (Qmax - Qmin) / dQN
    
    X = grid[0] * r[0] + Qmin[0]
    Y = grid[1] * r[1] + Qmin[1]
    Z = grid[2] * r[2] + Qmin[2]
    
    # data = np.exp(-(X**2 + Y**2 + Z**2) / (2 * sigma**2))
    #data = 1000*np.random.rand(size*size*size)
    #data  = 100*np.ones(size*size*size)
    
    out = array([np.ravel(X),
                 np.ravel(Y),
                 np.ravel(Z),
                 np.ravel(data)])
    print " out"
    print out
    print out.shape
                 
    #gridData, gridOccu, gridStd, gridOut, emptNb, gridbins = recip.process_grid(out, Qmax, Qmin, dQN)
    gridData, gridOccu, gridStd, gridOut = ctrans.grid3d(out, Qmin, Qmax, dQN, norm=1)
                 # print "gridout = ", gridData,
                 #print emptNb
    return gridData, gridOccu, gridOut"""


