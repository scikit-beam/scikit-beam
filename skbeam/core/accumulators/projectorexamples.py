import matplotlib.pyplot as plt

image = np.array((900, 1024))

params = [[self.image.shape[0],self.image.shape[1],False],
          [self.image.shape[1],self.image.shape[0],True]]:
for xsize, ysize, cartesian in params:
    radproj = RadialProjector(xsize, ysize, xc=0, yc=0, rmin=100,
                              rmax=900, nbins=100, phimin=5, phimax=60,
                              norm=False, cartesian=cartesian)
    projection = radproj(self.image)
    if cartesian:
        plt.title('Cartesian Axes')
        plt.imshow(radproj.weights,origin='lower')
    else:
        plt.title('Matrix Axes')
        plt.imshow(radproj.weights)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
