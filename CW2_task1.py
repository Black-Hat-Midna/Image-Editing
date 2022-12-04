# -*- coding: utf-8 -*-
"""

Notes:
    based on Poisson Image Editing by Patrick Perez, Michel Gangnet, Andrew Blake
    
    As the size of the size of the selected region increases the computation time
    required increases exponentially.

"""

from skimage import draw as d
from skimage import io
import numpy as np
import cv2
import matplotlib.pyplot as plt

points = []

def createPolygon(img):
    """
    adapted from
    https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
    
    Gets user input via a ui where the user and click on an image and 
    outputs the coordinates of the clicked on points as a list.
    """
    
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', getPoint)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return points

def getPolygon(img, points):
    """
    gets a polygon mask, image and the polygons perimeter from a set of points 
    and an image.
    """

    mask = d.polygon2mask(img.shape, points)
    
    points = np.array(points)
    x,y = d.polygon_perimeter(points[..., 0], points[..., 1], shape=img.shape, clip=True)
    perimeter = np.zeros(img.shape)
    perimeter[x, y] = 1
    
    mask = ((mask) | (perimeter).astype(dtype=bool))
    Oimg = (img*mask)
    
    return mask, Oimg, perimeter

def getSumFQ(polygon, originalImg, mask):
    """
    produces the values for B based on the sum of f star where the neighbouring
    points intersect with alpha omega
    """
    
    flatM = (mask-polygon).flatten()
    ind = np.where(flatM == 1)
    mp = mask-polygon
    imgConved = np.zeros(mask.shape)
    count=0
    for y in range(0, mask.shape[1]):
        for x in range(0, mask.shape[0]):
            if(mp[x, y] == 1):
                count+=1
                index = np.array([x, y])+np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
                if((polygon[index[..., 0], index[..., 1]] == 1).any()):
                    
                    index = index[np.where(polygon[index[..., 0], index[..., 1]] == 1)[0]]#get only outter edge
                    
                    imgConved[x, y] = np.sum(originalImg[index[..., 0], index[..., 1]])
                    
    imgConved = imgConved.flatten()[ind]

    return imgConved
    

def getParams(mask):
    """
    gets the values for A in the regression model given the image and the mask
    """

    N = np.count_nonzero(mask)
    ret = np.eye(N)
    ret = np.where(ret==1, 4, 0)
    #s1, s2 = img.shape
    flatM = mask.flatten()
    #flatI = img.flatten()
    ind = np.where(flatM == 1)
    
    count = 0
    for i in range(0, flatM.shape[0]):
        if(flatM[i]==1):
            pixelI = np.array(np.unravel_index(i, shape=mask.shape))
            
            offset = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])
            pixelI = pixelI[np.newaxis, :]+offset
            pixelI = np.array([pixelI[:, 0], pixelI[:, 1]])
            NpInd = np.ravel_multi_index(pixelI, mask.shape)
            NpInd = NpInd[np.where(flatM[NpInd]==1)]
            
            #if(len(np.where(np.isin(ind, NpInd)==True)[1])==0):
                #print("problem here")#problems occur if the mask has overlaps so you know, don't!!!
                
            ret[count, np.where(np.isin(ind, NpInd))[1]] = -1
            count+=1

    return ret

def solveLinearEqs(grads, params):
    """
    solves the sparse linear regression problem given B and A values 
    respectively
    """
    
    A = params
    B = grads
    print("solving...")
    X = np.linalg.solve(A, B)
    print("solved...")
    
    return X

def makeNewImage(mask, originalImg, solvedIntensities):
    """
    creates a new image given the mask, original image and the new
    intensities
    """
    
    flatM = mask.flatten()
    flatM = np.where(flatM==1)
    ind = np.unravel_index(flatM, shape=originalImg.shape)
    originalImg[ind] = solvedIntensities        
    
    return originalImg
    
def getPoint(event, x, y, flags, params):
    """
    the method below was adapted from 
    https://www.geeksforgeeks.org/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
    
    The method handles user input on the cv2 ui
    """
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([y, x])


if __name__ == "__main__":
    print("running...")
    
    #open and resize images
    img1 = cv2.imread('woodtable.jpg')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)/255.
    
    img1 = cv2.resize(img1, (250, 250), interpolation = cv2.INTER_AREA)
    
    #img2 = cv2.imread('grapes.jpg')
    #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)/255.
    
    #img2 = cv2.resize(img2, (250, 250), interpolation = cv2.INTER_AREA)
    
    #get user input poygon making sure its not too big for computers memory
    p = createPolygon(img1)
    Fm, fstar, per1 = getPolygon(img1, p)

    #get components for linear regression model and run model
    B = getSumFQ(per1, img1, Fm)
    print("getting params...")
    para = getParams(Fm-per1)
    print("params got...")
    X = solveLinearEqs(B, para)
    X = np.clip(X, 0, 1)
    output = makeNewImage(Fm-per1, img1, X)

    plt.imshow(output, cmap="gray")
    plt.show()
    io.imsave("outputT1.png", (output))
    
    
    
    
    
    