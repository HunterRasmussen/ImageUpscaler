from imageio import imread, imwrite
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
from skimage.transform import rescale
from pathlib import Path
import sys

def downscaleByHalf(img, antiAliasing=False):
    toReturn = rescale(img, (0.5, 0.5, 1), anti_aliasing=antiAliasing)
    return toReturn


def padEdge(img,amount):
    padded = np.pad(img,((amount,amount),(amount,amount),(0,0)),mode='edge')
    return padded


def getNeighborhoods(image, neighborhoodWidth):
    neighborhoods = []
    neighborhoodWidth = abs(neighborhoodWidth)
    if (neighborhoodWidth%2 == 0):
        neighborhoodWidth = neighborhoodWidth - 1
    offset = math.floor(neighborhoodWidth/2)
    paddedImage = padEdge(image,offset)
    #for each pixel
    for row in range(offset,paddedImage.shape[0]-offset):
        for column in range(offset, paddedImage.shape[1]-offset):
            pixelToAdd = []
            #neighborhood loops
            for i in range(-offset,offset+1):
                for j in range(-offset,offset+1):
                    for k in range(0,3): #color
                        pixelToAdd.append(paddedImage[row+i,column+j,k])
            neighborhoods.append(pixelToAdd)
    return neighborhoods


def breakupImage(img):
    im1 = img[0::2,0::2,:] #start 1st row, 1st column
    im2 = img[0::2,1::2,:] #start 1st row, 2nd column
    im3 = img[1::2,0::2,:] #start 2nd row, 1st column
    im4 = img[1::2,1::2,:] #start 2nd row, 2nd column
    return im1,im2,im3,im4

def calcRegression(images):
    neighborhoods = []

    images0r = []
    images0g = []
    images0b = []
    images1r = []
    images1g = []
    images1b = []
    images2r = []
    images2g = []
    images2b = []
    images3r = []
    images3g = []
    images3b = []
    
    for image in images:
        #convert image to 0-1 values as imread opens them as values 0-255
        image = image/255
        downscaled = downscaleByHalf(image)
        if not neighborhoods:
            neighborhoods = getNeighborhoods(downscaled,5)
        else:
            np.concatenate((neighborhoods,getNeighborhoods(downscaled,5)))
        im0,im1,im2,im3 = breakupImage(image)
        images0r.append(im0[:,:,0])
        images0g.append(im0[:,:,1])
        images0b.append(im0[:,:,2])
        images1r.append(im1[:,:,0])
        images1g.append(im1[:,:,1])
        images1b.append(im1[:,:,2])
        images2r.append(im2[:,:,0])
        images2g.append(im2[:,:,1])
        images2b.append(im2[:,:,2])
        images3r.append(im3[:,:,0])
        images3g.append(im3[:,:,1])
        images3b.append(im3[:,:,2])

    neighborhoods = np.insert(neighborhoods,0,1,axis =1)
        
    # b vectors (original size images) reshaped to be vectors
    r0 = np.vstack(np.reshape(images0r,-1))
    g0 = np.vstack(np.reshape(images0g,-1))
    b0 = np.vstack(np.reshape(images0b,-1))
    r1 = np.vstack(np.reshape(images1r,-1))
    g1 = np.vstack(np.reshape(images1g,-1))
    b1 = np.vstack(np.reshape(images1b,-1))
    r2 = np.vstack(np.reshape(images2r,-1))
    g2 = np.vstack(np.reshape(images2g,-1))
    b2 = np.vstack(np.reshape(images2b,-1))
    r3 = np.vstack(np.reshape(images3r,-1))
    g3 = np.vstack(np.reshape(images3g,-1))
    b3 = np.vstack(np.reshape(images3b,-1))


    #regressions
    result0r = np.linalg.lstsq(neighborhoods, r0, rcond=None)[0]
    print("Completed 1/12 regressions")
    result0g = np.linalg.lstsq(neighborhoods, g0, rcond=None)[0]
    result0b = np.linalg.lstsq(neighborhoods, b0, rcond=None)[0]
    print("Completed 3/12 regressions")
    result1r = np.linalg.lstsq(neighborhoods, r1, rcond=None)[0]
    result1g = np.linalg.lstsq(neighborhoods, g1, rcond=None)[0]
    result1b = np.linalg.lstsq(neighborhoods, b1, rcond=None)[0]
    print("Completed 6/12 regressions")
    result2r = np.linalg.lstsq(neighborhoods, r2, rcond=None)[0]
    result2g = np.linalg.lstsq(neighborhoods, g2, rcond=None)[0]
    result2b = np.linalg.lstsq(neighborhoods, b2, rcond=None)[0]
    result3r = np.linalg.lstsq(neighborhoods, r3, rcond=None)[0]
    result3g = np.linalg.lstsq(neighborhoods, g3, rcond=None)[0]
    result3b = np.linalg.lstsq(neighborhoods, b3, rcond=None)[0]
    print("Completed 12/12 regressions")
    
    results = [result0r, result0g, result0b, result1r, result1g, result1b, result2r, result2g, result2b, result3r, result3g, result3b]
    
    return results 
        


def upscaleImg(img, regressions):
    neighborhoods = getNeighborhoods(img, 5)
#     neighborhoods = np.insert(neighborhoods,0,1,axis =1)

    #pixelColorValues
    PCV = []
    #get rid of the the extra dimension
    clippedRegressions = []
    for regression in regressions:
        regression = regression[1:]
        clippedRegressions.append(regression)
    for i in range (0,12):
        values = neighborhoods @ clippedRegressions[i]
        PCV.append(values)


    #create final image
    finalImageWidth = np.shape(img)[1] * 2
    finalImageHeight = np.shape(img)[0] * 2
    toReturn = np.ndarray(shape=(finalImageHeight,finalImageWidth,3))
    for curRow in range(0,finalImageHeight):

        # only add an offset for every other row and max offset will only be half the width of upscaled image
        rowOffset = curRow//2 * (finalImageWidth//2)
        for curCol in range(0,finalImageWidth):
            #placeToIndex 
            pti = curCol//2 + rowOffset
            if curRow % 2 == 0:
                if curCol % 2 == 0:
                    pixels = [PCV[0][pti],PCV[1][pti],PCV[2][pti]]
                    toReturn[curRow][curCol] = pixels
                else:
                    pixels = [PCV[3][pti],PCV[4][pti],PCV[5][pti]]
                    toReturn[curRow][curCol] = pixels
            else:
                if curCol % 2 == 0:
                    pixels = [PCV[6][pti],PCV[7][pti],PCV[8][pti]]
                    toReturn[curRow][curCol] = pixels
                else:
                    pixels = [PCV[9][pti],PCV[10][pti],PCV[11][pti]]
                    toReturn[curRow][curCol] = pixels
    return toReturn
    
def openImages(imagesDirectory):
    images = []
    for file in Path(imagesDirectory).iterdir():
        try:
            im = imread(file)
            images.append(im)
        except:
            print("can't read a file, moving to next")
    return images
    


def main(pathToImages, imageToUpscale):

    # print(pathToImages, imageToUpscale)
    images = openImages(pathToImages)
    if len(images) <= 0:
        print("no images found in directory")
        return 0
    toUpscale = [] 
    try:
        toUpscale = imread(imageToUpscale)
    except:
        print("Image to upscale cannot be opened")
        
    regressions = calcRegression(images)
    upscaled = upscaleImg(toUpscale,regressions)
    plt.imshow(upscaled)
    imwrite('upscaled.jpg', upscaled[:, :, :])



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: pathToImagesFolder, fileName for image to upscale")
    main(sys.argv[1],sys.argv[2])