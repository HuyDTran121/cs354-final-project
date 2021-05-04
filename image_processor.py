from PIL import Image
import numpy as np
import cv2

def loadImage(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # img = img.convert("RGBA")
    print(type(img))
    return img

def drawImage(overlay, background, x, y):
    height, width,_ = np.shape(overlay)
    bgheight, bgwidth,_ = np.shape(background)
    rbound = min(x + width, bgwidth)
    dbound = min(y + height, bgheight)
    alphaOverlay = overlay[:,:,3]/255.0
    alphaBackground = 1- alphaOverlay
    # print(np.shape(alphaOverlay * overlay[:,:,0:3]))
    # cv2.addWeighted(overlay,1,background,0,0,background)
    for channel in range(0,3):
        background[y:dbound,x:rbound,channel] = (alphaOverlay * overlay[:,:,channel] + alphaBackground * background[y:dbound,x:rbound,channel])
    return background



# img = loadImage("./data/glasses.png")
# print(np.shape(img))
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()