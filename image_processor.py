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
     # Image ranges
    y1, y2 = max(0, y), min(background.shape[0], y + overlay.shape[0])
    x1, x2 = max(0, x), min(background.shape[1], x + overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(overlay.shape[0], background.shape[0] - y)
    x1o, x2o = max(0, -x), min(overlay.shape[1], background.shape[1] - x)

    alphaOverlay = overlay[y1o:y2o,x1o:x2o,3]/255.0
    alphaBackground = 1- alphaOverlay


    bg_crop = background[y1:y2, x1:x2]
    overlay_crop = overlay[y1o:y2o, x1o:x2o]

    # for channel in range(0,3):
    #     print(np.shape(alphaOverlay * overlay[y1o:y2o, x1o:x2o,channel]))
    #     print(np.shape(alphaBackground * background[y1:y2, x1:x2,channel]))
    # # cv2.addWeighted(overlay,1,background,0,0,background)
    for channel in range(0,3):
        background[y1:y2,x1:x2,channel] = (alphaOverlay * overlay[y1o:y2o, x1o:x2o,channel] + alphaBackground * background[y1:y2, x1:x2,channel])
    # background = alphaOverlay * overlay_crop + alphaBackground * bg_crop
    return background



# img = loadImage("./data/glasses.png")
# print(np.shape(img))
# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.imshow('image', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()