import cv2

name= "siuu"

cv2.namedWindow(name)
vc = cv2.VideoCapture(0)


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
    


while rval:
    frame = cv2.flip(frame, 0)
    cv2.imshow(name, frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow(name)
vc.release()