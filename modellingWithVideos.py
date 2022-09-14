import cv2
from time import sleep
import torch
import matplotlib.colors as mcolors
import time


def convert2RGB(hex_colour):
    rgb_colour = mcolors.to_rgb(hex_colour)
    rgb_colour = tuple(map(lambda x: round(x*255), rgb_colour))
    
    return rgb_colour

colours = list(mcolors.CSS4_COLORS.values())

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')

def current_milli_time():
    return round(time.time() * 1000)




name= "siuu"

cv2.namedWindow(name)
vc = cv2.VideoCapture(0)


if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False
    
print(bool(rval))

while rval:
    
    frame = cv2.flip(frame, 0)
        
        
    starttime = current_milli_time()
    results = model(frame)    
    stoptime = current_milli_time()
    
    tensor1 = results.pred[0]





        # results.show()


        #print(tensor1)



    i = 0
    while i < tensor1.size(dim=0):
        xmin = round(tensor1[i][0].item())
        ymin = round(tensor1[i][1].item())
        xmax = round(tensor1[i][2].item())
        ymax = round(tensor1[i][3].item())
        class_index = round(tensor1[i][5].item())
        class_name = results.names[class_index]
        conf = tensor1[i][4].item()

        print(f"detected {class_name} at x{xmin} y{ymin} - x{xmax} y{ymax} with confidence {conf}")
            
        c = convert2RGB(colours[class_index])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (c), 4)
            
            
            
        i = i + 1


    print(f"System time: {stoptime-starttime} ms")

    cv2.imshow(name, frame)
    rval, frame = vc.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

          
cv2.destroyWindow(name)
vc.release()
