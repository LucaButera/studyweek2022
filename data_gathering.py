import cv2
from time import sleep
import torch
import matplotlib.colors as mcolors
import time
import json
import argparse
from pathlib import Path

def current_milli_time():
    return round(time.time() * 1000)

def main(save_folder, model_type='n', n=100, bsmax=8):
    
    save_folder = Path.home().joinpath('studyweek2022', save_folder)
    
    if not save_folder.exists():
        
        save_folder.mkdir(exist_ok=True)
    
    assert model_type in ['n', 's', 'm', 'l', 'x']

    model_name = 'yolov5' + model_type
    model = torch.hub.load('ultralytics/yolov5', model_name)
    
    print(f"Running with model {model_name} up to batch size {bsmax} with {n} iterations each.")

    vc = cv2.VideoCapture(0)
    data_results = {}

    for bs in range(1, bsmax +1):
        
        total_result = 0
        
        for j in range(n):
            
            frames = []

            for i in range(bs):
                
                if vc.isOpened(): 
                    rval, frame = vc.read()
                    frames.append(frame)
                              
            starttime = current_milli_time()
            results = model(frames)
            stoptime = current_milli_time()

            result_time = (stoptime-starttime)/bs
            
            print(f"System time per frame for batch size {bs}: {result_time} ms")

            total_result = total_result + result_time
                   
        avg_time = total_result/n
        data_results[bs] = avg_time
        print()
        print(f"Average time for {bs} frame(s): {avg_time} ms")
        print()
        
        with save_folder.joinpath(f"results_{model_name}.json").open("w") as fp:
                json.dump(data_results, fp)
        
    vc.release()
    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", "-m", type=str, help="which model do you want to run?", default='n')
    parser.add_argument("-n", type=int, help="How many iterations?", default=100)
    parser.add_argument("--bsmax", "-b", type=int, help="Maximum frame batch size", default=8)
    parser.add_argument("save_folder", required=True, type=str)
    args = parser.parse_args()
   
    main(args.save_folder, args.model_type, args.n, args.bsmax)