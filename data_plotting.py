import seaborn as sns
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def gather_jsons(folder: Path)->list[Path]:
    
    files = []
    
    for p in folder.iterdir():
        
        if p.is_file() and p.suffix == '.json':
            
            files.append(p)
            
    return files

def main(folder, model = None):
    
    file_name = f'Average_time_chart_yolov5{model if model is not None else "all"}_{folder}.pdf' 
        
    folder = Path.home().joinpath('studyweek2022', folder)
    
    data_files = gather_jsons(folder) if model is None else [folder.joinpath(f"results_yolov5{model}.json")]
        
    print(f"Loading data from {data_files}.")
    
    data = {}
    
    
    
    for p in data_files:
        
        with p.open("r") as fp:           
            data[p.stem[8:]] = json.load(fp)
            
            
    pd_data = {'Model': [], 'Batch Size': [], 'Average time (ms)': []}
    
    for m in data:
        
        for b in data[m]:
            
            pd_data['Model'].append(m)
            pd_data['Batch Size'].append(b)
            pd_data['Average time (ms)'].append(data[m][b])
            
    result_frame = pd.DataFrame(data=pd_data)
    
    print(result_frame)

    sns.lineplot(data=result_frame, x='Batch Size', y='Average time (ms)', hue='Model')
    
    plt.savefig(Path.home().joinpath('studyweek2022', 'charts', file_name), format="pdf")
    
    plt.show()
       
    print("Done")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", "-f", type=str, help="Which Folder is used?", required=True)
    parser.add_argument("--model_type", "-m", type=str, help="which model do you want to display?", default=None)
    args = parser.parse_args()
    
    main(args.folder, args.model_type)