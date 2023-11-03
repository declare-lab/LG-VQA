import os
import copy
import json
import pandas as pd
from tqdm import tqdm
from engine import GuidanceModel
from argparse import ArgumentParser

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--device", type=int, default=0, help="CUDA device id.")
    parser.add_argument("--file", type=str, default="", help="json lines file with 'image_path' key.")
    args = parser.parse_args()
    
    device = "cuda:{}".format(args.device)
    model_args = {"device": device}
    model = GuidanceModel(model_args)

    data = [json.loads(line) for line in open(args.file).readlines()]
    
    new_data = copy.deepcopy(data)
    
    all_records = []
    for k, instance in tqdm(enumerate(data)):
        output = model.image_to_guidance(instance['image_path'])
        all_records.append(output)
            
        new_data[k]["objects"] = output[0]
        new_data[k]["scene_graph"] = output[1]
        new_data[k]["caption"] = output[2]
        new_data[k]["scores"] = output[3]
        
    with open(args.file.replace(".json", "_with_guidance.json"), "w") as f:
        for line in new_data:
            f.write(json.dumps(line) + "\n")    