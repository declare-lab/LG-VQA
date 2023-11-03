import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from scene_graph_engine import SceneGraphModel
from image_caption_engine import ImageCaptioning
from object_detection_engine import ObjectDetectionModel

class GuidanceModel:
    def __init__(self, args):
        self.args = args
        self.init_models()
        self.ref_image = None
    
    def init_models(self):
        device = self.args["device"]        
        print("Initializing models .. ")
        
        self.scene_graph_model = SceneGraphModel(device)
        self.image_caption_model = ImageCaptioning(device)
        self.object_detection_model = ObjectDetectionModel(device)
        
        print("Model initialization finished!")

    
    def image_to_guidance(self, image_path):    
        with torch.no_grad():
            objects = self.object_detection_model.detect_objects(image_path, "counts")
            scene_graph = " <SEP> ".join(self.scene_graph_model.obtain_scene_graph(image_path))
            caption, scores = self.image_caption_model.image_caption(image_path)
            
        return [objects, scene_graph, caption, scores]