import PIL.Image as Image
from collections import Counter
from num2words import num2words

import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

class ObjectDetectionModel:
    def __init__(self, device="cuda:0"):
        self.device = device
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50").to(device)
        self.model.eval()
        
    def detect_objects(self, image_path, style="counts"):
        
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        
        # keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
        objects = [self.model.config.id2label[l] for l in results["labels"].cpu().numpy()]
        
        detected = ""
        
        if style == "counts":
            if len(objects) > 0:
                counts = Counter(objects)
                counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
                detected = ", ".join(["{} {}".format(num2words(item[1]), item[0]) for item in counts])
                
        elif style == "positions":
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                detected += f"{self.model.config.id2label[label.item()]} at location {box}; "

            detected = detected[:-2]

        return detected