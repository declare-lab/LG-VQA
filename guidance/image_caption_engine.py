import numpy as np
import requests
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess
from transformers import CLIPProcessor, CLIPModel

class ImageCaptioning:
    def __init__(self, device):
        self.device = device
            
        self.model, self.processor, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl"
        )
                
        self.model.eval()
        self.model.to(self.device)
        
        clip_model_name = "openai/clip-vit-base-patch32"
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.clip_model.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        
        
    def score_captions(self, image, captions):
        inputs = self.clip_processor(text=captions, images=[image], return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.clip_model(**inputs)

        batch_size, num_captions = 1, len(captions)
        text_embeds = outputs["text_embeds"].reshape(batch_size, num_captions, -1)
        image_embeds = torch.repeat_interleave(outputs["image_embeds"].unsqueeze(1), num_captions, 1)

        similarity_logits = torch.einsum('bij,bij->bi', text_embeds, image_embeds)[0].cpu().numpy() * 100
        similarity_logits = similarity_logits.round(2)
        order = np.argsort(similarity_logits)
        ordered_captions = [captions[i] for i in order[::-1]]
        scores = list(similarity_logits[order[::-1]].astype("float64").round(2))
        return ordered_captions, scores

    
    def image_caption(self, image_src, num_captions=10):
        
        image = Image.open(image_src).convert("RGB")
        batch = self.processor["eval"](image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            captions = self.model.generate(
                {"image": batch}, use_nucleus_sampling=True, num_captions=num_captions, top_p=0.95, 
                temperature=1.0, num_beams=1
            )
            
        ordered_captions, scores = self.score_captions(image, captions)
        return ordered_captions, scores