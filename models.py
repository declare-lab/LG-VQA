import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, CLIPConfig
from transformers import CLIPSegProcessor, CLIPSegModel, CLIPSegConfig
from transformers import BlipProcessor, BlipModel, BlipConfig

from lavis.models import load_model_and_preprocess
# from lavis.models.blip2_models.blip2 import disabled_train, LayerNorm
    

class CLIPGuidedVQA(nn.Module):
    def __init__(
        self,
        name: str,
        num_choices: int,
    ):
        super().__init__()
        
        self.name = name
        self.num_choices = num_choices
            
        self.model = CLIPModel.from_pretrained(self.name)
        self.processor = CLIPProcessor.from_pretrained(self.name)
        self.config = CLIPConfig.from_pretrained(self.name)
            
        self.loss_func = nn.CrossEntropyLoss()
            
            
    def extend_position_embeddings(self, total_positions=512):
        num_additional_positions = total_positions - self.processor.tokenizer.model_max_length

        trained_embeddings = self.model.text_model.embeddings.position_embedding.weight.detach()
        new_embeddings = nn.Embedding(
            num_additional_positions, self.config.text_config.projection_dim
        ).to(self.model.device).weight.detach()

        final_embeddings = torch.cat([trained_embeddings, new_embeddings], 0)
        final_embeddings = nn.Embedding(
            self.processor.tokenizer.model_max_length + num_additional_positions, 
            self.config.text_config.projection_dim, _weight=final_embeddings
        ).to(self.model.device)
        
        self.model.text_model.embeddings.position_embedding = final_embeddings
        
        for param in self.model.text_model.embeddings.position_embedding.parameters():
            param.requires_grad = True

        self.config.text_config.max_position_embeddings += num_additional_positions
        self.model.config.text_config.max_position_embeddings += num_additional_positions
        self.model.text_model.config.max_position_embeddings += num_additional_positions
        self.model.text_model.embeddings.register_buffer(
            "position_ids", torch.arange(self.config.text_config.max_position_embeddings).expand((1, -1)).to(self.model.device)
        )
        self.processor.tokenizer.model_max_length = total_positions
        
        
    def score_input(self, images, texts, knowledge):
        
        flat_texts = [item for sublist in texts for item in sublist]
        flat_knowledge = [i for x in zip(*[knowledge]*self.num_choices) for i in x]
        merged = ["{} [SEP] {}".format(c, k) for c, k in zip(flat_texts, flat_knowledge)]
        
        inputs = self.processor(
            text=merged, images=images, return_tensors="pt", padding=True,
            truncation=True, max_length=self.processor.tokenizer.model_max_length
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        
        batch_size = len(images)
        text_embeds = outputs["text_embeds"].reshape(batch_size, self.num_choices, -1)
        image_embeds = torch.repeat_interleave(outputs["image_embeds"].unsqueeze(1), self.num_choices, 1)
        
        scale = self.model.logit_scale.exp()
        similarity_logits = torch.einsum('bij,bij->bi', text_embeds, image_embeds) * scale
        return similarity_logits
    

    def forward(self, batch):
        images, texts, knowledge, labels, qid = batch
        similarity_logits = self.score_input(images, texts, knowledge)
        labels = torch.tensor(labels, dtype=torch.long).to(similarity_logits.device)
        loss = self.loss_func(similarity_logits, labels)
        labels = list(labels.cpu().numpy())
        preds = list(torch.argmax(similarity_logits, 1).cpu().numpy())
        return loss, preds, labels, qid
    
    
class BLIP2GuidedVQA(nn.Module):
    def __init__(
        self,
        name: str,
        fusion: str,
        combine: str,
        num_choices: int,
    ):
        super().__init__()
        
        self.name = name
        self.fusion = fusion
        self.combine = combine
        self.num_choices = num_choices
        
        self.model, self.vis_processors, self.text_processors = load_model_and_preprocess("blip2_image_text_matching", self.name)
        output_dim = self.model.Qformer.config.hidden_size
        self.output_dim = output_dim
        
        self.loss_func = nn.CrossEntropyLoss()
        
        self.ikm_head = nn.Linear(self.model.Qformer.config.hidden_size, 1)
            
        if self.combine == "features":
            self.merged_feature_head = nn.Linear(4 * self.model.Qformer.config.hidden_size, 1)
            
            
    def tokenize_and_encode(self, sentences, image_embeds=None):
        text = self.model.tokenizer(
            sentences, padding=True, truncation=True, max_length=self.model.max_txt_len, return_tensors="pt"
        ).to(self.model.device)

        if image_embeds is not None:
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.model.device)
            query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(self.model.device)
            
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)
            output_ikm = self.model.Qformer.bert(
                text.input_ids, query_embeds=query_tokens, attention_mask=attention_mask,
                encoder_hidden_states=image_embeds, encoder_attention_mask=image_atts, return_dict=True
            )
            
            features = output_ikm.last_hidden_state[:, : query_tokens.size(1), :]
            features = features.mean(1)
        else:
            text_output = self.model.Qformer.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True)
            features = text_output.last_hidden_state[:, 0, :]
            
        return features
        
    def score_input(self, images, texts, knowledge, qid):
        
        flat_images = [i for x in zip(*[images]*self.num_choices) for i in x]
        flat_texts = [item for sublist in texts for item in sublist]
        flat_knowledge = [i for x in zip(*[knowledge]*self.num_choices) for i in x]
        merged = ["{} [SEP] {}".format(c, k) for c, k in zip(flat_texts, flat_knowledge)]
        
        # Match between image and choices
        image_batch = torch.cat([self.vis_processors["eval"](image).unsqueeze(0).to(self.model.device) for image in flat_images], 0)
        text_batch = [self.text_processors["eval"](ft) for ft in flat_texts]
        batch = {"image": image_batch, "text_input": text_batch}
        logits, features, image_embeds = self.model(batch, match_head="itm")
        question_logits = logits[:, 1].reshape(-1, self.num_choices)
        
        # Match between image and choices with guidance
        merged_batch = [self.text_processors["eval"](ft) for ft in merged]
        
        if self.fusion == "concat":
            knowledge_features = self.tokenize_and_encode(merged_batch)
            knowledge_logits = self.ikm_head(knowledge_features).reshape(-1, self.num_choices)
        
        elif self.fusion == "concat-image":
            knowledge_features = self.tokenize_and_encode(merged_batch, image_embeds)
            knowledge_logits = self.ikm_head(knowledge_features).reshape(-1, self.num_choices)

        if self.combine == "logits":
            similarity_logits = question_logits + knowledge_logits
        elif self.combine == "only-knowledge":
            similarity_logits = knowledge_logits
        elif self.combine == "features":
            merged = torch.cat([features, knowledge_features, features-knowledge_features, features*knowledge_features], 1)
            similarity_logits = self.merged_feature_head(merged).reshape(-1, self.num_choices)
            
        return similarity_logits

    def forward(self, batch):
        images, texts, knowledge, labels, qid = batch
        similarity_logits = self.score_input(images, texts, knowledge, qid)
        labels = torch.tensor(labels, dtype=torch.long).to(similarity_logits.device)
        loss = self.loss_func(similarity_logits, labels)
        labels = list(labels.cpu().numpy())
        preds = list(torch.argmax(similarity_logits, 1).cpu().numpy())
        return loss, preds, labels, qid