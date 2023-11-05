import torch
from transformers import AutoTokenizer, BertForPreTraining


class MakeEmbedding:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = BertForPreTraining.from_pretrained("bert-base-uncased")

    def get_embedding(self, string) -> torch.Tensor:
        inputs = self.tokenizer(string, return_tensors="pt")
        outputs = self.model(**inputs)

        return outputs.prediction_logits
