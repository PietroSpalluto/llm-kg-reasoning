from transformers import AutoTokenizer, BertForPreTraining


class MakeEmbedding:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
        self.model = BertForPreTraining.from_pretrained("bert-large-uncased")

    def get_embedding(self, string):
        inputs = self.tokenizer(string, return_tensors="pt")
        outputs = self.model(**inputs)

        return outputs.prediction_logits
