from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_input(text):
	tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
	model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

	encoded_input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')

	with torch.no_grad():
		model_output = model(**encoded_input)

	sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

	sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

	return sentence_embeddings
