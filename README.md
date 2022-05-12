# Interpreting Language Models with Contrastive Explanations

Code supporting the paper [Interpreting Language Models with Contrastive Explanations](https://arxiv.org/abs/2202.10419)

Currently supports:
* Contrastive explanations for language models (GPT-2, GPT-Neo) [(Colab)](https://colab.research.google.com/drive/1L6VjQ9_XAlbkPENmJxMpCntR3X_grpih?usp=sharing)
* Contrastive explanations for NMT models (MarianMT) [(Colab)](https://colab.research.google.com/drive/1rkSOGGxinVH_pzHxmswtt0ZDKmgf-sPL?usp=sharing)

## Requirements 
* [PyTorch](https://pytorch.org/get-started/locally/) >= 1.11.0
* [SentencePiece](https://github.com/google/sentencepiece) >= 0.1.90
* [Transformers](https://github.com/huggingface/transformers)
* Python >= 3.6

## Examples
### 1. Load models

LM:
```
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

NMT:
```
from transformers import MarianTokenizer, MarianMTModel

model_name = f"Helsinki-NLP/opus-mt-en-fr" 
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

### 2. Define inputs

LM:
```
input = "Can you stop the dog from "
input_tokens = tokenizer(input)['input_ids']
attention_ids = tokenizer(input)['attention_mask']
```

NMT:
```
encoder_input = "I can't find the seat, do you know where it is?"
decoder_input = "Je ne trouve pas la place, tu sais où"
decoder_input = f"<pad> {decoder_input.strip()} "

input_ids = tokenizer(encoder_input, return_tensors="pt").input_ids.to(device)
decoder_input_ids = tokenizer(decoder_input, return_tensors="pt", add_special_tokens=False,).input_ids.to(device)
```

### 3. Visualize explanations

LM:
```
from lm_saliency import *

target = "barking"
foil = "crying"
CORRECT_ID = tokenizer(" "+ target)['input_ids'][0]
FOIL_ID = tokenizer(" "+ foil)['input_ids'][0]

base_saliency_matrix, base_embd_matrix = saliency(model, input_tokens, attention_ids)
saliency_matrix, embd_matrix = saliency(model, input_tokens, attention_ids, foil=FOIL_ID)

# Input x gradient
base_explanation = input_x_gradient(base_saliency_matrix, base_embd_matrix, normalize=True)
contra_explanation = input_x_gradient(saliency_matrix, embd_matrix, normalize=True)

# Gradient norm
base_explanation = l1_grad_norm(base_saliency_matrix, normalize=True)
contra_explanation = l1_grad_norm(saliency_matrix, normalize=True)

# Erasure
base_explanation = erasure_scores(model, input_tokens, attention_ids, normalize=True)
contra_explanation = erasure_scores(model, input_tokens, attention_ids, correct=CORRECT_ID, foil=FOIL_ID, normalize=True)

visualize(np.array(base_explanation), tokenizer, [input_tokens], print_text=True, title=f"Why did the model predict {target}?")
visualize(np.array(contra_explanation), tokenizer, [input_tokens], print_text=True, title=f"Why did the model predict {target} instead of {foil}?")
```

NMT:
```
from lm_saliency import visualize
from mt_saliency import *

target = "elle"
foil = "il"
CORRECT_ID = tokenizer(" "+ target)['input_ids'][0]
FOIL_ID = tokenizer(" "+ foil)['input_ids'][0]

base_enc_saliency, base_enc_embed, base_dec_saliency, base_dec_embed = saliency(model, input_ids, decoder_input_ids)
enc_saliency, enc_embed, dec_saliency, dec_embed = saliency(model, input_ids, decoder_input_ids, foil=FOIL_ID)

# Input x gradient
base_enc_explanation = input_x_gradient(base_enc_saliency, base_enc_embed, normalize=False)
base_dec_explanation = input_x_gradient(base_dec_saliency, base_dec_embed, normalize=False)
enc_explanation = input_x_gradient(enc_saliency, enc_embed, normalize=False)
dec_explanation = input_x_gradient(dec_saliency, dec_embed, normalize=False)

# Gradient norm
base_enc_explanation = l1_grad_norm(base_enc_saliency, normalize=False)
base_dec_explanation = l1_grad_norm(base_dec_saliency, normalize=False)
enc_explanation = l1_grad_norm(enc_saliency, normalize=False)
dec_explanation = l1_grad_norm(dec_saliency, normalize=False)  

# Erasure
base_enc_explanation, base_dec_explanation = erasure_scores(model, input_ids, decoder_input_ids, correct=CORRECT_ID, normalize=False)
enc_explanation, dec_explanation = erasure_scores(model, input_ids, decoder_input_ids, correct=CORRECT_ID, foil=FOIL_ID, normalize=False)

# Normalize
base_norm = np.linalg.norm(np.concatenate((base_enc_explanation, base_dec_explanation)), ord=1)
base_enc_explanation /= base_norm
base_dec_explanation /= base_norm
norm = np.linalg.norm(np.concatenate((enc_explanation, dec_explanation)), ord=1)
enc_explanation /= norm
dec_explanation /= norm

# Visualize
visualize(base_enc_explanation, tokenizer, input_ids, print_text=True, title=f"Why did the model predict {target}? (encoder input)")
visualize(base_dec_explanation, tokenizer, decoder_input_ids, print_text=True, title=f"Why did the model predict {target}? (decoder input)")
visualize(enc_explanation, tokenizer, input_ids, print_text=True, title=f"Why did the model predict {target} instead of {foil}? (encoder input)")
visualize(dec_explanation, tokenizer, decoder_input_ids, print_text=True, title=f"Why did the model predict {target} instead of {foil}? (decoder input)")
```
