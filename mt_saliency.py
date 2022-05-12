import argparse, json
import random
import torch
import numpy as np
from transformers import (
    WEIGHTS_NAME,
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,
)

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 10]

config = GPT2Config.from_pretrained("gpt2")
VOCAB_SIZE = config.vocab_size

def model_preds(model, input_ids, decoder_input_ids, tokenizer, k=10, verbose=False):
    softmax = torch.nn.Softmax(dim=0)
    A = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    probs = softmax(A.logits[0][-1])
    top_preds = probs.topk(k)
    if verbose:
        print("Top model predictions:")
        for p,i in zip(top_preds.values, top_preds.indices):
            print(f"{np.round(p.item(), 3)}: {tokenizer.decode(i)}")
    return top_preds.indices

# Adapted from AllenNLP Interpret and Han et al. 2020

def register_embedding_list_hook(model, embeddings_list):
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())
    embedding_layer = model.model.encoder.embed_tokens
    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle

def register_embedding_gradient_hooks(model, embeddings_gradients):
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0].detach().cpu().numpy())
    embedding_layer = model.model.encoder.embed_tokens
    hook = embedding_layer.register_backward_hook(hook_layers)
    return hook

def saliency(model, input_ids, decoder_input_ids, batch=0, correct=None, foil=None):
    torch.enable_grad()
    model.eval()
    embeddings_list = []
    handle = register_embedding_list_hook(model, embeddings_list)
    embeddings_gradients = []
    hook = register_embedding_gradient_hooks(model, embeddings_gradients)
    
    if correct is None:
        correct = input_ids[0][-1]
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(model.device)
    decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long).to(model.device)

    model.zero_grad()
    A = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

    if foil is not None:
        if correct == foil:
            (A.logits[0][-1][correct]).backward()
        else:
            (A.logits[0][-1][correct]-A.logits[0][-1][foil]).backward()
    else:
        (A.logits[0][-1][correct]).backward()
    handle.remove()
    hook.remove()

    dec_saliency, enc_saliency = embeddings_gradients
    enc_embed, dec_embed = embeddings_list
    return enc_saliency.squeeze(), enc_embed, dec_saliency.squeeze(), dec_embed

def input_x_gradient(grads, embds, normalize=False):
    # same as LM saliency
    input_grad = np.sum(grads * embds, axis=-1).squeeze()

    if normalize:
        norm = np.linalg.norm(input_grad, ord=1)
        input_grad /= norm
        
    return input_grad

def l1_grad_norm(grads, normalize=False):
    # same as LM saliency
    l1_grad = np.linalg.norm(grads, ord=1, axis=-1).squeeze()

    if normalize:
        norm = np.linalg.norm(l1_grad, ord=1)
        l1_grad /= norm
    return l1_grad


def erasure_scores(model, input_ids, decoder_input_ids, correct=None, foil=None, normalize=False):
    model.eval()
    if correct is None:
        correct = input_ids[0][-1]
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(model.device)
    decoder_input_ids = torch.tensor(decoder_input_ids, dtype=torch.long).to(model.device)
    
    A = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    softmax = torch.nn.Softmax(dim=0)
    logits = A.logits[0][-1]
    probs = softmax(logits)
    if foil is not None and correct != foil:
        base_score = (probs[correct]-probs[foil]).detach().cpu().numpy()
    else:
        base_score = (probs[correct]).detach().cpu().numpy()

    enc_scores = np.zeros(len(input_ids[0]))
    for i in range(len(input_ids[0])):
        input_ids_i = torch.cat((input_ids[0][:i], input_ids[0][i+1:])).unsqueeze(0)
        A = model(input_ids=input_ids_i, decoder_input_ids=decoder_input_ids)
        logits = A.logits[0][-1]
        probs = softmax(logits)
        if foil is not None and correct != foil:
            erased_score = (probs[correct]-probs[foil]).detach().cpu().numpy()
        else:
            erased_score = (probs[correct]).detach().cpu().numpy()
                    
        enc_scores[i] = base_score - erased_score # higher score = lower confidence in correct = more influential input

    dec_scores = np.zeros(len(decoder_input_ids[0]))
    for i in range(len(decoder_input_ids[0])):
        decoder_input_ids_i = torch.cat((decoder_input_ids[0][:i], decoder_input_ids[0][i+1:])).unsqueeze(0)
        A = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids_i)
        logits = A.logits[0][-1]
        probs = softmax(logits)
        if foil is not None and correct != foil:
            erased_score = (probs[correct]-probs[foil]).detach().cpu().numpy()
        else:
            erased_score = (probs[correct]).detach().cpu().numpy()
                    
        dec_scores[i] = base_score - erased_score # higher score = lower confidence in correct = more influential input
    
    
    if normalize:
        norm = np.linalg.norm(enc_scores, ord=1)
        enc_scores /= norm
        norm = np.linalg.norm(dec_scores, ord=1)
        dec_scores /= norm
        
    return enc_scores, dec_scores


def main():
    pass

if __name__ == "__main__":
    main()
