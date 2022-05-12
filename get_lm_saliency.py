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

# Adapted from AllenNLP Interpret and Han et al. 2020

def register_embedding_list_hook(model, embeddings_list):
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())
    embedding_layer = model.transformer.wte
    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle

def register_embedding_gradient_hooks(model, embeddings_gradients):
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0].detach().cpu().numpy())
    embedding_layer = model.transformer.wte
    hook = embedding_layer.register_backward_hook(hook_layers)
    return hook

def input_x_gradient(grads, embds, normalize=False):
    input_grad = np.sum(grads * embds, axis=-1)
    if len(input_grad.shape) > 2:
        input_grad = np.squeeze(input_grad)

    if normalize:
        input_grad = np.squeeze(input_grad)
        norm = np.linalg.norm(input_grad, ord=1)
        input_grad =[e / norm for e in input_grad]
        
    return input_grad

def l1_grad_norm(grads, normalize=False):
    l1_grad = np.linalg.norm(grads, ord=1, axis=-1)
    if len(l1_grad.shape) > 2:
        l1_grad = np.squeeze(l1_grad)
    if normalize:
        l1_grad = np.squeeze(l1_grad)
        norm = np.linalg.norm(l1_grad, ord=1)
        l1_grad = [e / norm for e in l1_grad] 
    return l1_grad

def vectorize(line, correct, tokenizer, model, pos=None):
    tokens = tokenizer(line)
    input_ids = torch.tensor([tokens['input_ids']], dtype=torch.long).to(model.device)
    attention_ids = torch.tensor([tokens['attention_mask']], dtype=torch.long).to(model.device)
    
    token_start = 0
    target_idx = []
    for i, token in enumerate(input_ids[0]):
        token_len = len(tokenizer.decode(token))
        if tokenizer.decode(token) != line[token_start : token_start + token_len]:
            token_len += 1
        if line[token_start : token_start + token_len].strip() == correct:
            target_idx.append(i)
        token_start += token_len
    if pos:
        try:
            return target_idx[pos]
        except:
            target_pos = random.choice(target_idx)
    else:
        target_pos = random.choice(target_idx)

    target_id = input_ids[0][target_pos]
    return input_ids, attention_ids, target_id, target_pos

def logits_to_probs(a):
    p = np.exp(a)
    return p / (1 + p)


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

def model_proba(model, input_ids, input_mask, pos, target=None):
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(model.device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(model.device)
    A = model(input_ids[:pos], attention_mask=input_mask[:pos])
    if target:
        return A.logits[pos-1][target]
    return A.logits[pos-1][input_ids[pos]]


def all_saliency(model, input_ids, input_mask, correct=None, foils=None, pos=-1):
    saliency_matrix = []
    embd_matrix = []
    for i, (input_id, input_m, pos_i) in enumerate(zip(input_ids, input_mask, pos)):
        if foils is not None:
            saliency_mat = []
            embd_mat = []
            for foil in foils[i]:
                sal, embd = saliency(model, np.array([input_id]), np.array([input_m]), pos=pos_i, correct=correct, foil=foil)
                saliency_mat.append(sal)
                embd_mat.append(embd)
            saliency_mat = np.stack(saliency_mat)
            embd_mat = np.stack(embd_mat)
        else:
            saliency_mat, embd_mat = saliency(model, np.array([input_id]), np.array([input_m]), pos=pos_i, correct=correct)
        saliency_matrix.append(saliency_mat.tolist()) # num_sent x vocab_size x input_size x n_embd
        embd_matrix.append(np.array(embd_mat).tolist())
    
    return np.array(saliency_matrix, dtype=object), np.array(embd_matrix, dtype=object)

def saliency(model, input_ids, input_mask, batch=0, pos=-1, correct=None, foil=None):
    torch.enable_grad()
    model.eval()
    embeddings_list = []
    handle = register_embedding_list_hook(model, embeddings_list)
    embeddings_gradients = []
    hook = register_embedding_gradient_hooks(model, embeddings_gradients)
    
    if correct is None:
        correct = input_ids[0][pos]
    input_ids = input_ids[:, :pos]
    input_mask = input_mask[:, :pos]
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(model.device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(model.device)

    model.zero_grad()
    A = model(input_ids, attention_mask=input_mask)

    if foil is not None:
        if correct == foil:
            (A.logits[0][pos-1][correct]).backward()
        else:
            (A.logits[0][pos-1][correct]-A.logits[0][pos-1][foil]).backward()
    else:
        (A.logits[0][pos-1][correct]).backward()
    handle.remove()
    hook.remove()

    saliency_grad = embeddings_gradients[0]
    return saliency_grad[0], embeddings_list[0]

def erasure_scores(model, input_ids, input_mask, pos=-1, correct=None, foil=None, remove=False, normalize=False):
    model.eval()
    if correct is None:
        correct = input_ids[pos]
    input_ids = input_ids[:pos]
    input_mask = input_mask[:pos]
    input_ids = torch.unsqueeze(torch.tensor(input_ids, dtype=torch.long).to(model.device), 0)
    input_mask = torch.unsqueeze(torch.tensor(input_mask, dtype=torch.long).to(model.device), 0)
    
    A = model(input_ids, attention_mask=input_mask)
    softmax = torch.nn.Softmax(dim=0)
    logits = A.logits[0][pos-1]
    probs = softmax(logits)
    if foil is not None and correct != foil:
        base_score = (logits[correct]-logits[foil]).detach().cpu().numpy()
    else:
        base_score = (probs[correct]).detach().cpu().numpy()

    scores = np.zeros(len(input_ids[0]))
    for i in range(len(input_ids[0])):
        if remove:
            input_ids_i = torch.cat((input_ids[0][:i], input_ids[0][i+1:])).unsqueeze(0)
            input_mask_i = torch.cat((input_mask[0][:i], input_mask[0][i+1:])).unsqueeze(0)
        else:
            input_ids_i = torch.clone(input_ids)
            input_mask_i = torch.clone(input_mask)
            input_mask_i[0][i] = 0

        A = model(input_ids_i, attention_mask=input_mask_i)
        logits = A.logits[0][-1]
        probs = softmax(logits)
        if foil is not None and correct != foil:
            erased_score = (logits[correct]-logits[foil]).detach().cpu().numpy()
        else:
            erased_score = (probs[correct]).detach().cpu().numpy()
                    
        scores[i] = base_score - erased_score # higher score = lower confidence in correct = more influential input
    if normalize:
        norm = np.linalg.norm(scores, ord=1)
        scores = [e / norm for e in scores] 
    return scores

def all_erasure_scores(model, input_ids, input_mask, correct=None, foils=None, pos=-1):
    all_scores = []
    for i, (input_id, input_m, pos_i) in enumerate(zip(input_ids, input_mask, pos)):
        if foils is not None:
            scores_mat = []
            for foil in foils[i]:
                scores = erasure_scores(model, input_id, input_m, pos=pos_i, correct=correct, foil=foil)
                scores_mat.append(scores)
            scores_mat = np.stack(scores_mat)
        else:
            scores_mat = np.array(erasure_scores(model, input_id, input_m, pos=pos_i, correct=correct))
        all_scores.append(scores_mat.tolist()) 
    return np.array(all_scores, dtype=object)

def replacement_scores(model, tokenizer, input_ids, input_mask, pos=-1, foil=None):
    model.eval()
    correct = input_ids[pos]
    input_ids = input_ids[:pos]
    input_mask = input_mask[:pos]
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(model.device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(model.device)
    
    A = model(input_ids, attention_mask=input_mask)
    if foil is not None:
        if correct == foil:
            base_score = (A.logits[pos-1][correct]).detach().cpu().numpy()
        else:
            base_score = (A.logits[pos-1][correct]-A.logits[pos-1][foil]).detach().cpu().numpy()
    else:
        base_score = (A.logits[pos-1][correct]).detach().cpu().numpy()
        
    scores = np.zeros(len(input_ids))
    replacements = np.zeros(len(input_ids))
    for i in tqdm(range(len(input_ids))):
        input_ids_i = torch.clone(input_ids)
        best_score = np.inf
        best_vocab = None
        for vocab_id in range(tokenizer.vocab_size):
            input_ids_i[i] = vocab_id
            A = model(input_ids_i, attention_mask=input_mask)
            if foil is not None and correct != foil:
                replaced_score = (A.logits[pos-2][correct]-A.logits[pos-2][foil]).detach().cpu().numpy()
            else:
                replaced_score = (A.logits[pos-2][correct]).detach().cpu().numpy()
            if replaced_score < best_score: # lower confidence of correct answer if we replace
                best_score = replaced_score
                best_vocab = vocab_id
        scores[i] =  base_score - best_score
        replacements[i] = vocab_id
    return scores, replacements

def all_replacement_scores(model, tokenizer, input_ids, input_mask, correct=None, foils=None, pos=-1):
    all_scores = []
    all_repl = []
    for i, (input_id, input_m, pos_i) in enumerate(zip(input_ids, input_mask, pos)):
        if foils is not None:
            scores_mat = []
            replacement_mat = []
            for foil in foils[i]:
                scores, replacements = replacement_scores(model, tokenizer, input_id, input_m, pos=pos_i, foil=foil)
                scores_mat.append(scores)
                replacement_mat.append(replacements)
            scores_mat = np.stack(scores_mat)
        else:
            scores_mat, replacement_mat = replacement_scores(model, tokenizer, input_id, input_m, pos=pos_i)
        all_scores.append(scores_mat.tolist()) 
        all_repl.append(replacement_mat.tolist()) 
    return np.array(all_scores, dtype=object), np.array(all_repl, dtype=object)
    

def saliency_map(model, input_ids, input_mask, batch=0, pos=-1, foil=None):
    torch.enable_grad()
    model.eval()
    embeddings_list = []
    handle = register_embedding_list_hook(model, embeddings_list)
    embeddings_gradients = []
    hook = register_embedding_gradient_hooks(model, embeddings_gradients)
        
    model.zero_grad()
    A = model(input_ids, attention_mask=input_mask)
    if foil is not None:
        correct = input_ids[batch][pos]
        if correct == foil:
            A.logits[0][pos-1][correct].backward()
        else:
            (A.logits[batch][pos-1][correct]-A.logits[0][pos-1][foil]).backward()
    else:
        pred_label_ids = np.argmax(A.logits[0][pos-1].detach().cpu().numpy())
        A.logits[0][pos-1][pred_label_ids].backward()
    handle.remove()
    hook.remove()

    saliency_grad = embeddings_gradients[0]
    l1_grad = np.mean(np.abs(saliency_grad[0]), axis=-1)
    input_grad = np.sum(saliency_grad[0] * embeddings_list[0]   , axis=1)
    norm = np.linalg.norm(input_grad, ord=1)
    input_grad = [e / norm for e in input_grad] 
    return input_grad, list(l1_grad)

def visualize(attention, tokenizer, input_ids, gold=None, normalize=False, print_text=True, save_file=None, title=None, figsize=60, fontsize=36):
    tokens = [tokenizer.decode(i) for i in input_ids[0][:len(attention) + 1]]
    if gold is not None:
        for i, g in enumerate(gold):
            if g == 1:
                tokens[i] = "**" + tokens[i] + "**"

    # Normalize to [-1, 1]
    if normalize:
        a,b = min(attention), max(attention)
        x = 2/(b-a)
        y = 1-b*x
        attention = [g*x + y for g in attention]
    attention = np.array([list(map(float, attention))])

    fig, ax = plt.subplots(figsize=(figsize,figsize))
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    im = ax.imshow(attention, cmap='seismic', norm=norm)

    if print_text:
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, fontsize=fontsize)
    else:
        ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for (i, j), z in np.ndenumerate(attention):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=fontsize)


    ax.set_title("")
    fig.tight_layout()
    if title is not None:
        plt.title(title, fontsize=36)
    
    if save_file is not None:
        plt.savefig(save_file, bbox_inches = 'tight',
        pad_inches = 0)
        plt.close()
    else:
        plt.show()

def all_saliency_maps(model, tokenizer, input_ids, input_mask):
    A = model(input_ids, attention_mask=input_mask)
    for batch_id in range(A.logits.size(0)):
        for pos_id in range(5, 6):
            for vocab_id in range(A.logits.size(2)):
                print(f"{tokenizer.decode(vocab_id)} ||| {saliency_map(model, input_ids, input_mask, batch=batch_id, pos=pos_id, foil=vocab_id)}")

def vocab_saliency(model, data, tokenizer, correct):
    for j, line in enumerate(data):
        tokens = tokenizer(line)
        line = tokenizer.decode(tokens['input_ids'])
        tokens = tokenizer(line)
        input_ids = torch.tensor([tokens['input_ids']], dtype=torch.long).to(model.device)
        attention_ids = torch.tensor([tokens['attention_mask']], dtype=torch.long).to(model.device)
        
        token_start = 0
        target_idx = []
        for i, token in enumerate(input_ids[0]):
            token_len = len(tokenizer.decode(token))
            if line[token_start : token_start + token_len].strip() == correct:
                target_idx.append(i)
            token_start += token_len
             
        for target_pos in target_idx:
            for vocab_id in range(VOCAB_SIZE):
                input_grad, l1_grad = saliency_map(model, input_ids, attention_ids, pos=target_pos, foil=vocab_id)
                print(f"{j}-{target_pos}-input ||| {input_grad}")
                print(f"{j}-{target_pos}-l1 ||| {l1_grad}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default='questions.json', type=str)
    parser.add_argument("--output", default="question-gradients", type=str)
    args = parser.parse_args()

    config = GPT2Config.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    tokenizer.pad_token = tokenizer.eos_token
    questions = json.load(open(args.questions, 'r'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open(f'{args.output}.html', 'w') as file:
        for i, question in enumerate(questions):
            if question['answer']:
                answer = question['answer']
                tokens = tokenizer(question['question'].replace('________', answer))

                input_ids = torch.tensor([tokens['input_ids']], dtype=torch.long).to(device)
                attention_ids = torch.tensor([tokens['attention_mask']], dtype=torch.long).to(device)
                pos = question['question'].split().index('________') - 1 # FIXME
                file.write('<!DOCTYPE html>')
                file.write('<html>')
                file.write('<body>')
                input_scores, l1_scores = saliency_map(model, input_ids, 
                                                    attention_ids, pos=pos)
                visualize(input_scores, tokenizer, input_ids, save_file=f'{args.output}/{i}_input.png', title=f'Input x Grad, Correct: {answer}')
                file.write(f"<p>Input x Grad, Correct: {answer}</p>")
                file.write(f"<img src='{args.output}/{i}_input.png'/><p></p>")
                visualize(l1_scores, tokenizer, input_ids, save_file=f'{args.output}/{i}_l1.png', title=f'L1 Norm, Correct: {answer}')
                file.write(f"<p>L1 Norm, Correct: {answer}</p>")
                file.write(f"<img src='{args.output}/{i}_l1.png'/><p></p>")
                for foil in question['options']:
                    if foil != answer:
                        input_scores, l1_scores = saliency_map(model, input_ids, 
                                                    attention_ids, pos=pos, foil=tokenizer(' '+foil)['input_ids'][0])
                        visualize(input_scores, tokenizer, input_ids, save_file=f'{args.output}/{i}-{foil}_input.png', title=f'Input x Grad, Correct: {answer}, Foil: {foil}' )
                        file.write(f"<p>Input x Grad, Correct: {answer}, Foil: {foil}</p>")
                        file.write(f"<img src='{args.output}/{i}-{foil}_input.png'/><p></p>")
                        visualize(l1_scores, tokenizer, input_ids, save_file=f'{args.output}/{i}-{foil}_l1.png', title=f'L1 Norm, Correct: {answer}, Foil: {foil}' )
                        file.write(f"<p>L1 Norm, Correct: {answer}, Foil: {foil}</p>")
                        file.write(f"<img src='{args.output}/{i}-{foil}_l1.png'/><p></p>")
        file.write('</body>')
        file.write('</html>')

def all():
    config = GPT2Config.from_pretrained("gpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    with open("data/wiki.test.Chinese_rand", "r") as file:
        data = file.readlines()

    one_target_vocab_saliency(model, data, tokenizer, "Chinese")

    # tokens = tokenizer("Adam lived in China so he speaks Chinese fluently.")
    # input_ids = torch.tensor([tokens['input_ids']], dtype=torch.long).to(device)
    # attention_ids = torch.tensor([tokens['attention_mask']], dtype=torch.long).to(device)

    # all_saliency_maps(model, tokenizer, input_ids, attention_ids)


if __name__ == "__main__":
    all()
