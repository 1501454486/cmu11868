import os
import sys
from pathlib import Path
from sacrebleu.metrics import BLEU
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoTokenizer
import torch
import tqdm
import time
import numpy as np

cousin_dir = Path(__file__).resolve().parents[1]


def get_tokenizer(examples, vocab_size, src_key, tgt_key, workdir):
    tokenizer = ByteLevelBPETokenizer()

    # Customized training
    tokenizer.train_from_iterator(
        [[example[src_key], example[tgt_key]] for example in examples],
        vocab_size=vocab_size,
        special_tokens=[f'<eos_{src_key}>', f'<eos_{tgt_key}>', '<pad>'])

    tokenizer.save(f'{workdir}/tokenizer.json')
    assert os.path.exists(f'{workdir}/config.json')
    tokenizer = AutoTokenizer.from_pretrained(
        workdir,
        eos_token=None,
        bos_token=None,
        pad_token=None,
        unk_token=None)

    return tokenizer


def evaluate_bleu(examples, gen_sents, tgt_key):
    return {
        'bleu': BLEU().corpus_score(
            hypotheses=gen_sents,
            references=[[example[tgt_key] for example in examples]]).score
    }

def collate_batch(
        examples, src_key, tgt_key, tokenizer, model_max_length, device):
    token_ids, tgt_token_mask = [], []
    max_length = model_max_length + 1
    pad_token_id = tokenizer.vocab['<pad>']
    for example in examples:
        token_ids_src = tokenizer(
            f'{example[src_key]}<eos_{src_key}>')['input_ids']
        token_ids_tgt = tokenizer(
            f'{example[tgt_key]}<eos_{tgt_key}>')['input_ids']

        example_token_ids = token_ids_src + token_ids_tgt
        example_tgt_token_mask = (
                [0] * len(token_ids_src) + [1] * len(token_ids_tgt))
        example_token_ids = example_token_ids[:max_length]
        example_tgt_token_mask = example_tgt_token_mask[:max_length]
        pad_ids = [pad_token_id] * (max_length - len(example_token_ids))

        token_ids.append(example_token_ids + pad_ids)
        tgt_token_mask.append(example_tgt_token_mask + [0] * len(pad_ids))

    token_ids = torch.tensor(token_ids, device=device)
    tgt_token_mask = torch.tensor(tgt_token_mask, device=device)
    
    return {
        'input_ids': token_ids[:, :-1],
        'labels': token_ids[:, 1:],
        'label_token_weights': tgt_token_mask[:, 1:]
    }

def loss_fn(batch, model):
    logits = model(input_ids=batch['input_ids']).logits

    loss = torch.nn.functional.cross_entropy(
        input=logits.reshape((-1, logits.shape[-1])),
        target=batch['labels'].reshape(-1),
        reduction='none')

    return (torch.sum(loss * batch['label_token_weights'].reshape(-1)) /
            torch.sum(batch['label_token_weights']))


def evaluate_loss(model, examples, batch_size, collate_fn, desc):
    model.eval()
    losses = []

    for batch in (prog_bar := tqdm.tqdm(examples, desc=f'Evaluating ({desc})')):
        with torch.no_grad():
            loss = loss_fn(batch=batch, model=model)

        losses.append(loss.item())
        prog_bar.set_postfix(loss=loss.item())

    return np.mean(losses)


def generate(model, examples, src_key, tgt_key, tokenizer, model_max_length, device, desc):
    model.eval()

    gen_sents = []
    for example in tqdm.tqdm(examples, desc=f'Generating {desc}'):
        token_ids = tokenizer(f'{example[src_key]}<eos_{src_key}>')['input_ids']
        len_src = len(token_ids)

        while len(token_ids) <= model_max_length:
            with torch.no_grad():
                logits = model(
                    input_ids=torch.tensor([token_ids], device=device)
                ).logits[0, -1]
                gen_id = torch.argmax(logits).item()

            if gen_id == tokenizer.vocab[f'<eos_{tgt_key}>']:
                break
            else:
                token_ids.append(gen_id)

        gen_sents.append(tokenizer.decode(token_ids[len_src:]))

    return gen_sents

def train(model, optimizer, examples, batch_size, collate_fn, desc, rank=0, average_gradients_fn=None):
    model.train()
    
    tokens_per_sec = []
    tokens_num = []
    for i, batch in enumerate(prog_bar := tqdm.tqdm(examples, desc=f'Training ({desc})')):
        t0 = time.time()
        optimizer.zero_grad()
        logits = model(input_ids=batch['input_ids']).logits

        loss = torch.nn.functional.cross_entropy(
            input=logits.reshape((-1, logits.shape[-1])),
            target=batch['labels'].reshape(-1),
            reduction='none')
        loss = (torch.sum(loss * batch['label_token_weights'].reshape(-1)) /
            torch.sum(batch['label_token_weights']))
        loss.backward()
        ''' Call the `average_gradients_fn` function to reduce and broadcast the gradients in Data Parallel
            Just few lines of code. Think simply.
        '''
        # BEGIN SOLUTION
        average_gradients_fn(model)
        # END SOLUTION
        optimizer.step()
        batch_time = time.time() - t0
        tokens = np.prod(batch['input_ids'].shape)
        tokens_per_sec.append(tokens / batch_time)
        tokens_num.append(tokens)
        prog_bar.set_postfix(
            tokens_per_sec=tokens / batch_time,
            loss=loss.item())
    return np.mean(tokens_per_sec), tokens_num


def save_grad_weights(model, rank):
    gradients = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.data.detach().cpu()
    torch.save(gradients, f'{cousin_dir}/tests/model{rank}_gradients.pth')