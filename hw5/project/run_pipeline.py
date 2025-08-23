import sys
from pathlib import Path

cousin_dir = Path(__file__).resolve().parents[1] 
sys.path.append(str(cousin_dir))

from functools import partial
import time
import os
import argparse
import math
import tqdm
import json
import random
import datasets
import numpy as np
from sacrebleu.metrics import BLEU
from transformers import AutoConfig, AutoTokenizer, GPT2LMHeadModel
from tokenizers import ByteLevelBPETokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from pipeline.model_parallel import GPT2LMHeadModelParallel
from utils import get_tokenizer, evaluate_bleu, collate_batch, evaluate_loss, generate, train

PYTEST = False


def run_pp(
    dataset_name='bbaaaa/iwslt14-de-en-preprocess',
    model_max_length=128,
    n_epochs=2,
    batch_size=32,
    n_chunk = 4, # the number of microbatches for pipeline parallelism
    learning_rate=1e-4,
    device='cuda',
    model_parallel_mode=None):
    workdir = f'./workdir'
    os.makedirs(workdir, exist_ok=True)

    config = AutoConfig.from_pretrained('gpt2')
    config.save_pretrained(workdir)

    device_count = torch.cuda.device_count()
    first_device = "cuda:0" if device_count > 0 else "cpu"


    split_size = math.ceil(batch_size/n_chunk)
    
    model = GPT2LMHeadModelParallel(config=config)
    if model_parallel_mode == 'model_parallel':
        model.parallelize()
    elif model_parallel_mode == 'pipeline_parallel':
        model.parallelize()
        model._prepare_pipeline_parallel(split_size=split_size)
    else:
        model = model.to(first_device) # single device
    
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    dataset = {
        split: datasets.load_dataset(dataset_name, split=split)['translation']
        for split in ['train', 'validation', 'test']
    }
    src_key, tgt_key = 'de', 'en'

    ### MAKE SMALLER
    dataset['train'] = dataset['train'][:5000]
    dataset['validation'] = dataset['validation'][:1000]
    dataset['test'] = dataset['test'][:100]
    ###

    tokenizer = get_tokenizer(
        examples=dataset['train'],
        vocab_size=config.vocab_size,
        src_key=src_key,
        tgt_key=tgt_key,
        workdir=workdir)

    collate_fn = partial(
        collate_batch,
        src_key=src_key,
        tgt_key=tgt_key,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        device=first_device)

    train_loader = DataLoader(dataset["train"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    val_loader = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(dataset["test"], batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    total_time = []
    total_tokens_per_sec = []

    for epoch_idx in range(n_epochs):
        desc = f'epoch {epoch_idx} / {n_epochs}'

        start = time.time()
        _, token_nums = train(
            model=model,
            optimizer=optimizer,
            examples=train_loader,
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc)
        end = time.time()
        avg_tokens_per_sec = sum(token_nums) / (end - start)

        if not PYTEST:
            training_time = end - start
            print(f'Epoch {epoch_idx}: Training Time = {training_time}, Tokens_per_sec = {avg_tokens_per_sec}')
            total_time.append(training_time)
            total_tokens_per_sec.append(avg_tokens_per_sec)

            validation_loss = evaluate_loss(
                model=model,
                examples=val_loader,
                batch_size=batch_size,
                collate_fn=collate_fn,
                desc=desc)

            print(f'Epoch {epoch_idx}: Validation Loss = {validation_loss}')

            gen_sents = generate(
                model=model,
                examples=dataset['test'],
                src_key=src_key,
                tgt_key=tgt_key,
                tokenizer=tokenizer,
                model_max_length=model_max_length,
                device=first_device,
                desc=desc)

            gen_examples = []
            for example, gen_sent in zip(dataset['test'], gen_sents):
                gen_examples.append({'example': example, 'gen': gen_sent})
            json.dump(gen_examples, open(
                f'{workdir}/gen_epoch{epoch_idx}.json', 'w'), indent=4)

            eval_scores = evaluate_bleu(
                examples=dataset['test'], gen_sents=gen_sents, tgt_key=tgt_key)
            print(f'Epoch {epoch_idx}: {eval_scores}')

            json.dump(
                {'validation_loss': validation_loss, **eval_scores},
                open(f'{workdir}/eval_results_epoch{epoch_idx}.json', 'w'))
        else:
            break
    if not PYTEST:
        # You only get the average training time and tokens_per_second per device
        # To compute the throughput, you need to sum up the tokens_per_sec across all the devices based on epochs
        print(f'Training time: avg:{np.mean(total_time)}, std:{np.std(total_time)}, \
        tokens_per_second: avg: {np.mean(total_tokens_per_sec)}, std:{np.std(total_tokens_per_sec)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pytest', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='bbaaaa/iwslt14-de-en-preprocess')
    parser.add_argument('--model_max_length', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=2)
    parser.add_argument('--n_chunk', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--model_parallel_mode', type=str, default=None)

    args = parser.parse_args()

    PYTEST = args.pytest

    run_pp(
        dataset_name=args.dataset,
        model_max_length=args.model_max_length,
        n_epochs=args.n_epochs,
        n_chunk=args.n_chunk,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_parallel_mode=args.model_parallel_mode
    )