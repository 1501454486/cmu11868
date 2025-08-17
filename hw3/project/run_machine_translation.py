from functools import partial
import time
import os
import fire
import tqdm
import json
import random
import datasets
import numpy as np
from sacrebleu.metrics import BLEU
from transformers import AutoTokenizer
from tokenizers import ByteLevelBPETokenizer

import minitorch
from minitorch import DecoderLM
from minitorch.cuda_kernel_ops import CudaKernelOps
import pickle


def get_model_parameters(module, prefix=''):
    """
    Recursively collect all parameters from a module and its submodules.
    
    Args:
        module: The module to collect parameters from
        prefix: Prefix for parameter names
    
    Returns:
        dict: Dictionary mapping parameter names to Parameter objects
    """
    parameters = {}
    
    # Get direct parameters
    if hasattr(module, '_parameters'):
        for name, param in module._parameters.items():
            full_name = f"{prefix}.{name}" if prefix else name
            parameters[full_name] = param
    
    # Get parameters from submodules
    if hasattr(module, '_modules'):
        for name, submodule in module._modules.items():
            sub_prefix = f"{prefix}.{name}" if prefix else name
            sub_params = get_model_parameters(submodule, sub_prefix)
            parameters.update(sub_params)
    
    return parameters


def save_checkpoint(model, optimizer, epoch, validation_loss, bleu_score, workdir, is_best=False):
    """
    Save model checkpoint with training state.
    
    Args:
        model (DecoderLM): Model to save
        optimizer (Adam): Optimizer to save
        epoch (int): Current epoch number
        validation_loss (float): Current validation loss
        bleu_score (float): Current BLEU score
        workdir (str): Directory to save checkpoint
        is_best (bool): Whether this is the best model so far
    """
    # Save model parameters using our custom function
    model_state = {}
    model_params = get_model_parameters(model)
    for name, param in model_params.items():
        if param.value is not None:
            model_state[name] = param.value.to_numpy()  # Convert tensor to numpy for serialization
    
    # Save optimizer state
    optimizer_state = {
        'lr': optimizer.lr,
        'beta1': optimizer.beta1, 
        'beta2': optimizer.beta2,
        'eps': optimizer.eps,
        'states': {}
    }
    
    # Save Adam states - need to map parameter names to states
    param_id_to_name = {}
    model_params = get_model_parameters(model)
    for name, param in model_params.items():
        param_id_to_name[id(param)] = name
    
    for param_id, state in optimizer._states.items():
        if param_id in param_id_to_name:
            param_name = param_id_to_name[param_id]
            # Convert tensors to numpy arrays for serialization
            saved_state = {}
            for key, value in state.items():
                if hasattr(value, 'to_numpy'):
                    saved_state[key] = value.to_numpy()
                else:
                    saved_state[key] = value
            optimizer_state['states'][param_name] = saved_state
    
    checkpoint = {
        'epoch': epoch,
        'model_state': model_state,
        'optimizer_state': optimizer_state,
        'validation_loss': validation_loss,
        'bleu_score': bleu_score,
        'timestamp': time.time()
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(workdir, f'checkpoint_epoch_{epoch}.pkl')
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    # Save latest checkpoint
    latest_path = os.path.join(workdir, 'checkpoint_latest.pkl')
    with open(latest_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    
    # Save best checkpoint if this is the best model
    if is_best:
        best_path = os.path.join(workdir, 'checkpoint_best.pkl')
        with open(best_path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    print(f"Checkpoint saved: {checkpoint_path}")
    if is_best:
        print(f"Best checkpoint saved: {best_path}")


def load_checkpoint(checkpoint_path, model, optimizer):
    """
    Load model checkpoint and restore training state.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        model (DecoderLM): Model to load state into
        optimizer (Adam): Optimizer to load state into
        
    Returns:
        dict: Checkpoint data containing epoch, losses, etc.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        checkpoint = pickle.load(f)
    
    # Restore model parameters
    model_state = checkpoint['model_state']
    model_params = get_model_parameters(model)
    param_dict = {name: param for name, param in model_params.items()}
    
    for param_name, numpy_value in model_state.items():
        if param_name in param_dict:
            param = param_dict[param_name]
            # Convert numpy back to tensor with same backend
            restored_tensor = minitorch.tensor_from_numpy(numpy_value, backend=param.value.backend, requires_grad=True)
            param.update(restored_tensor)
    
    # Restore optimizer state
    optimizer_state = checkpoint['optimizer_state']
    optimizer.lr = optimizer_state['lr']
    optimizer.beta1 = optimizer_state['beta1']
    optimizer.beta2 = optimizer_state['beta2']
    optimizer.eps = optimizer_state['eps']
    
    # Restore Adam states - map parameter names back to IDs
    optimizer._states = {}
    model_params = get_model_parameters(model)
    name_to_param = {name: param for name, param in model_params.items()}
    
    for param_name, saved_state in optimizer_state['states'].items():
        if param_name in name_to_param:
            param = name_to_param[param_name]
            param_id = id(param)
            
            # Restore state tensors
            restored_state = {}
            for key, value in saved_state.items():
                if key in ['exp_avg', 'exp_avg_sq']:
                    # Convert numpy back to tensor
                    restored_state[key] = minitorch.tensor_from_numpy(
                        value, backend=param.value.backend, requires_grad=False
                    )
                else:
                    # Keep scalar values as-is (like 'step')
                    restored_state[key] = value
            
            optimizer._states[param_id] = restored_state
    
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Resuming from epoch {checkpoint['epoch']}")
    print(f"Previous validation loss: {checkpoint['validation_loss']:.4f}")
    print(f"Previous BLEU score: {checkpoint['bleu_score']:.2f}")
    
    return checkpoint


def cleanup_checkpoints(workdir, keep_last_n=3):
    """
    Clean up old checkpoint files, keeping only the most recent ones.
    
    Args:
        workdir (str): Directory containing checkpoints
        keep_last_n (int): Number of recent checkpoints to keep
    """
    checkpoint_files = []
    
    for filename in os.listdir(workdir):
        if filename.startswith('checkpoint_epoch_') and filename.endswith('.pkl'):
            epoch_num = int(filename.split('_')[2].split('.')[0])
            checkpoint_files.append((epoch_num, filename))
    
    # Sort by epoch number and remove old ones
    checkpoint_files.sort(key=lambda x: x[0])
    
    if len(checkpoint_files) > keep_last_n:
        files_to_remove = checkpoint_files[:-keep_last_n]
        for epoch_num, filename in files_to_remove:
            filepath = os.path.join(workdir, filename)
            os.remove(filepath)
            print(f"Removed old checkpoint: {filename}")


def get_dataset(dataset_name, model_max_length):
    """
    Load and preprocess IWSLT (de-en) dataset.
    
    Args:
        dataset_name (str): Name of the dataset to load
        model_max_length (int): Maximum sequence length for filtering examples

    Returns:
        tuple: (dataset, src_key, tgt_key) where:
            - dataset: Dictionary with 'train', 'validation', 'test' splits
            - src_key (str): Source language key ('de')
            - tgt_key (str): Target language key ('en')
    """
    dataset = {
        split: datasets.load_dataset(dataset_name, split=split)['translation']
        for split in ['train', 'validation', 'test']
    }
    src_key, tgt_key = 'de', 'en'

    dataset = {
        split: [
            example for example in dataset[split]
            if len(example[src_key].split()) + len(
                example[tgt_key].split()) < model_max_length
        ] for split in dataset.keys()
    }

    dataset['test'] = dataset['test'][:100]  # 6750

    print(json.dumps(
        {'data_size': {split: len(dataset[split]) for split in dataset.keys()}},
        indent=4))

    return dataset, src_key, tgt_key


def get_tokenizer(examples, vocab_size, src_key, tgt_key, workdir):
    """
    Train and save a ByteLevelBPETokenizer on the provided dataset.
    
    Args:
        examples (list): Dataset examples for tokenizer training
        vocab_size (int): Desired vocabulary size
        src_key (str): Source language key in examples
        tgt_key (str): Target language key in examples
        workdir (str): Directory to save tokenizer files

    Returns:
        AutoTokenizer: Trained tokenizer with special tokens
                      (e.g., "<eos_de>", "<eos_en>", "<pad>")
    """
    tokenizer = ByteLevelBPETokenizer()

    # Customized training
    tokenizer.train_from_iterator(
        [[example[src_key], example[tgt_key]] for example in examples],
        vocab_size=vocab_size,
        special_tokens=[f'<eos_{src_key}>', f'<eos_{tgt_key}>', '<pad>'])

    tokenizer.save(f'{workdir}/tokenizer.json')
    json.dump({'model_type': 'gpt2'}, open(f'{workdir}/config.json', 'w'))

    tokenizer = AutoTokenizer.from_pretrained(
        workdir,
        eos_token=None,
        bos_token=None,
        pad_token=None,
        unk_token=None)

    return tokenizer


def collate_batch(
        examples, src_key, tgt_key, tokenizer, model_max_length, backend):
    """
    Prepare a batch of examples for model training or evaluation.
    
    Args:
        examples (list): List of examples to process
        src_key (str): Key for source texts in examples
        tgt_key (str): Key for target texts in examples
        tokenizer (AutoTokenizer): Tokenizer for encoding texts
        model_max_length (int): Maximum sequence length
        backend (TensorBackend): Backend for minitorch tensors

    Returns:
        dict: Dictionary containing:
            - input_ids: Tokenized input sequences of shape (batch_size, model_max_length-1)
            - labels: Target sequences of shape (batch_size, model_max_length-1)
            - label_token_weights: Weight mask for loss computation of shape (batch_size, model_max_length-1)
            
    Note:
        input_ids format: <de_tokens> + <de_eos> + <en_tokens> + <en_eos> + <pad>
        labels: Next tokens to predict (shifted by 1)
        label_token_weights: 0 for source tokens, 1 for target tokens
    """
    token_ids, tgt_token_mask = [], []
    max_length = model_max_length
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

    # TODO: make examples in a 1d list, provide shape to initialize minitorch.Tensor
    token_ids = np.array(token_ids)
    tgt_token_mask = np.array(tgt_token_mask)

    input_ids = token_ids[:, :-1]
    labels    = token_ids[:, 1:]
    label_token_weights = tgt_token_mask[:, 1:]

    input_ids = minitorch.tensor_from_numpy(input_ids, backend=backend)
    labels    = minitorch.tensor_from_numpy(labels, backend=backend)
    label_token_weights = minitorch.tensor_from_numpy(label_token_weights, backend=backend)
    
    # input_ids = token_ids[:, :-1].tolist()
    # labels    = token_ids[:, 1:].tolist()
    # label_token_weights = tgt_token_mask[:, 1:].tolist()

    # input_ids = minitorch.tensor(input_ids, backend=backend)
    # labels    = minitorch.tensor(labels, backend=backend)
    # label_token_weights = minitorch.tensor(label_token_weights, backend=backend)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'label_token_weights': label_token_weights
    }


def loss_fn(batch, model):
    """
    Compute MLE loss for a batch of examples.
    
    Args:
        batch (dict): Batch data containing 'input_ids', 'labels', 'label_token_weights'
        model (DecoderLM): Language model for prediction

    Returns:
        Tensor: Average loss across all target tokens
    """

    idx = batch['input_ids']
    idx.requires_grad_(True)
    # print("getting into loss_fn")
    logits = model(idx=idx)
    # print("finish prediction")
    bs, l, c = logits.shape
    logits = logits.view(bs * l, c)
    targets = batch['labels'].view(bs * l)
    label_token_weights = batch['label_token_weights'].view(bs * l)

    targets.requires_grad_(True)
    # print("start calculating loss")
    # import pdb
    # pdb.set_trace()
    loss = minitorch.nn.softmax_loss(
        logits=logits,
        target=targets
    )

    return ((loss * label_token_weights).sum() / label_token_weights.sum())


def train(model, optimizer, examples, n_samples, collate_fn, batch_size, desc):
    """
    Train the model on provided examples.
    
    Args:
        model (DecoderLM): Model to train
        optimizer (Adam): Optimizer for parameter updates
        examples (list): Training dataset examples
        n_samples (int): Number of random samples to use
        collate_fn (callable): Function to collate examples into batches
        batch_size (int): Number of examples per batch
        desc (str): Description for progress bar
    """
    model.train()
    random.shuffle(examples)
    examples = examples[:n_samples]

    for i in (prog_bar := tqdm.trange(
            0, len(examples), batch_size, desc=f'Training ({desc})')):
        batch = collate_fn(examples=examples[i:i + batch_size])

        t0 = time.time()
        optimizer.zero_grad()
        loss = loss_fn(batch=batch, model=model)
        t1 = time.time()

        loss.backward()
        t2 = time.time()

        optimizer.step()
        t3 = time.time()

        print(f"Forward: {t1 - t0}")
        print(f"Backward: {t2 - t1}")
        print(f"Opt.step: {t3 - t2}")

        batch_time = time.time() - t0
        prog_bar.set_postfix(
            tokens_per_sec=np.prod(batch['input_ids'].shape) / batch_time,
            loss=loss.item(),
            lr=optimizer.lr)


def evaluate_loss(model, examples, batch_size, collate_fn, desc):
    """
    Evaluate model loss on provided examples.
    
    Args:
        model (DecoderLM): Model to evaluate
        examples (list): Evaluation dataset examples
        batch_size (int): Number of examples per batch
        collate_fn (callable): Function to collate examples into batches
        desc (str): Description for progress bar

    Returns:
        float: Average loss across all batches
    """
    model.eval()
    losses = []

    for i in (prog_bar := tqdm.trange(
        0, len(examples), batch_size, desc=f'Evaluating ({desc})')):
        batch = collate_fn(examples=examples[i:i + batch_size])
        loss = loss_fn(batch=batch, model=model)

        losses.append(loss.item())
        prog_bar.set_postfix(loss=loss.item())

    return np.mean(losses)


def generate(
    model,
    examples,
    src_key,
    tgt_key,
    tokenizer,
    model_max_length,
    backend,
    desc
):
    """
    Generate target sequences for source sequences using argmax decoding.
    
    Args:
        model (DecoderLM): Model for generation
        examples (list): Dataset examples containing source sequences
        src_key (str): Key for source texts in examples
        tgt_key (str): Key for target texts in examples
        tokenizer (AutoTokenizer): Tokenizer for encoding/decoding
        model_max_length (int): Maximum sequence length
        backend (TensorBackend): Backend for minitorch tensors
        desc (str): Description for progress bar

    Returns:
        list: Generated target sequences
    """

    model.eval()
    gen_sents = []
    for example in tqdm.tqdm(examples, desc=f'Generating {desc}'):
        # Run generation for every single example

        token_ids = tokenizer(f'{example[src_key]}<eos_{src_key}>')['input_ids']
        len_src = len(token_ids)

        while len(token_ids) <= model_max_length:
            # BEGIN ASSIGN3_4
            # TODO
            # run the model with current token_ids, and predict the next token (gen_id)
            # hint: obtain the logits of next token, and take the argmax.
            gen_id = 0
            
            input_tensor = minitorch.tensor([token_ids], backend = backend)
            logits = model.forward(input_tensor)
            
            logits_np = logits.to_numpy()
            
            # Shape: (vocab_size, )
            last_token_logits_np = logits_np[0, -1, :]
            
            gen_id = np.argmax(last_token_logits_np)
            
            # END ASSIGN3_4

            if gen_id == tokenizer.vocab[f'<eos_{tgt_key}>']:
                break
            else:
                token_ids.append(gen_id)

        gen_sents.append(tokenizer.decode(token_ids[len_src:]))

    return gen_sents


def evaluate_bleu(examples, gen_sents, tgt_key):
    """
    Evaluate BLEU score for generated sentences against target sentences.
    
    Args:
        examples (list): Dataset examples containing target sentences
        gen_sents (list): Generated sentences to evaluate
        tgt_key (str): Key for target texts in examples

    Returns:
        dict: Dictionary containing BLEU score
    """
    return {
        'bleu': BLEU().corpus_score(
            hypotheses=gen_sents,
            references=[[example[tgt_key] for example in examples]]).score
    }


def main(
    dataset_name='bbaaaa/iwslt14-de-en-preprocess',
    model_max_length=40,
    n_epochs=20,
    batch_size=128,
    learning_rate=0.01,
    samples_per_epoch=20000,
    n_vocab=10000,
    n_embd=256,
    seed=11111,
    resume=True,
    checkpoint_path="/home/lixinyuan/cmu11868hw/hw3/workdir_vocab10000_lr0.02_embd256/checkpoint_latest.pkl"
):
    """
    Train and evaluate a decoder-only transformer language model.
    
    Args:
        dataset_name (str): Name of the dataset to use, default 'bbaaaa/iwslt14-de-en-preprocess'
        model_max_length (int): Maximum sequence length, default 40
        n_epochs (int): Number of training epochs, default 20
        batch_size (int): Number of examples per batch, default 128
        learning_rate (float): Learning rate for optimizer, default 0.02
        samples_per_epoch (int): Training samples per epoch, default 20000
        n_vocab (int): Vocabulary size for tokenizer, default 10000
        n_embd (int): Embedding dimension, default 256
        seed (int): Random seed, default 11111
        resume (bool): Whether to resume training from checkpoint, default False
        checkpoint_path (str): Specific checkpoint path to load, default None (loads latest)
    """

    np.random.seed(seed)
    random.seed(seed)

    workdir = f'./workdir_vocab{n_vocab}_lr{learning_rate}_embd{n_embd}'
    os.makedirs(workdir, exist_ok=True)

    backend = minitorch.TensorBackend(CudaKernelOps)

    config = {
        'n_vocab': n_vocab,  # vocab_size
        'n_embd': n_embd,  # n_embed
        'n_head': 8,  # n_head
        'n_positions': model_max_length,  # n_ctx == n_positions
        # 'n_layer'     : 4,    # n_layer
        'p_dropout': 0.1,  # x_pdrop
        'ln_eps': 1e-5,  # layer_norm_epsilon
        'backend': backend
    }

    model = DecoderLM(**config)
    optimizer = minitorch.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize training state
    start_epoch = 0
    best_validation_loss = float('inf')
    best_bleu_score = 0.0
    
    # Handle resume functionality
    if resume:
        if checkpoint_path is None:
            # Try to load latest checkpoint
            latest_checkpoint = os.path.join(workdir, 'checkpoint_latest.pkl')
            if os.path.exists(latest_checkpoint):
                checkpoint_path = latest_checkpoint
            else:
                print("No checkpoint found to resume from. Starting from scratch.")
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint_data = load_checkpoint(checkpoint_path, model, optimizer)
                start_epoch = checkpoint_data['epoch'] + 1
                best_validation_loss = checkpoint_data.get('validation_loss', float('inf'))
                best_bleu_score = checkpoint_data.get('bleu_score', 0.0)
                print(f"Successfully resumed training from epoch {start_epoch}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting from scratch.")
                start_epoch = 0

    dataset, src_key, tgt_key = get_dataset(
        dataset_name=dataset_name, model_max_length=model_max_length)

    tokenizer = get_tokenizer(
        examples=dataset['train'],
        vocab_size=config['n_vocab'],
        src_key=src_key,
        tgt_key=tgt_key,
        workdir=workdir)

    collate_fn = partial(
        collate_batch,
        src_key=src_key,
        tgt_key=tgt_key,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
        backend=backend)

    for epoch_idx in range(start_epoch, n_epochs):
        desc = f'epoch {epoch_idx} / {n_epochs}'

        train(
            model=model,
            optimizer=optimizer,
            examples=dataset['train'],
            n_samples=samples_per_epoch,
            batch_size=batch_size,
            collate_fn=collate_fn,
            desc=desc)

        validation_loss = evaluate_loss(
            model=model,
            examples=dataset['validation'],
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
            backend=backend,
            desc=desc)

        gen_examples = []
        for example, gen_sent in zip(dataset['test'], gen_sents):
            gen_examples.append({'example': example, 'gen': gen_sent})
        json.dump(gen_examples, open(
            f'{workdir}/gen_epoch{epoch_idx}.json', 'w'), indent=4)

        eval_scores = evaluate_bleu(
            examples=dataset['test'], gen_sents=gen_sents, tgt_key=tgt_key)
        current_bleu = eval_scores['bleu']
        print(f'Epoch {epoch_idx}: {eval_scores}')

        json.dump(
            {'validation_loss': float(validation_loss), **eval_scores},
            open(f'{workdir}/eval_results_epoch{epoch_idx}.json', 'w'))

        # Determine if this is the best model
        is_best_loss = validation_loss < best_validation_loss
        is_best_bleu = current_bleu > best_bleu_score
        is_best = is_best_loss or is_best_bleu
        
        # Update best scores
        if is_best_loss:
            best_validation_loss = validation_loss
            print(f"New best validation loss: {validation_loss:.4f}")
        if is_best_bleu:
            best_bleu_score = current_bleu
            print(f"New best BLEU score: {current_bleu:.2f}")

        # Save checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch_idx,
            validation_loss=validation_loss,
            bleu_score=current_bleu,
            workdir=workdir,
            is_best=is_best
        )
        
        # Clean up old checkpoints (keep last 3)
        cleanup_checkpoints(workdir, keep_last_n=3)


if __name__ == '__main__':
    fire.Fire(main)
