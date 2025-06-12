# coding=utf-8

import os
import os.path
import argparse
from tqdm import tqdm
from itertools import count
from tokenizers import Tokenizer
from statistics import mean
import json
import math
import copy
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lamb import Lamb
from model_extra import Bert
from utils import cosine_schedule_with_warmup_cooldown, is_main_process, seed_everything
from dataset import MaskedDataset, CausalDataset, ValidationDataset
from model_logging import ModelLogger


if int(os.environ["SLURM_PROCID"]) == 0:
    import wandb


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_path", default="../data/train_100M_tokenized.bin", type=Path, help="Path to the training data.")
    parser.add_argument("--valid_path", default="../data/dev_100M_tokenized.bin", type=Path, help="Path to the validation data.")
    parser.add_argument("--name", default="hybrid_100M", type=str, help="Name of the run.")
    parser.add_argument("--wandb_project", default="YOUR_WANDB_PROJECT_NAME", type=str, help="Name of the WandB project to log into.")
    parser.add_argument("--wandb_entity", default="YOUR_WANDB_ENTITY", type=str, help="The entity to log to on WandB (typically your wandb username).")
    parser.add_argument("--config_file", default="../configs/base.json", type=Path, help="The BERT model config")
    parser.add_argument("--tokenizer_path", default="../tokenizers/tokenizer_100M.json", type=Path, help="Path to the tokenizer.")
    parser.add_argument("--output_dir", default="../model_checkpoints", type=Path, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--checkpoint_filename", default=None, type=Path, help="The checkpoint filename to resume training.")
    parser.add_argument("--optimizer", default="lamb", type=str, help="The optimizer to use.")
    parser.add_argument("--hybrid_numerator", default=15, type=int, help="The numerator of the hybrid ratio.")
    parser.add_argument("--hybrid_denominator", default=16, type=int, help="The denominator of the hybrid ratio (the number of GPUs should be divisible by this number).")
    parser.add_argument("--seq_length", default=128, type=int, help="Sequence length for training.")
    parser.add_argument("--local_batch_size", default=256, type=int, help="Batch size for training per GPU.")
    parser.add_argument("--global_batch_size", default=256, type=int, help="Total batch size for training per GPUs and per grad accumulation step.")
    parser.add_argument("--batch_reduction", default=4, type=int, help="The initial batch size reduction factor.")
    parser.add_argument("--learning_rate", default=1e-2, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--max_steps", default=31_250 // 2, type=int, help="Total number of training steps to perform.")
    parser.add_argument("--ema_decay", default=0.999, type=float, help="Exponential moving average decay.")
    parser.add_argument("--validate_every", default=1_000, type=int, help="Run validation after every X training shards.")
    parser.add_argument("--validation_steps", default=1, type=int, help="Number of validation steps.")
    parser.add_argument("--log_stats_every", default=100, type=int, help="Log stats every X steps.")
    parser.add_argument("--warmup_proportion", default=0.016, type=float, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument("--cooldown_proportion", default=0.016, type=float, help="Proportion of training to perform linear learning rate cooldown for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--save_every', type=int, default=1_000, help="save every X steps")
    parser.add_argument("--mask_p_start", default=0.3, type=float, help="Initial masking probability.")
    parser.add_argument("--mask_p_end", default=0.15, type=float, help="Final masking probability.")
    parser.add_argument("--mask_random_p", default=0.1, type=float, help="Probability of replacing the masked token with a random token.")
    parser.add_argument("--mask_keep_p", default=0.1, type=float, help="Probability of keeping the masked token.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--optimizer_eps", default=1e-8, type=float, help="Optimizer epsilon.")
    parser.add_argument("--optimizer_beta1", default=0.9, type=float, help="Optimizer beta1.")
    parser.add_argument("--optimizer_beta2", default=0.98, type=float, help="Optimizer beta2.")
    parser.add_argument("--max_gradient", default=2.0, type=float, help="Max value for gradient clipping.")
    parser.add_argument('--mixed_precision', default=True, action=argparse.BooleanOptionalAction, help="Mixed precision training.")
    parser.add_argument('--n_special_tokens', default=16, type=int, help="Number of special tokens.")
    parser.add_argument('--z_loss_weight', default=1e-4, type=float, help="Weight for the z loss.")
    parser.add_argument('--token_weighted_loss', default=False, action=argparse.BooleanOptionalAction, help="Use token weighted loss.")
    args = parser.parse_args()

    args.name = "_".join([args.name, str(args.hybrid_numerator), str(args.hybrid_denominator)])
    args.output_path = (args.output_dir / args.name).with_suffix(".bin")

    return args


def setup_training(args, tokenizer):
    assert torch.cuda.is_available()
    seed_everything(args.seed)

    args.device = torch.device("cuda")

    print(f"Training for {args.max_steps:,} steps")
    print(f"In total, the model will be trained on 'steps'({args.max_steps:,}) x 'batch_size'({args.global_batch_size:,}) x 'seq_len'({args.seq_length:,}) = {args.max_steps * args.global_batch_size * args.seq_length:,} subword instances")

    args.vocab_size = tokenizer.get_vocab_size()

    wandb.init(
        name=args.name,
        project=args.wandb_project,
        entity=args.wandb_entity
    )


def load_config(args):
    with args.config_file.open("r") as f:
        config = json.load(f)
    for k, v in config.items():
        setattr(args, k, v)
    return args


def prepare_model_and_optimizer(args):
    args = load_config(args)
    model = Bert(args)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    wandb.config.update(args)
    wandb.config.update({"n_params": n_params})
    print(model)
    print(f"NUMBER OF PARAMETERS: {n_params}\n", flush=True)

    model.to(args.device)

    no_decay = ['bias', 'layer_norm']
    decay_params = [(n, p) for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    no_decay_params = [(n, p) for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer_grouped_parameters = [
        {'params': [p for _, p in decay_params], 'weight_decay': args.weight_decay},
        {'params': [p for _, p in no_decay_params], 'weight_decay': 0.0}
    ]

    print("Parameters without weight decay:")
    for n, _ in no_decay_params:
        print(n)
    print()
    print("Parameters with weight decay:")
    for n, _ in decay_params:
        print(n)
    print(flush=True)

    if args.optimizer == "adam" or args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )
    elif args.optimizer == "lamb":
        optimizer = Lamb(
            optimizer_grouped_parameters,
            args.learning_rate,
            betas=(args.optimizer_beta1, args.optimizer_beta2),
            eps=args.optimizer_eps,
        )

    scheduler = cosine_schedule_with_warmup_cooldown(
        optimizer,
        int(args.max_steps * args.warmup_proportion),
        int(args.max_steps * args.cooldown_proportion),
        args.max_steps,
        0.1
    )

    ema_model: nn.Module = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.requires_grad = False

    global_step, epoch = 0, 0
    if args.checkpoint_filename is not None:
        state_dict = torch.load(args.checkpoint_filename, map_location="cpu")
        model.load_state_dict(state_dict["model"])
        ema_model.load_state_dict(state_dict["ema_model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        scheduler.load_state_dict(state_dict["scheduler"])
        global_step = state_dict["global_step"]
        epoch = state_dict["epoch"]

    return model, ema_model, optimizer, scheduler, global_step, epoch


def get_batch(dataloader, device, global_step):
    dataloader._dataset.set_global_step(global_step)
    batch = next(dataloader)
    input_ids, target_ids, attention_mask, mask_p = [t.pin_memory().to(device, non_blocking=True) for t in batch]
    input_ids, target_ids = input_ids.t(), target_ids.t()
    mask_p = mask_p.mean()

    return input_ids, attention_mask, target_ids, mask_p


@torch.no_grad()
def validation_epoch(model, valid_dataloader, masked_epoch, causal_epoch, args, commit=False):
    model = model.eval()

    losses, accuracies = [], []
    valid_dataloader = iter(valid_dataloader)
    input_ids, attention_mask, target_ids, _ = get_batch(valid_dataloader, args.device, 0)
    for local_step in tqdm(range(args.validation_steps), desc="Valid iteration", disable=not is_main_process()):

        with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
            loss, accuracy, _, num_tokens = model(input_ids, attention_mask, target_ids)

        if local_step < args.validation_steps - 1:
            input_ids, attention_mask, target_ids, _ = get_batch(valid_dataloader, args.device, 0)

        losses.append(loss.item())
        accuracies.append(accuracy.item())

    if is_main_process():
        wandb.log(
            {
                "masked_epoch": masked_epoch,
                "causal_epoch": causal_epoch,
                "validation/loss": mean(losses),
                "validation/accuracy": mean(accuracies) * 100.0,
                "validation/perplexity": math.exp(mean(losses))
            },
            commit=commit
        )


def save(model, ema_model, optimizer, scheduler, global_step, masked_epoch, causal_epoch, args):
    if is_main_process():
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model itself
        torch.save(model_to_save.state_dict(), args.output_path)
        torch.save(ema_model.state_dict(), args.output_path.replace(".bin", "_ema.bin"))
        torch.save(
            {
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "global_step": global_step,
                "masked_epoch": masked_epoch,
                "causal_epoch": causal_epoch
            },
            args.output_path.replace(".bin", "_state_dict.bin")
        )


def load_dataset(args, tokenizer, epoch, global_step, train_dataloader, mode="masked"):
    train_seed = args.seed + epoch

    if (global_step + 1) / args.max_steps >= 0.9:
        seq_length = args.seq_length * 4
        global_batch_size = args.global_batch_size // 4
    elif (global_step + 1) / args.max_steps >= 0.7:
        seq_length = args.seq_length * 2
        global_batch_size = args.global_batch_size // 2
    else:
        seq_length = args.seq_length
        global_batch_size = args.global_batch_size

    if train_dataloader is None or train_dataloader.dataset.seq_length != seq_length:
        if mode == "masked":
            ratio = args.hybrid_numerator / args.hybrid_denominator
            train_data = MaskedDataset(args.train_path, tokenizer, args, seq_length, None, None)
        else:
            ratio = 1 - (args.hybrid_numerator / args.hybrid_denominator)
            train_data = CausalDataset(args.train_path, tokenizer, args, seq_length, None, None)

        train_data.show_random_item(tokenizer)
    else:
        train_data = train_dataloader.dataset

    # linear batch size scaling
    args.current_global_batch_size = int(global_batch_size / args.batch_reduction * (1 - global_step / args.max_steps) + global_batch_size * (global_step / args.max_steps) + 0.5)
    total_local_batch_size = int(args.current_global_batch_size * ratio + 0.5)
    if total_local_batch_size:
        total_local_batch_size = 1
        print(f"WARNING: The current {mode} ratio gives a batch size smaller than 1, the batch size is now set to 1.")

    train_dataloader = DataLoader(
        train_data,
        shuffle=True,
        batch_size=total_local_batch_size,
        num_workers=0,  # non-zero num_workers causes segmenation fault
        generator=torch.Generator().manual_seed(train_seed),
        drop_last=True,
        pin_memory=True,
    )


def init_datasets(args, tokenizer):
    train_seed = args.seed

    seq_length = args.seq_length
    global_batch_size = args.global_batch_size
    args.ratio = args.hybrid_numerator / args.hybrid_denominator

    # linear batch size scaling
    args.current_global_batch_size = int(global_batch_size / args.batch_reduction + 0.5)

    if args.ratio != 0:
        masked_train_data = MaskedDataset(args.train_path, tokenizer, args, seq_length, None, None)
        masked_train_data.show_random_item(tokenizer)

        total_masked_local_batch_size = int(args.current_global_batch_size * args.ratio + 0.5)
        if total_masked_local_batch_size == 0:
            total_masked_local_batch_size = 1
            print("WARNING: The current masked ratio gives a batch size smaller than 1, the batch size is now set to 1.")

        masked_train_dataloader = DataLoader(
            masked_train_data,
            shuffle=True,
            batch_size=total_masked_local_batch_size,
            num_workers=0,  # non-zero num_workers causes segmenation fault
            generator=torch.Generator().manual_seed(train_seed),
            drop_last=True,
            pin_memory=True,
        )
    else:
        masked_train_dataloader = None

    if args.ratio != 1:
        causal_train_data = CausalDataset(args.train_path, tokenizer, args, seq_length, None, None)
        causal_train_data.show_random_item(tokenizer)

        total_causal_local_batch_size = int(args.current_global_batch_size * (1 - args.ratio) + 0.5)
        if total_causal_local_batch_size == 0:
            total_causal_local_batch_size = 1
            print("WARNING: The current causal ratio gives a batch size smaller than 1, the batch size is now set to 1.")

        causal_train_dataloader = DataLoader(
            causal_train_data,
            shuffle=True,
            batch_size=total_causal_local_batch_size,
            num_workers=0,  # non-zero num_workers causes segmenation fault
            generator=torch.Generator().manual_seed(train_seed),
            drop_last=True,
            pin_memory=True,
        )
    else:
        causal_train_dataloader = None

    valid_data = ValidationDataset(args.valid_path, tokenizer, args)

    valid_dataloader = DataLoader(
        valid_data,
        shuffle=False,
        batch_size=args.local_batch_size,
        num_workers=0,  # non-zero num_workers causes segmenation fault
        generator=torch.Generator().manual_seed(42),
        drop_last=True,
        pin_memory=True,
    )

    return masked_train_dataloader, causal_train_dataloader, valid_dataloader


def training_epoch(model, ema_model, train_dataloader, valid_dataloader, optimizer, scheduler, global_step, epoch, args):
    model = model.train()
    optimizer.zero_grad(set_to_none=True)

    # calculate the number of steps to perform in this epoch
    num_steps = min(len(train_dataloader), (args.max_steps - global_step) * args.accumulate_steps)

    # initialize the dataloader and the metrics
    train_dataloader = iter(train_dataloader)
    total_loss, total_accuracy, total_z_loss, total_grad_norm = 0.0, 0.0, 0.0, 0.0

    # get the first batch
    input_ids_, attention_mask_, target_ids_, mask_p_ = get_batch(train_dataloader, args.device, global_step)

    # iterate over the steps
    for local_step in tqdm(range(num_steps), desc="Train iteration", initial=global_step, total=args.max_steps):
        full_input_ids, full_attention_mask, full_target_ids, mask_p = input_ids_, attention_mask_, target_ids_, mask_p_

        accumulate_steps = full_input_ids.size(1) / args.local_batch_size

        for start in range(0, full_input_ids.size(1), args.local_batch_size):
            input_ids = full_input_ids[:, start:start+args.local_batch_size]
            attention_mask = full_attention_mask[start:start+args.local_batch_size]
            target_ids = full_target_ids[:, start:start+args.local_batch_size]

            # forward pass, do a more detailed check of the model every 100 steps
            with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
                with ModelLogger(enable=global_step % 100 == 0, module=model):
                    loss, accuracy, z_loss, num_tokens = model(input_ids, attention_mask, target_ids)

            # calculate the weight for the loss (either token-weighted or not)
            weight = (input_ids.size(1) / args.local_batch_size) / accumulate_steps

            # backward pass through both losses
            ((loss + args.z_loss_weight * z_loss) * weight).backward()

            # add the tracked metrics (for gradient accumulation)
            total_loss += loss.detach() * weight
            total_accuracy += accuracy * weight
            total_z_loss += z_loss * weight

        # get the next batch
        if local_step < num_steps - 1:
            full_input_ids, full_attention_mask, full_target_ids, mask_p = get_batch(train_dataloader, args.device, global_step)

        # clip the gradients
        total_grad_norm += nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient) * weight

        # optimizer step
        optimizer.step()
        scheduler.step()

        with torch.no_grad():

            # EMA update
            for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
                param_k.data.mul_(args.ema_decay).add_((1.0 - args.ema_decay) * param_q.detach().data)

            # be careful here, not all GPUs work with the same training objective
            if args.ratio == 1:
                masked_loss = total_loss.item()
                causal_loss = 0.0
                masked_accuracy = total_accuracy.item()
                causal_accuracy = 0.0
                masked_epoch = epoch
                causal_epoch = 0
            else:
                masked_loss = 0.0
                causal_loss = total_loss.item()
                masked_accuracy = 0.0
                causal_accuracy = total_accuracy.item()
                masked_epoch = 0
                causal_epoch = epoch

        # log the metrics
        wandb.log(
            {
                "masked_epoch": masked_epoch,
                "causal_epoch": causal_epoch,
                "train/loss": total_loss.item(),
                "train/z_loss": total_z_loss.item(),
                "train/perplexity": math.exp(total_loss.item()),
                "train/accuracy": total_accuracy.item() * 100.0,
                "train/masked_accuracy": masked_accuracy * 100.0,
                "train/causal_accuracy": causal_accuracy * 100.0,
                "train/mlm_loss": masked_loss,
                "train/clm_loss": causal_loss,
                "stats/learning_rate": optimizer.param_groups[0]['lr'],
                "stats/grad_norm": total_grad_norm,
                "stats/seq_length": train_dataloader.dataset.seq_length,
                "stats/global_batch_size": args.current_global_batch_size,
                "stats/local_batch_size": args.current_local_batch_size,
                "stats/accumulate_steps": args.accumulate_steps,
                "stats/mask_p": mask_p.item(),
            },
            commit=False
        )

        # zero the accumulated gradients and the metrics
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_accuracy, total_z_loss, total_grad_norm = 0.0, 0.0, 0.0, 0.0

        # checkpoint the model and the full training state
        if global_step % args.save_every == 0:
            save(model, ema_model, optimizer, scheduler, global_step, masked_epoch, causal_epoch, args)

        # validate the model
        if (global_step + 1) % args.validate_every == 0:
            validation_epoch(model, valid_dataloader, masked_epoch, causal_epoch, args)
            model.train()

        # log the stats and commit
        if is_main_process():
            wandb.log({"global_step": global_step}, commit=True)

        global_step += 1

        # Exiting the training due to hitting max steps
        if global_step >= args.max_steps:
            return global_step

    return global_step


def training(model, ema_model, masked_train_dataloader, causal_train_dataloader, valid_dataloader, optimizer, scheduler, global_step, args):
    model = model.train()
    optimizer.zero_grad(set_to_none=True)

    # calculate the number of steps to perform in this epoch
    num_steps = args.max_steps
    masked_epoch, causal_epoch = 0, 0

    # initialize the dataloader and the metrics
    train_progress_bar = tqdm(total=args.max_steps)
    train_masked_iter = iter(masked_train_dataloader)
    train_causal_iter = iter(causal_train_dataloader)
    total_loss, total_masked_loss, total_causal_loss = 0.0, 0.0, 0.0
    total_accuracy, total_masked_accuracy, total_causal_accuracy = 0.0, 0.0, 0.0
    total_z_loss, total_grad_norm = 0.0, 0.0

    # iterate over the steps
    for local_step in range(num_steps):
        try:
            masked_input_ids, masked_attention_mask, masked_target_ids, mask_p = get_batch(train_masked_iter, args.device, global_step)
        except StopIteration:
            masked_epoch += 1
            masked_train_dataloader = load_dataset(args, tokenizer, masked_epoch, global_step, masked_train_dataloader)
            train_masked_iter = iter(masked_train_dataloader)
            masked_input_ids, masked_attention_mask, masked_target_ids, mask_p = get_batch(train_masked_iter, args.device, global_step)
            save(model, ema_model, optimizer, scheduler, global_step, masked_epoch, causal_epoch, args)

        try:
            causal_input_ids, causal_attention_mask, causal_target_ids, mask_p = get_batch(train_causal_iter, args.device, global_step)
        except StopIteration:
            causal_epoch += 1
            causal_train_dataloader = load_dataset(args, tokenizer, causal_epoch, global_step, causal_train_dataloader, mode="causal")
            train_causal_iter = iter(causal_train_dataloader)
            causal_input_ids, causal_attention_mask, causal_target_ids, mask_p = get_batch(train_causal_iter, args.device, global_step)
            save(model, ema_model, optimizer, scheduler, global_step, masked_epoch, causal_epoch, args)

        if masked_train_dataloader.dataset.seq_length != causal_train_dataloader.dataset.seq_length:
            if masked_train_dataloader.dataset.seq_length < causal_train_dataloader.dataset.seq_length:
                masked_epoch += 1
                masked_train_dataloader = load_dataset(args, tokenizer, masked_epoch, global_step, masked_train_dataloader)
                train_masked_iter = iter(masked_train_dataloader)
                masked_input_ids, masked_attention_mask, masked_target_ids, mask_p = get_batch(train_masked_iter, args.device, global_step)
            else:
                causal_epoch += 1
                causal_train_dataloader = load_dataset(args, tokenizer, causal_epoch, global_step, causal_train_dataloader)
                train_causal_iter = iter(causal_train_dataloader)
                causal_input_ids, causal_attention_mask, causal_target_ids, mask_p = get_batch(train_causal_iter, args.device, global_step)

        num_masked = masked_input_ids.size(1)
        full_input_ids = torch.cat([masked_input_ids, causal_input_ids], dim=1)
        full_attention_mask = torch.cat([masked_attention_mask, causal_attention_mask], dim=0)
        full_target_ids = torch.cat([masked_target_ids, causal_target_ids], dim=1)

        accumulate_steps = full_input_ids.size(1) / args.local_batch_size

        for start in range(0, full_input_ids.size(1), args.local_batch_size):
            input_ids = full_input_ids[:, start:start+args.local_batch_size]
            attention_mask = full_attention_mask[start:start+args.local_batch_size]
            target_ids = full_target_ids[:, start:start+args.local_batch_size]

            # forward pass, do a more detailed check of the model every 100 steps
            with torch.cuda.amp.autocast(args.mixed_precision, dtype=torch.bfloat16):
                with ModelLogger(enable=global_step % 100 == 0, module=model):
                    loss, masked_loss, causal_loss, accuracy, masked_accuracy, causal_accuracy, z_loss, num_tokens = model(input_ids, attention_mask, target_ids, num_masked, args.ratio)

            # calculate the weight for the loss
            weight = (input_ids.size(1) / args.local_batch_size) / accumulate_steps

            # backward pass through both losses
            ((loss + args.z_loss_weight * z_loss) * weight).backward()

            # add the tracked metrics (for gradient accumulation)
            total_loss += loss.detach() * weight
            total_masked_loss += (masked_loss.detach() if masked_loss > 0.0 else 0.0) * weight
            total_causal_loss += (causal_loss.detach() if causal_loss > 0.0 else 0.0) * weight
            total_accuracy += accuracy * weight
            total_masked_accuracy += masked_accuracy * weight
            total_causal_accuracy += causal_accuracy * weight
            total_z_loss += z_loss * weight

            num_masked = max(0, num_masked - args.local_batch_size)

        # clip the gradients
        total_grad_norm += nn.utils.clip_grad_norm_(model.parameters(), args.max_gradient) * weight

        # optimizer step
        optimizer.step()
        scheduler.step()

        with torch.no_grad():

            # EMA update
            for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
                param_k.data.mul_(args.ema_decay).add_((1.0 - args.ema_decay) * param_q.detach().data)

        # log the metrics
        if is_main_process():
            wandb.log(
                {
                    "masked_epoch": masked_epoch,
                    "causal_epoch": causal_epoch,
                    "train/loss": total_loss.item(),
                    "train/masked_loss": total_masked_loss.item(),
                    "train/causal_loss": total_causal_loss.item(),
                    "train/z_loss": total_z_loss.item(),
                    "train/perplexity": math.exp(total_loss.item()),
                    "train/accuracy": total_accuracy.item() * 100.0,
                    "train/masked_accuracy": total_masked_accuracy.item() * 100.0,
                    "train/causal_accuracy": total_causal_accuracy.item() * 100.0,
                    "stats/learning_rate": optimizer.param_groups[0]['lr'],
                    "stats/grad_norm": total_grad_norm,
                    "stats/seq_length": masked_train_dataloader.dataset.seq_length,
                    "stats/global_batch_size": args.current_global_batch_size,
                    "stats/local_batch_size": args.current_local_batch_size,
                    "stats/accumulate_steps": args.accumulate_steps,
                    "stats/mask_p": mask_p.item(),
                },
                commit=False
            )

        # zero the accumulated gradients and the metrics
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_masked_loss, total_causal_loss = 0.0, 0.0, 0.0
        total_accuracy, total_masked_accuracy, total_causal_accuracy = 0.0, 0.0, 0.0
        total_z_loss, total_grad_norm = 0.0, 0.0

        # checkpoint the model and the full training state
        if global_step % args.save_every == 0:
            save(model, ema_model, optimizer, scheduler, global_step, masked_epoch, causal_epoch, args)

        # validate the model
        if (global_step + 1) % args.validate_every == 0:
            validation_epoch(model, valid_dataloader, masked_epoch, causal_epoch, args)
            model.train()

        # log the stats and commit
        if is_main_process():
            wandb.log({"global_step": global_step}, commit=True)

        global_step += 1
        train_progress_bar.update()

        # Exiting the training due to hitting max steps
        if global_step >= args.max_steps:
            return global_step

    return global_step, masked_epoch, causal_epoch


if __name__ == "__main__":
    args = parse_arguments()

    tokenizer = Tokenizer.from_file(str(args.tokenizer_path))
    setup_training(args, tokenizer)
    model, ema_model, optimizer, scheduler, global_step, start_epoch = prepare_model_and_optimizer(args)

    masked_train_dataloader, causal_train_dataloader, valid_dataloader = init_datasets(args, tokenizer)
    if args.ratio != 1 and args.ratio != 0:
        global_step, masked_epoch, causal_epoch = training(model, ema_model, masked_train_dataloader, causal_train_dataloader, valid_dataloader, optimizer, scheduler, global_step, args)
    elif args.ratio == 1:
        causal_epoch = 0
        for masked_epoch in count(start=start_epoch):
            global_step = training_epoch(model, ema_model, masked_train_dataloader, valid_dataloader, optimizer, scheduler, global_step, masked_epoch, args)
            masked_train_dataloader = load_dataset(args, tokenizer, masked_epoch, global_step, masked_train_dataloader, mode="masked")

            if global_step >= args.max_steps:
                break
    else:
        masked_epoch = 0
        for causal_epoch in count(start=start_epoch):
            global_step = training_epoch(model, ema_model, causal_train_dataloader, valid_dataloader, optimizer, scheduler, global_step, causal_epoch, args)
            causal_train_dataloader = load_dataset(args, tokenizer, causal_epoch, global_step, causal_train_dataloader, mode="causal")

            if global_step >= args.max_steps:
                break

    save(model, ema_model, optimizer, scheduler, global_step, masked_epoch, causal_epoch, args)
    validation_epoch(model, valid_dataloader, masked_epoch, causal_epoch, args, commit=True)
