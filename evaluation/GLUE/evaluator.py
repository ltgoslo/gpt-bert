from __future__ import annotations

import torch
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
import copy
import os

if int(os.environ["SLURM_PROCID"]) <= 3:
    import wandb


def train(model, train_dataloader, args, optimizer, scheduler, device, valid_dataloader, verbose: bool = False):
    total_steps = len(train_dataloader)
    step: int = 0
    best_score = None
    best_model = None
    update_best: bool = False

    for epoch in range(args.num_epochs):
        step = train_epoch(model, train_dataloader, args, epoch, step, total_steps, optimizer, scheduler, device, verbose)

        if valid_dataloader is not None:
            metrics = evaluate(model, valid_dataloader, args.metrics, device, verbose)
            if args.keep_best_model:
                score: float = metrics[args.metric_for_valid]
                if compare_scores(best_score, score, args.higher_is_better):
                    best_model = copy.deepcopy(model)
                    best_score = score
                    update_best = True

        if args.save:
            if args.keep_best_model and update_best:
                save_model(best_model, args)
                update_best = False
            else:
                save_model(model, args)

    return best_model


def train_epoch(model, train_dataloader, args, epoch, global_step: int, total_steps: int, optimizer, scheduler, device: str, verbose: bool = False):
    model.train()

    progress_bar = tqdm(initial=global_step, total=total_steps)

    for input_data, attention_mask, labels in train_dataloader:
        input_data = input_data.to(device=device)
        attention_mask = attention_mask.to(device=device)
        labels = labels.to(device=device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(True, dtype=torch.bfloat16):
            logits = model(input_data, attention_mask)  # loss = model(input_data, attention_mask, labels)
        loss = F.cross_entropy(logits, labels)
        loss.backward()

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        metrics = calculate_metrics(logits, labels, args.metrics)

        metrics_string = ", ".join([f"{key}: {value}" for key, value in metrics.items()])

        wandb.log({
            "loss": loss.item(),
            **{f"train_{key}": value for key, value in metrics.items()}
        })

        progress_bar.update()

        if verbose:
            progress_bar.set_postfix_str(metrics_string)

        global_step += 1

    progress_bar.close()

    return global_step


@torch.no_grad()
def evaluate(model, valid_dataloader, metrics_to_calculate: list[str], device: str, verbose: bool = False):
    model.eval()

    progress_bar = tqdm(total=len(valid_dataloader))

    labels = []
    logits = []

    for input_data, attention_mask, label in valid_dataloader:
        input_data = input_data.to(device=device)
        attention_mask = attention_mask.to(device=device)
        label = label.to(device=device)

        with torch.cuda.amp.autocast(True, dtype=torch.bfloat16):
            logit = model(input_data, attention_mask)

        logits.append(logit)
        labels.append(label)

        progress_bar.update()

    labels = torch.cat(labels, dim=0)
    logits = torch.cat(logits, dim=0)

    metrics = calculate_metrics(logits, labels, metrics_to_calculate)

    wandb.log({
        **{f"valid_{key}": value for key, value in metrics.items()}
    })

    progress_bar.close()

    if verbose:
        metrics_string = "\n".join([f"{key}: {value}" for key, value in metrics.items()])
        print(metrics_string)

    return metrics


@torch.no_grad()
def predict_classification(model, valid_dataloader, metrics_to_calculate: list[str], device: str, verbose: bool = False):
    model.eval()

    progress_bar = tqdm(total=len(valid_dataloader))

    labels = []
    logits = []

    for input_data, attention_mask, label in valid_dataloader:
        input_data = input_data.to(device=device)
        attention_mask = attention_mask.to(device=device)
        label = label.to(device=device)

        logit = model(input_data, attention_mask)

        logits.append(logit)
        labels.append(label)

        progress_bar.update()

    labels = torch.cat(labels, dim=0)
    logits = torch.cat(logits, dim=0)
    preds = logits.argmax(dim=-1)

    metrics = calculate_metrics(logits, labels, metrics_to_calculate)

    progress_bar.close()

    if verbose:
        metrics_string = "\n".join([f"{key}: {value}" for key, value in metrics.items()])
        print(metrics_string)

    return metrics, preds


def save_model(model, args) -> None:
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), args.save_dir)


def compare_scores(best: float, current: float, bigger_better: bool) -> bool:
    if best is None:
        return True
    else:
        if current > best and bigger_better:
            return True
        elif current < best and not bigger_better:
            return True
        return False


def calculate_metrics(logits, labels, metrics_to_calculate: list[str]) -> dict[str, float]:
    predictions = logits.argmax(dim=-1).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    metrics = dict()

    for metric in metrics_to_calculate:
        if metric == "f1":
            metrics["f1"] = f1_score(labels, predictions)
        elif metric == "accuracy":
            metrics["accuracy"] = accuracy_score(labels, predictions)
        elif metric == "mcc":
            metrics["mcc"] = matthews_corrcoef(labels, predictions)
        else:
            print(f"Metric {metric} is unknown / not implemented. It will be skipped!")

    return metrics
