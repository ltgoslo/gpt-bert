from __future__ import annotations

import torch
import torch.nn as nn
import argparse
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils import seed_everything, cosine_schedule_with_warmup
from functools import partial
import json
import os
import pathlib
import copy

from evaluator import train, predict_classification
from dataset import Dataset, collate_function
from classifier import ModelForSequenceClassification

if int(os.environ["SLURM_PROCID"]) <= 10:
    import wandb


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Required Parameters
    parser.add_argument("--results_dir", default="/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue_results", type=pathlib.Path, help="The output directory where the results will be written.")
    parser.add_argument("--train_data", default="/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/data/mnli.subs.jsonl", type=pathlib.Path, help="Path to file containing the training dataset, we expect it to be in a JSONL format.")
    parser.add_argument("--model_path_or_name", default="/scratch/project_465000144/dasamuel/babylm-v2/checkpoints/small_10M_15-16_babycosmofine_ema.bin", type=pathlib.Path, help="The local path to the model binary.")
    parser.add_argument("--tokenizer_path", default="/scratch/project_465000144/dasamuel/babylm-v2/tokenizer_babycosmofine_10M.json", type=str, help="The vocabulary the model was trained on.")
    #parser.add_argument("--model_path_or_name", default="/scratch/project_465000144/dasamuel/babylm-v2/checkpoints/base_100M_7-8_ema.bin", type=pathlib.Path, help="The local path to the model binary.")
    #parser.add_argument("--tokenizer_path", default="/scratch/project_465000144/dasamuel/babylm-v2/tokenizer_100M.json", type=str, help="The vocabulary the model was trained on.")
    parser.add_argument("--config_file", default="/scratch/project_465000144/dasamuel/babylm-v2/configs/small.json", type=pathlib.Path, help="Path to the configuration file of the model.")
    parser.add_argument("--architecture", default="base", type=str, help="The architecture of the model, available: base, attglu, attgate, densemod, densesubmod, densecont, elc, qkln")
    parser.add_argument("--metrics", default=["accuracy"], nargs='+', help="List of metrics to evaluate for the model (accuracy, f1, and mcc).")
    parser.add_argument("--num_labels", default=3, type=int, help="The number of labels in the dataset. (3 for MNLI, 2 for all other tasks)")
    parser.add_argument("--seed", default=42, type=int, help="The seed for the Random Number Generator.")
    parser.add_argument("--task", default="mnli", type=str, help="The task to fine-tune for.")

    # Optinal Parameters
    parser.add_argument("--ema_decay", default=0.0, type=float, help="The maximum learning rate during fine-tuning.")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False, help="Whether to output the metrics in terminal during the run.")
    parser.add_argument("--valid_data", type=pathlib.Path, help="Path to file containing the validation dataset to test on, we expect it to be in a JSONL format.")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the fine-tuned model.")
    parser.add_argument("--save_dir", type=pathlib.Path, help="The directory in which to save the fine-tuned model.")
    parser.add_argument("--keep_best_model", action=argparse.BooleanOptionalAction, default=True, help="Whether to only keep the model with the best score based on the metric_for_valid.")
    parser.add_argument("--metric_for_valid", type=str, help="The metric used to compare the model when finding the best model.")
    parser.add_argument("--higher_is_better", action=argparse.BooleanOptionalAction, default=True, help="Wheter a higher value for the metric for valid is better or not.")

    # Hyperparameters
    parser.add_argument("--batch_size", default=16, type=int, help="The batch size during fine-tuning.")
    parser.add_argument("--valid_batch_size", default=64, type=int, help="The batch size during inference.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The maximum learning rate during fine-tuning.")
    parser.add_argument("--sequence_length", default=512, type=int, help="The max sequence length before truncation.")
    parser.add_argument("--num_epochs", default=10, type=int, help="Number of epochs to fine-tune the code for.")
    parser.add_argument("--classifier_dropout", default=0.1, type=float, help="The dropout applied to the classifier head. (Needs to be a value between 0 and 1)")
    parser.add_argument("--classifier_layer_norm_eps", default=1.0e-5, type=float, help="The epsilon to add to the layer norm operations to stabalize the division and avoid dividing by zero.")
    # Dropout in attention?
    # Dropout in pre-trained feed-forward?
    parser.add_argument("--weight_decay", default=0.01, type=float, help="The weight decay to apply for the optimizer (if a weight decay is relevant). (Needs to be a value between 0 and 1)")
    parser.add_argument("--warmup_proportion", default=0.06, type=float, help="The proportion of the fine-tuning steps where the learning rate increases from 0 to its maximum value. (Needs to be a value between 0 and 1)")
    parser.add_argument("--min_factor", default=0.1, type=float, help="The final factor to which the max learning rate is multiplied to find the final learning rate.")
    parser.add_argument("--scheduler", default="cosine", type=str, help="The learning rate scheduler to use for fine-tuning. none means that no learning rate scheduling was chosen.")  # Not implemented
    parser.add_argument("--optimizer", default="adamw", type=str, help="The optimizer to use for the fine-tuning of the model.")  # Not implemented
    parser.add_argument("--beta1", default=0.9, type=float, help="The value of beta1 (or beta) in optimizers that require it.")
    parser.add_argument("--beta2", default=0.999, type=float, help="The value of beta2 in optimizers that require it.")
    parser.add_argument("--optimizer_eps", default=1e-8, type=float, help="The epsilon to add to the optimizer operations (if relevant) to stabalize and avoid dividing by zero.")
    parser.add_argument("--amsgrad", default=False, action=argparse.BooleanOptionalAction, help="Whether to use AMSGrad variant of the AdamW optimizer. (Only relevant if adamw chosen for optimizer)")
    # Other things?

    args = parser.parse_args()

    args.rank = int(os.environ["SLURM_PROCID"])

    args.metrics = ["accuracy", "f1", "mcc"]
    if args.rank == 0:
        args.task = "boolq"
        args.train_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/boolq.train.jsonl"
        args.valid_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/boolq.valid.jsonl"
        args.num_labels = 2
        args.metric_for_valid = "accuracy"
        args.num_epochs = 10
        args.learning_rate = 1e-4
        args.batch_size = 16
    
    elif args.rank == 1:
        args.task = "cola"
        args.train_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/cola.train.jsonl"
        args.valid_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/cola.valid.jsonl"
        args.num_labels = 2
        args.metric_for_valid = "mcc"
        args.num_epochs = 10
        args.learning_rate = 1e-4
        args.batch_size = 16

    elif args.rank == 2:
        args.task = "mnli"
        args.train_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/mnli.train.jsonl"
        args.valid_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/mnli.valid.jsonl"
        args.num_labels = 3
        args.metric_for_valid = "accuracy"
        args.metrics = ["accuracy"]
        args.num_epochs = 3
        args.learning_rate = 1e-4
        args.batch_size = 32
    
    elif args.rank == 3:
        args.task = "mnli"
        args.train_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/mnli.train.jsonl"
        args.valid_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/mnli-mm.valid.jsonl"
        args.num_labels = 3
        args.metric_for_valid = "accuracy"
        args.metrics = ["accuracy"]
        args.num_epochs = 3
        args.learning_rate = 1e-4
        args.batch_size = 32

    elif args.rank == 4:
        args.task = "mrpc"
        args.train_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/mrpc.train.jsonl"
        args.valid_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/mrpc.valid.jsonl"
        args.num_labels = 2
        args.metric_for_valid = "f1"
        args.num_epochs = 10
        args.learning_rate = 1e-4
        args.batch_size = 16
    
    elif args.rank == 5:
        args.task = "multirc"
        args.train_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/multirc.train.jsonl"
        args.valid_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/multirc.valid.jsonl"
        args.num_labels = 2
        args.metric_for_valid = "accuracy"
        args.num_epochs = 3
        args.learning_rate = 1e-4
        args.batch_size = 32
    
    elif args.rank == 6:
        args.task = "qnli"
        args.train_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/qnli.train.jsonl"
        args.valid_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/qnli.valid.jsonl"
        args.num_labels = 2
        args.metric_for_valid = "accuracy"
        args.num_epochs = 3
        args.learning_rate = 1e-4
        args.batch_size = 32

    elif args.rank == 7:
        args.task = "qqp"
        args.train_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/qqp.train.jsonl"
        args.valid_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/qqp.valid.jsonl"
        args.num_labels = 2
        args.metric_for_valid = "f1"
        args.num_epochs = 3
        args.learning_rate = 1e-4
        args.batch_size = 32

    elif args.rank == 8:
        args.task = "rte"
        args.train_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/rte.train.jsonl"
        args.valid_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/rte.valid.jsonl"
        args.num_labels = 2
        args.metric_for_valid = "accuracy"
        args.num_epochs = 10
        args.learning_rate = 1e-4
        args.batch_size = 16
    
    elif args.rank == 9:
        args.task = "sst2"
        args.train_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/sst2.train.jsonl"
        args.valid_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/sst2.valid.jsonl"
        args.num_labels = 2
        args.metric_for_valid = "accuracy"
        args.num_epochs = 10
        args.learning_rate = 1e-4
        args.batch_size = 16
    
    elif args.rank == 10:
        args.task = "wsc"
        args.train_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/wsc.train.jsonl"
        args.valid_data = "/scratch/project_465000144/dasamuel/babylm-v2/5_evaluation/glue/data/wsc.valid.jsonl"
        args.num_labels = 2
        args.metric_for_valid = "accuracy"
        args.num_epochs = 10
        args.learning_rate = 1e-4
        args.batch_size = 16
    
    else:
        exit()

    args.train_data = pathlib.Path(args.train_data)
    args.valid_data = pathlib.Path(args.valid_data)

    wandb.init(
        name=f"{args.model_path_or_name.stem}_{args.task}",
        project="BabyLM-v2-GLUE",
        entity="nor-ret"
    )

    return args


def load_config(args: argparse.Namespace) -> argparse.Namespace:
    with args.config_file.open("r") as f:
        config = json.load(f)
    for k, v in config.items():
        if k not in args:
            setattr(args, k, v)
    return args


if __name__ == "__main__":
    args: argparse.Namespace = parse_arguments()
    args = load_config(args)

    seed_everything(args.seed)

    args.world_size = int(os.environ["WORLD_SIZE"])
    args.gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
    args.local_rank = args.rank - args.gpus_per_node * (args.rank // args.gpus_per_node)
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    tokenizer: Tokenizer = Tokenizer.from_file(args.tokenizer_path)
    tokenizer.enable_padding(pad_id=3, pad_token="␢")
    tokenizer.enable_truncation(args.sequence_length)

    train_dataset: Dataset = Dataset(args.train_data, args.task)
    train_dataloader: DataLoader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=partial(collate_function, tokenizer), shuffle=True, drop_last=True)

    valid_dataloader: DataLoader | None = None
    if args.valid_data is not None:
        valid_dataset: Dataset = Dataset(args.valid_data, args.task)
        valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, collate_fn=partial(collate_function, tokenizer))
        model_name = args.model_path_or_name.stem
        if args.task == "mnli":
            args.task = args.valid_data.stem.split(".")[0]
        output_path = args.results_dir / model_name / args.task

        if not os.path.exists(args.results_dir):
            os.mkdir(args.results_dir)
        if not os.path.exists(args.results_dir / model_name):
            os.mkdir(args.results_dir / model_name)
        if not os.path.exists(output_path):
            os.mkdir(output_path)

    model: nn.Module = ModelForSequenceClassification(args).to(device)
    model.transformer.load_state_dict(torch.load(args.model_path_or_name, map_location="cpu", weights_only=True))

    optimizer: torch.optim.Optimizer = AdamW(model.parameters(), lr=args.learning_rate, betas=(args.beta1, args.beta2), eps=args.optimizer_eps, weight_decay=args.weight_decay, amsgrad=args.amsgrad)
    total_steps: int = args.num_epochs * len(train_dataloader)
    scheduler: torch.optim.lr_scheduler.LRScheduler = cosine_schedule_with_warmup(optimizer, int(args.warmup_proportion * total_steps), total_steps, 0.1)
    best_model: nn.Module = train(model, train_dataloader, args, optimizer, scheduler, device, valid_dataloader, args.verbose)

    if best_model is not None:
        model.load_state_dict(best_model.state_dict())

    if valid_dataloader is not None:
        metrics, preds = predict_classification(model, valid_dataloader, args.metrics, device, args.verbose)
        pred_dict = {f"{args.task}": {"predictions": []}}
        for i, pred in enumerate(preds):
            pred_dict[f"{args.task}"]["predictions"].append({"id": f"{args.task}_{i}", "pred": int(pred)})
        with (output_path / "results.txt").open("w") as file:
            file.write("\n".join([f"{key}: {value}" for key, value in metrics.items()]))
        with (output_path / "predictions.json").open("w") as file:
            json.dump(pred_dict, file)


# model = pathlib.Path(...)
# model.stem
# with (res_dir / f"results_{model.stem}_{args.task}").open("r")
