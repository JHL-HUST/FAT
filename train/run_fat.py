"""
Script for running finetuning on glue tasks under Freelb.

Largely copied from:
    https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
"""
import argparse
import logging
import os
import sys
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import transformers
from transformers import (
    AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
)
from tqdm import tqdm

sys.path.append("..")
import utils as utils

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.

    From:
        https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_loss(logits, labels):
    loss = F.cross_entropy(logits, labels.squeeze(-1))
    return loss

class myDataset(utils.Huggingface_dataset):
    def __init__(
            self,
            args,
            tokenizer,
            name_or_dataset: str,
            subset: str = None,
            split="train",
            shuffle=False,
    ):
        utils.Huggingface_dataset.__init__(self, args, tokenizer, name_or_dataset, subset, split, shuffle)

    def _format_examples(self, idx, examples):
        sentence = (
            (examples[self.input_columns[0]],) if self.input_columns[1] is None else (examples[self.input_columns[0]], examples[self.input_columns[1]])
        )
        inputs = self.tokenizer(*sentence, truncation=True, max_length=self.max_seq_len, return_tensors='pt')
        output = int(examples['label'])
        return (idx, inputs, output)

    def __getitem__(self, i):
        """Return i-th sample."""
        if isinstance(i, int):
            return self._format_examples(i, self.dataset[i])
        else:
            # `idx` could be a slice or an integer. if it's a slice,
            # return the formatted version of the proper slice of the list
            return [
                self._format_examples(j, self.dataset[j]) for j in range(i.start, i.stop)
            ]

class myCollator:
    """
    Collates transformer outputs.
    """

    def __init__(self, pad_token_id=0):
        self._pad_token_id = pad_token_id

    def __call__(self, features):
        # Separate the list of inputs and labels
        sample_idx, model_inputs, labels = list(zip(*features))
        # Assume that all inputs have the same keys as the first
        proto_input = model_inputs[0]
        keys = list(proto_input.keys())
        padded_inputs = {}
        for key in keys:
            if key == 'input_ids':
                padding_value = self._pad_token_id
            else:
                padding_value = 0
            # NOTE: We need to squeeze to get rid of fake batch dim.
            sequence = [x[key] for x in model_inputs]
            padded = utils.pad_squeeze_sequence(sequence, batch_first=True, padding_value=padding_value)
            padded_inputs[key] = padded
        labels = torch.tensor([x for x in labels])
        sample_idx = torch.tensor([x for x in sample_idx])
        return sample_idx, padded_inputs, labels
    

def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
    model.to(device)

    collator = myCollator(pad_token_id=tokenizer.pad_token_id)
    ori_collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)

    # for training
    train_dataset = myDataset(args, tokenizer, name_or_dataset=args.dataset_name, subset=args.task_name)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

    # for dev
    dev_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                            subset=args.task_name, split=args.valid)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=ori_collator)

    # for test
    if args.do_test:
        test_dataset = utils.Huggingface_dataset(args, tokenizer, name_or_dataset=args.dataset_name,
                                                 subset=args.task_name, split='test')
        test_loader = DataLoader(test_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    if args.bias_correction:
        betas = (0.9, 0.999)
    else:
        betas = (0.0, 0.000)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=args.lr,
        betas=betas,
        eps=args.adam_epsilon
    )


    num_training_steps = len(train_dataset) * args.epochs // args.bsz
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, num_training_steps)
    if args.dataset_name == 'imdb' or args.dataset_name == 'ag_news':
        output_dir = Path(os.path.join(args.ckpt_dir, 'fat_{}_{}_{}_adv-steps{}_adv-lr{}_epochs{}_init{}'
                                       .format(args.nt_at_interval, args.model_name, args.dataset_name,
                                               args.adv_steps, args.adv_lr, args.epochs, args.adv_init_mag)))
    else:
        output_dir = Path(os.path.join(args.ckpt_dir, 'fat_{}_{}_{}-{}_adv-steps{}_adv-lr{}_epochs{}_init{}'
                                       .format(args.nt_at_interval, args.model_name, args.dataset_name, args.task_name,
                                               args.adv_steps, args.adv_lr, args.epochs, args.adv_init_mag)))
    if not output_dir.exists():
        logger.info(f'Making checkpoint directory: {output_dir}')
        output_dir.mkdir(parents=True)
    elif not args.force_overwrite:
        raise RuntimeError('Checkpoint directory already exists.')
    log_file = os.path.join(output_dir, 'INFO.log')
    logger.addHandler(logging.FileHandler(log_file))

    try:
        best_accuracy = 0
        delta_cache = np.zeros((len(train_dataset), utils.task_to_MAX_SEQ_LEN[args.dataset_name], model.bert.embeddings.word_embeddings.weight.shape[-1]), dtype=np.float32) # 25000, 256, 768
        adv_flag = False # if the epoch for AT
        for epoch in range(args.epochs):
            adv_flag = (epoch % args.nt_at_interval == 0)
            logger.info('Training...')
            model.train()
            avg_loss = utils.ExponentialMovingAverage()
            pbar = tqdm(train_loader)
            for sample_idx, model_inputs, labels in pbar:
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                labels = labels.to(device)
                model.zero_grad()
                word_embedding_layer = model.get_input_embeddings()

                # ---------------- Adversarial training under Freelb ----------------- #
                # for PGD-K, clean batch is not used when training

                input_ids = model_inputs['input_ids']
                attention_mask = model_inputs['attention_mask']
                embedding_init = word_embedding_layer(input_ids)

                delta = torch.zeros_like(embedding_init)

                if adv_flag:
                    # initialize delta
                    if args.adv_init_mag > 0:
                        input_mask = attention_mask.to(embedding_init)
                        input_lengths = torch.sum(input_mask, 1)
                        if args.adv_norm_type == 'l2':
                            if epoch == 0:
                                delta = torch.zeros_like(embedding_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                            else:
                                direction = torch.tensor(delta_cache[sample_idx.numpy()][:, :input_ids.shape[-1], :], dtype=torch.float32).to(device)
                                delta = torch.zeros_like(embedding_init).uniform_(0, 1) * torch.sign(direction) * input_mask.unsqueeze(2)
                            dims = input_lengths * embedding_init.size(-1)
                            magnitude = args.adv_init_mag / torch.sqrt(dims)
                            delta = (delta * magnitude.view(-1, 1, 1))
                        elif args.adv_norm_type == 'linf':
                            delta = torch.zeros_like(embedding_init).uniform_(-args.adv_init_mag,
                                                                        args.adv_init_mag) * input_mask.unsqueeze(2)
                        random_init = delta.detach().clone()
                    total_loss = 0.0
                    for astep in range(args.adv_steps):
                        # (0) forward
                        delta.requires_grad_()
                        batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
                        logits = model(**batch).logits

                        # (1) backward
                        losses = get_loss(logits, labels)
                        loss = torch.mean(losses)
                        # loss = loss / args.adv_steps
                        total_loss += loss.item()
                        loss.backward()
                        # loss.backward(retain_graph=True)

                        # if astep == args.adv_steps - 1:
                        #     break

                        # (2) get gradient on delta
                        delta_grad = delta.grad.clone().detach()

                        # (3) update and clip
                        if args.adv_norm_type == "l2":
                            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                            denorm = torch.clamp(denorm, min=1e-8)
                            delta = (delta + args.adv_lr * delta_grad / denorm).detach()
                            if args.adv_max_norm > 0:
                                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                                exceed_mask = (delta_norm > args.adv_max_norm).to(embedding_init)
                                reweights = (args.adv_max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1)
                                delta = (delta * reweights).detach()
                        elif args.adv_norm_type == "linf":
                            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1,
                                                                                                                    1)
                            denorm = torch.clamp(denorm, min=1e-8)
                            delta = (delta + args.adv_lr * delta_grad / denorm).detach()

                        model.zero_grad()
                        optimizer.zero_grad()
                        embedding_init = word_embedding_layer(input_ids)
                    # tr_loss += total_loss
                    
                    if args.adv_init_mag > 0:
                        current_delta = (delta - random_init).detach().cpu().numpy()
                    else:
                        current_delta = delta.detach().cpu().numpy()
                    current_delta = np.pad(current_delta, ((0,0),(0,delta_cache.shape[1]-current_delta.shape[1]),(0,0)))
                    delta_cache[sample_idx.numpy()] = current_delta
                else:
                    delta = torch.tensor(delta_cache[sample_idx.numpy()][:, :input_ids.shape[-1], :], dtype=torch.float32).to(device)

                delta.requires_grad = False
                batch = {'inputs_embeds': delta + embedding_init, 'attention_mask': attention_mask}
                model.zero_grad()
                optimizer.zero_grad()
                logits = model(**batch).logits
                losses = get_loss(logits, labels)
                loss = torch.mean(losses)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                avg_loss.update(total_loss)
                pbar.set_description(f'epoch: {epoch: d}, '
                                     f'loss: {avg_loss.get_metric(): 0.4f}, '
                                     f'lr: {optimizer.param_groups[0]["lr"]: .3e}')
                
                    
            s = Path(str(output_dir) + '/epoch' + str(epoch))
            if not s.exists():
                s.mkdir(parents=True)
            model.save_pretrained(s)
            tokenizer.save_pretrained(s)
            torch.save(args, os.path.join(s, "training_args.bin"))

            logger.info('Evaluating...')
            model.eval()
            if epoch == args.epochs - 1:
                correct = 0
                total = 0
                with torch.no_grad():
                    for model_inputs, labels in dev_loader:
                        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                        labels = labels.to(device)
                        logits = model(**model_inputs).logits
                        _, preds = logits.max(dim=-1)
                        correct += (preds == labels.squeeze(-1)).sum().item()
                        total += labels.size(0)
                    accuracy = correct / (total + 1e-13)
                logger.info(f'Epoch: {epoch}, '
                            # f'Loss: {avg_loss.get_metric(): 0.4f}, '
                            # f'Lr: {optimizer.param_groups[0]["lr"]: .3e}, '
                            f'Accuracy: {accuracy}')

        #     if accuracy > best_accuracy:
        #         logger.info('Best performance so far.')
        #         # model.save_pretrained(output_dir)
        #         # tokenizer.save_pretrained(output_dir)
        #         # torch.save(args, os.path.join(output_dir, "training_args.bin"))
        #         best_accuracy = accuracy
        #         best_dev_epoch = epoch
        # logger.info(f'Best dev metric: {best_accuracy} in Epoch: {best_dev_epoch}')
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        torch.save(args, os.path.join(output_dir, "training_args.bin"))

    except KeyboardInterrupt:
        logger.info('Interrupted...')

    if args.do_test:
        logger.info('Testing...')
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for model_inputs, labels in test_loader:
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                labels = labels.to(device)
                logits, *_ = model(**model_inputs)
                _, preds = logits.max(dim=-1)
                correct += (preds == labels.squeeze(-1)).sum().item()
                total += labels.size(0)
            accuracy = correct / (total + 1e-13)
        logger.info(f'Accuracy: {accuracy : 0.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='bert-base-uncased')
    parser.add_argument("--dataset_name", default='glue', type=str)
    parser.add_argument('--task-name', type=str, default=None)
    parser.add_argument('--field-a', type=str, default='sentence')
    parser.add_argument('--field-b', type=str, default=None)
    parser.add_argument('--label-field', type=str, default='label')
    parser.add_argument('--ckpt-dir', type=Path, default=Path('../saved_models/'))
    parser.add_argument('--num-labels', type=int, default=2)
    parser.add_argument('--valid', type=str, default='validation')
    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=False)
    parser.add_argument('--do_lower_case', type=bool, default=True)
    parser.add_argument('--eval_size', type=int, default=32)

    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--lr', type=float, default=2e-5)  # different with normal finetune
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_steps", default=500, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bias-correction', default=True)
    parser.add_argument('--weight-decay', type=float, default=1e-6)  # small value has little effect
    parser.add_argument('-f', '--force-overwrite', default=True)
    parser.add_argument('--debug', action='store_true')

    # Adversarial training specific
    parser.add_argument('--adv-steps', default=3, type=int,
                        help='Number of gradient ascent steps for the adversary')
    parser.add_argument('--adv-lr', default=0.03, type=float,
                        help='Step size of gradient ascent')
    parser.add_argument('--adv-init-mag', default=0.05, type=float,
                        help='Magnitude of initial (adversarial?) perturbation')
    parser.add_argument('--adv-max-norm', default=0, type=float,
                        help='adv_max_norm = 0 means unlimited')
    parser.add_argument('--adv-norm-type', default='l2', type=str,
                        help='norm type of the adversary')
    # parser.add_argument('--adv-change-rate', default=0.2, type=float,
    #                     help='change rate of a sentence')
    parser.add_argument('--max-grad-norm', default=1, type=float, help='max gradient norm')
    parser.add_argument('--nt-at-interval', default=3, type=int,
                        help='The value of I for the variant FAT-I.')


    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)
