# coding=utf-8
"""
Attack Module
"""
import argparse
import os
import csv
import logging

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["HF_DATASETS_OFFLINE"] = '1'
# os.environ["TRANSFORMERS_OFFLINE"] = '1'

# os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import textattack
from textattack.attack_recipes import (PWWSRen2019,
                                       GeneticAlgorithmAlzantot2018,
                                       FasterGeneticAlgorithmJia2019,
                                       DeepWordBugGao2018,
                                       PSOZang2020,
                                       TextBuggerLi2018,
                                       BERTAttackLi2020,
                                       TextFoolerJin2019,
                                       HotFlipEbrahimi2017)
from textattack.transformations import WordSwapEmbedding, WordSwapWordNet, WordSwapMaskedLM
from textattack.transformations import (
    WordSwapNeighboringCharacterSwap,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterSubstitution,
)
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.constraints.pre_transformation import InputColumnModification
from textattack.datasets import HuggingFaceDataset
from textattack.attack_results import SuccessfulAttackResult, MaximizedAttackResult, FailedAttackResult
from transformers import (
    AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AdamW
)
from textattack.models.wrappers.huggingface_model_wrapper import HuggingFaceModelWrapper
logger = logging.getLogger(__name__)


def attack_parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--do_lower_case', type=bool, default=True)
    parser.add_argument("--model_name_or_path",
                        default='/root/robust_transfer/saved_models/flooding_bert-base-uncased_glue-sst2_lr2e-05_epochs10_b0.01/epoch9',
                        type=str)
    parser.add_argument("--results_file", default='attack_log.csv', type=str)
    parser.add_argument("--dataset_name", default='glue', type=str)
    parser.add_argument("--task_name", default=None, type=str)
    parser.add_argument('--valid', type=str, default='validation')  # test for imdb, agnews; validation for GLUEs
    parser.add_argument("--num_examples", default=872, type=int)  # number of attack sentences
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--attack_method", default=None, type=str)
    parser.add_argument("--modify_ratio", default=0.25, type=float)

    args = parser.parse_args()
    return args

def build_attacker(args, model: HuggingFaceModelWrapper):
    if args.attack_method == 'hotflip':
        return HotFlipEbrahimi2017.build(model)
    if args.attack_method == 'pwws':
        attacker = PWWSRen2019.build(model)
    elif args.attack_method == 'pso':
        attacker = PSOZang2020.build(model)
    elif args.attack_method == 'ga':
        attacker = GeneticAlgorithmAlzantot2018.build(model)
    elif args.attack_method == 'fga':
        attacker = FasterGeneticAlgorithmJia2019.build(model)
    elif args.attack_method == 'textfooler':
        attacker = TextFoolerJin2019.build(model)
    elif args.attack_method == 'bae':
        attacker = BERTAttackLi2020.build(model)
        # attacker.transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=args.neighbour_vocab_size)
    elif args.attack_method == 'deepwordbug':
        attacker = DeepWordBugGao2018.build(model)
        # attacker.transformation = RandomCompositeTransformation(
        #     [
        #         # (1) Swap: Swap two adjacent letters in the word.
        #         WordSwapNeighboringCharacterSwap(),
        #         # (2) Substitution: Substitute a letter in the word with a random letter.
        #         WordSwapRandomCharacterSubstitution(),
        #         # (3) Deletion: Delete a random letter from the word.
        #         WordSwapRandomCharacterDeletion(),
        #         # (4) Insertion: Insert a random letter in the word.
        #         WordSwapRandomCharacterInsertion(),
        #     ],
        #     total_count=args.neighbour_vocab_size
        # )
    elif args.attack_method == 'textbugger':
        attacker = TextBuggerLi2018.build(model)
    else:
        attacker = TextFoolerJin2019.build(model)

    input_column_modification = InputColumnModification(
        ["premise", "hypothesis"], {"premise"}
    )
    input_column_modification = InputColumnModification(
        ["question", "sentence"], {"question"}
    )
    attacker.pre_transformation_constraints.append(input_column_modification)

    return attacker

def main():
    args = attack_parse_args()

    # for model
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # for dataset
    model_wrapper = HuggingFaceModelWrapper(model, tokenizer)
    # attack = TextFoolerJin2019.build(model_wrapper)
    attack = build_attacker(args, model_wrapper)
    dataset = HuggingFaceDataset(args.dataset_name, args.task_name, split=args.valid)

    # for attack
    attack_args = textattack.AttackArgs(num_examples=args.num_examples,
                                        disable_stdout=True, random_seed=args.seed)
    attacker = textattack.Attacker(attack, dataset, attack_args)
    num_results = 0
    num_successes = 0
    num_failures = 0
    for result in attacker.attack_dataset():
        logger.info(result)
        num_results += 1
        if (
                type(result) == SuccessfulAttackResult
                or type(result) == MaximizedAttackResult
        ):
            num_successes += 1
        if type(result) == FailedAttackResult:
            num_failures += 1
    logger.info("[Succeeded / Failed / Total] {} / {} / {}".format(num_successes, num_failures, num_results))

    # compute metric
    original_accuracy = (num_successes + num_failures) * 100.0 / num_results
    accuracy_under_attack = num_failures * 100.0 / num_results
    attack_succ = (original_accuracy - accuracy_under_attack) * 100.0 / original_accuracy
    out_csv = open(args.results_file, 'a', encoding='utf-8', newline="")
    csv_writer = csv.writer(out_csv)
    csv_writer.writerow([args.model_name_or_path, original_accuracy, accuracy_under_attack, attack_succ])
    out_csv.close()


if __name__ == "__main__":
    main()
