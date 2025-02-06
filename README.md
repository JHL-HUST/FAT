This is the code for paper [Synonym-unaware Fast Adversarial Training against Textual Adversarial Attacks (NAACL Findings 2025)](https://arxiv.org/abs/2401.12461).

## Experiment Example
1. Training the BERT model with FAT method on the IMDB dataset:
    ```shell
    sh ./train/run_fat.sh {your device}
    ```

2. Training the BERT model with PGD-AT method on the IMDB dataset:
    ```shell
    sh ./train/run_pgd_at.sh {your device}
    ```

3. Attacking the model:
    ```shell
    sh ./attack/run_attack.sh {your device} {attak method(textfooler/bae/textbugger)} {model_name(folder name under the path ./train/saved_models)}
    ```

## Acknowledgment
This repository benefits from [Flooding-X](https://github.com/qinliu9/Flooding-X).
