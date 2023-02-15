# Improving Sign Recognition With Phonology

This repository contains the necessary code to replicate the findings of "Improving Sign Recognition with Phonology".

The project extends the [OpenHands project](https://openhands.ai4bharat.org/en/latest/) in the following ways:
1. Adds phoneme types to the [WLASL dataset class](https://github.com/leekezar/ImprovingSignRecognitionWithPhonology/blob/4307ee692350b4c9c2baa0fc96b5644051a4cbea/openhands/datasets/isolated/wlasl.py#L28)
2. Adds the ability for [additional linear layers](https://github.com/leekezar/ImprovingSignRecognitionWithPhonology/blob/4307ee692350b4c9c2baa0fc96b5644051a4cbea/openhands/models/decoder/fc.py#L53) (classification heads) according to the "parameters" property in the config file.
3. Adds [enhanced performance evaluation metrics](https://github.com/leekezar/ImprovingSignRecognitionWithPhonology/blob/4307ee692350b4c9c2baa0fc96b5644051a4cbea/openhands/apis/inference.py#L59) to the InferenceModel

For instructions on how to utilize OpenHands, please see the [original repository](https://github.com/AI4Bharat/OpenHands).
An example config file is provided in the root directory ([gcn_top2_all_train.config](https://github.com/leekezar/ImprovingSignRecognitionWithPhonology/blob/main/gcn_top2_all_train.yaml), [bert_top2_all_train.config](https://github.com/leekezar/ImprovingSignRecognitionWithPhonology/blob/main/bert_top2_all_train.yaml)).

Citation:

    @inproceedings{KezarImprovingISLR,
      title = {Improving Sign Recognition with Phonology},
      author = {Kezar, Lee and Thomason, Jesse and Sehyr, Zed Sevcikova},
      publisher = {The 17th Conference of the European Chapter of the Association for Computational Linguistics (EACL)},
      year = {2023},
      url = {https://arxiv.org/abs/2302.05759}
    }
