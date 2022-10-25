# Improving Sign Recognition With Phonology

This repository contains the necessary code to replicate the findings of "Improving Sign Recognition with Phonology".

The project extends the OpenHands project in the following ways:
1. Adds phoneme types to the WLASL dataset class
2. Adds the ability for additional linear layers (classification heads) according to the "parameters" property in the config file.
3. Adds enhanced performance evaluation metrics to the InferenceModel

For instructions on how to utilize OpenHands, please see the [original repository](https://github.com/AI4Bharat/OpenHands).
An example config file is provided in the root directory ([https://github.com/leekezar/ImprovingSignRecognitionWithPhonology/blob/main/gcn_top2_all_train.yaml](gcn_top2_all_train.config), [https://github.com/leekezar/ImprovingSignRecognitionWithPhonology/blob/main/bert_top2_all_train.yaml](bert_top2_all_train.config)).
