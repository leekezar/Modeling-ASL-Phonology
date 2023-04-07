python -m train_adpt --targets ${1} \
                     --config_yaml gcn_individual_finetune_train.yaml \
                     --pretrained_model_ckpt /home/lee/wlasl/slgcn_adapters/pretrained_models/adapter_base.ckpt \
                     --learning_strategy ind \
                     --output /data/experiments/ASL-Tejas/gcn_individual_finetune \
                     --wandb gcn_individual_finetune