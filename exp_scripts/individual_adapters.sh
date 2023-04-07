python -m train_adpt --targets ${1} \
                     --config_yaml gcn_individual_adapter_train.yaml \
                     --pretrained_model_ckpt /home/lee/wlasl/slgcn_adapters/pretrained_models/adapter_base.ckpt \
                     --learning_strategy ind \
                     --learn_adapter \
                     --output /data/experiments/ASL-Tejas/gcn_individual_adapter \
                     --wandb gcn_individual_adapter