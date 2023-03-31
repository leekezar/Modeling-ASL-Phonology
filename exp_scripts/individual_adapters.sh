python -m train_adpt --targets ${1} \
                     --pretrained_model_ckpt /home/lee/wlasl/slgcn_adapters/pretrained_models/adapter_base.ckpt \
                     --learning_strategy ind \
                     --learn_adapter \
                     --adapter_strategy vanilla \
                     --adapters_dir /home/lee/wlasl/slgcn_adapters/pretrained_models/vanilla_adapters/ \
                     --output /data/experiments/ASL-Tejas \
                     --wandb