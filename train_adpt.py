import omegaconf
from openhands.apis.classification_model import ClassificationModel
from openhands.core.exp_utils import get_trainer
import glob

import argparse
import pathlib

def learn_model(cfg):
    trainer = get_trainer(cfg)
    model = ClassificationModel(cfg=cfg, trainer=trainer)
    model.init_from_checkpoint_if_available()
    model.fit()
    import pdb; pdb.set_trace()

def parse_args():
    parser = argparse.ArgumentParser(description='Learn an SL-GCN model for ASL phonology')
    
    parser.add_argument('targets', nargs='+', metavar='PHONEME_TYPE(S)',
        help='The phoneme targets to predict. Options are: All (all) or 1+ of the following, separated by a space: Handshape (hs), Nondominant Handshape (nhs), Major Location (mjl), Minor Location (mnl), Second Minor Location (sml), Contact (con), Thumb Contact (tc), Thumb Position (tp), Wrist Twist (wt), Spread (spr), Spread Change (sc), Flexion (fx), Flexion Change (fc), Selected Fingers (sf), Path Movement (pm), Sign Type (st), Repeated Movement (rm)',
        choices=['all', 'hs', 'nds', 'mjl', 'mnl', 'sml', 'con', 'tc', 'tp', 'wt', 'spr', 'sc', 'fx', 'fc', 'sf', 'pm', 'st', 'rm'])
    
    parser.add_argument('-learn_adapter', '-a', action='store_true',
                help='When provided, the provided pretrained model will be frozen and adapters will be learned for the targets instead.')

    parser.add_argument('-pretrained', '-p', type=pathlib.Path, nargs=1, metavar='CKPT_FILE',
        help='The pretrained model to initialize. Must be a .ckpt file.')

    parser.add_argument('-fuse', '-f', type=pathlib.Path, nargs='+', metavar='ADPT_FILE',
        help='The pretrained adapters (.adpt) to include. Can use wildcards such as "./adapters/*/*.adpt".')

    parser.add_argument('-strategy', '-s', nargs=1,
        help='One of the following methods for learning the targets: One at a time (ind), all at once (multi), in a curriculum (curr). If a curriculum, the ordering of the provided targets will be the curriculum.',
        choices=['ind', 'multi', 'curr'])

    parser.add_argument('-output', '-o', type=pathlib.Path, nargs='?', metavar='FOLDER',
        help='The folder to save the best model. If multiple runs involved (e.g. strategy is "ind" and multiple targets selected), the target\'s name will be appended to the end for each one.')

    parser.add_argument('-wandb', '-w', nargs=1, metavar='NAME',
        help='When selected, the results will be logged on Weights and Biases. If multiple runs involved (e.g. strategy is "ind" and multiple targets selected), the target\'s name will be appended to the end for each one.')

    parser.add_argument('-epochs', '-e', type=int, nargs=1, metavar='N', default=100, 
        help='The number of epochs to train for. If strategy is "curr", then this will be the number of epochs per target.')

    parser.add_argument('-train', type=pathlib.Path, nargs=1, metavar='PATH', default='/misery/lee/asllex_large/poses/',
        help='The folder containing the poses for training.')

    parser.add_argument('-val', type=pathlib.Path, nargs=1, metavar='PATH', default='/misery/lee/asllex_large/poses/',
        help='The folder containing the poses for validation.')

    parser.add_argument('-test', type=pathlib.Path, nargs=1, metavar='PATH', default='/misery/lee/asllex_large/poses',
        help='The folder containing the poses for test. (Currently not functional!)')

    parser.add_argument('-metadata', '-m', type=pathlib.Path, nargs=1, metavar='PATH', default='/misery/lee/asllex_large/asllex_large.json',
        help='The .json file containing the ground truth labels and train/val splits.')

    parser.add_argument('-ngpus', '-g', type=int, nargs=1, metavar='N', default=1,
        help='The number of gpus to utilize.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cfg = omegaconf.OmegaConf.load("gcn_adapter_train.yaml")

    # Strategy-independent params
    cfg.pretrained = args.pretrained
    cfg.trainer.max_epochs = args.epochs
    cfg.exp_manager.checkpoint_callback_params.dirpath = args.output
    cfg.trainer.trainer.gpus = args.ngpus

    # Update the train/val data
    cfg.train_pipeline.dataset.split_file = args.metadata
    cfg.train_pipeline.dataset.root_dir = args.train
    cfg.val_pipeline.dataset.split_file = args.metadata
    cfg.val_pipeline.dataset.root_dir = args.val

    if args.learn_adapter:
        cfg.model.learn_adapter = True
        cfg.model.adapter_source = list(glob.glob(args.fuse))

    # Multi does not need the loop
    if args.strategy == 'multi':
        cfg.train_pipeline.parameters = args.targets
        cfg.val_pipeline.parameters = args.targets
        cfg.exp_manager.wandb_logger_kwargs.name = args.wandb

        learn_model(cfg)
        return
    
    for n,target in enumerate(args.targets):
        cfg.exp_manager.checkpoint_callback_params.dirpath += f"_{target}"

        if args.strategy == 'curr' and n>1:
            # instead of learning from the pretrained model, learn from the last saved model.
            cfg.pretrained = None
            cfg.trainer.resume_from_checkpoint = args.output + "_" + args.targets[n-1]
            cfg.trainer.max_epochs += args.epochs

        if args.wandb and args.strategy == 'curr':
            cfg.exp_manager.wandb_logger_kwargs.name += f"_{target}"

        learn_model(cfg)
