import omegaconf
from openhands.apis.classification_model import ClassificationModel
from openhands.core.exp_utils import get_trainer
import glob

import os
import argparse
import pathlib

PTYPE_SHORTHAND_TO_FULL = {
    "hs": "Handshape",
    "sf": "Selected Fingers",
    "fx": "Flexion",
    "spr": "Spread",
    "sc": "Spread Change",
    "tp": "Thumb Position",
    "tc": "Thumb Contact",
    "st": "Sign Type",
    "pm": "Path Movement",
    "rm": "Repeated Movement",
    "mjl": "Major Location",
    "mnl": "Minor Location",
    "sml": "Second Minor Location",
    "con": "Contact",
    "nds": "Nondominant Handshape", 
    "wt": "Wrist Twist",
}

def learn_model(cfg, args):
    os.environ["WANDB_API_KEY"] = cfg.exp_manager.wandb_api_key
    trainer = get_trainer(cfg)
    model = ClassificationModel(cfg=cfg, trainer=trainer)
    model.init_from_checkpoint_if_available()

    # Create (and activate) Adapter modules, if any
    if cfg.model.learn_adapter is True:
        for t in args.targets:
            model.model.encoder.add_adapter(t)
        if args.learning_strategy == 'ind':
            model.model.encoder.activate_adapter(args.targets[0])

    # Train the model
    model.fit()

    # Load best validation checkpoint
    model.load_best_validation_checkpoint()

    # Save adapter params
    if args.learning_strategy == 'ind':
        model.model.encoder.save_adapter_weights(args.targets[0], cfg.model.adapters_dir)

    # Evaluate model on target ptype
    #if args.learning_strategy == 'ind':
    #    model.model.encoder.load_adapter_weights(args.targets[0], cfg.model.adapters_dir)
    #    model.model.encoder.activate_adapter(args.targets[0])
    #model.eval()
    #model.compute_test_accuracy()



def parse_args():
    parser = argparse.ArgumentParser(description='Learn an SL-GCN model for ASL phonology')
    
    parser.add_argument('--targets', nargs='+', metavar='PHONEME_TYPE(S)',
        help='The phoneme targets to predict. Options are: All (all) or 1+ of the following, separated by a space: Handshape (hs), Nondominant Handshape (nhs), Major Location (mjl), Minor Location (mnl), Second Minor Location (sml), Contact (con), Thumb Contact (tc), Thumb Position (tp), Wrist Twist (wt), Spread (spr), Spread Change (sc), Flexion (fx), Flexion Change (fc), Selected Fingers (sf), Path Movement (pm), Sign Type (st), Repeated Movement (rm)',
        choices=['all', 'hs', 'nds', 'mjl', 'mnl', 'sml', 'con', 'tc', 'tp', 'wt', 'spr', 'sc', 'fx', 'fc', 'sf', 'pm', 'st', 'rm'])

    parser.add_argument('--config_yaml', '-c', type=str, metavar='YAML_FILE',
        help='The config file for training. Must be a .yaml file.')

    parser.add_argument('--pretrained_model_ckpt', '-p', type=str, metavar='CKPT_FILE',
        help='The pretrained model to initialize. Must be a .ckpt file.')

    parser.add_argument('--learning_strategy', choices=['ind', 'multi', 'curr'],
        help='One of the following methods for learning the targets: One at a time (ind), all at once (multi), in a curriculum (curr). If a curriculum, the ordering of the provided targets will be the curriculum.')

    parser.add_argument('--learn_adapter', '-a', action='store_true',
                help='When provided, the provided pretrained model will be frozen and adapters will be learned for the targets instead. Requires that learning_strategy is ind.')

    #parser.add_argument('--adapter_strategy', type=str,
    #            help='Specifies to learn an adapter for target ptype independently, or using fusion.')

    #parser.add_argument('--adapters_dir', type=str,
    #            help='Specifies where adapter checkpoints are saved.')

    parser.add_argument('--fuse', '-f', type=pathlib.Path, nargs='+', default=None, metavar='ADPT_FILE',
        help='The pretrained adapters (.adpt) to include. Can use wildcards such as "./adapters/*/*.adpt".')


    parser.add_argument('--output', '-o', type=str, nargs='?', metavar='FOLDER',
        help='The folder to save the best model. If multiple runs involved (e.g. strategy is "ind" and multiple targets selected), the target\'s name will be appended to the end for each one.')

    parser.add_argument('--wandb', '-w', nargs=1, metavar='NAME',
        help='When selected, the results will be logged on Weights and Biases. If multiple runs involved (e.g. strategy is "ind" and multiple targets selected), the target\'s name will be appended to the end for each one.')

    parser.add_argument('-epochs', '-e', type=int, nargs=1, metavar='N', default=100, 
        help='The number of epochs to train for. If strategy is "curr", then this will be the number of epochs per target.')

    parser.add_argument('-train', type=str, metavar='PATH', default='/misery/lee/asllex_large/poses/',
        help='The folder containing the poses for training.')

    parser.add_argument('-val', type=str, metavar='PATH', default='/misery/lee/asllex_large/poses/',
        help='The folder containing the poses for validation.')

    parser.add_argument('-test', type=str, nargs=1, metavar='PATH', default='/misery/lee/asllex_large/poses',
        help='The folder containing the poses for test. (Currently not functional!)')

    parser.add_argument('-metadata', '-m', type=str, metavar='PATH', default='/misery/lee/asllex_large.json',
        help='The .json file containing the ground truth labels and train/val splits.')

    parser.add_argument('-ngpus', '-g', type=int, nargs=1, metavar='N', default=1,
        help='The number of gpus to utilize.')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    cfg = omegaconf.OmegaConf.load(args.config_yaml)

    # Strategy-independent params
    cfg.pretrained = args.pretrained_model_ckpt
    cfg.trainer.max_epochs = args.epochs
    cfg.exp_manager.checkpoint_callback_params.dirpath = args.output
    cfg.trainer.gpus = args.ngpus

    # Update the train/val data
    cfg.data.train_pipeline.dataset.split_file = args.metadata
    cfg.data.train_pipeline.dataset.root_dir = args.train
    cfg.data.valid_pipeline.dataset.split_file = args.metadata
    cfg.data.valid_pipeline.dataset.root_dir = args.val
    cfg.model.adapter_source = [] if args.fuse == None else list(glob.glob(args.fuse))

    if args.learn_adapter:
        cfg.model.learn_adapter = True
        cfg.model.adapter_source = [] if args.fuse == None else list(glob.glob(args.fuse))
        #cfg.model.adapters_dir = args.adapters_dir
        #cfg.model.adapter_strategy = args.adapter_strategy

        assert args.learning_strategy == 'ind'
        assert len(args.targets) == 1
    
    cfg.data.train_pipeline.parameters = [PTYPE_SHORTHAND_TO_FULL[t] for t in args.targets]
    cfg.data.valid_pipeline.parameters = [PTYPE_SHORTHAND_TO_FULL[t] for t in args.targets]
    # Multi does not need the loop
    if args.learning_strategy == 'multi':
        cfg.exp_manager.wandb_logger_kwargs.name = args.wandb

        learn_model(cfg, args)
        #return
    
    for n,target in enumerate(args.targets):
        cfg.exp_manager.checkpoint_callback_params.dirpath += f"_{target}"
        cfg.exp_manager.wandb_logger_kwargs.name += f"_{target}"
        if args.learning_strategy == 'curr' and n>1:
            # instead of learning from the pretrained model, learn from the last saved model.
            cfg.pretrained = None
            cfg.trainer.resume_from_checkpoint = args.output + "_" + args.targets[n-1]
            cfg.trainer.max_epochs += args.epochs

        #if args.wandb and args.learning_strategy == 'curr':
        #    cfg.exp_manager.wandb_logger_kwargs.name += f"_{target}"

        learn_model(cfg, args)
