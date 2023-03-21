import omegaconf
from openhands.apis.classification_model import ClassificationModel
from openhands.core.exp_utils import get_trainer
import glob

PTYPES = [
    "Major Location", 
    "Minor Location",
    "Second Minor Location", 
    "Contact", 
    "Thumb Contact", 
    "Sign Type", 
    "Repeated Movement", 
    "Path Movement",
    "Wrist Twist", 
    "Spread", 
    "Flexion", 
    "Thumb Position",
	"Selected Fingers", 
    "Spread Change",
    "Nondominant Handshape", 
    "Handshape",
    # "Handshape Morpheme 2"
]

cfg = omegaconf.OmegaConf.load("curriculum_train.yaml")

for i, ptype in enumerate(PTYPES):
	cfg.data.train_pipeline.parameters.append(ptype)
	cfg.data.valid_pipeline.parameters.append(ptype)

	if i > 0:
		cfg.pretrained = False
		prev_ckpt = glob.glob("./pretrained_models/" + PTYPES[i-1].lower().replace(" ","") + "/*.ckpt")
		cfg.trainer.resume_from_checkpoint = prev_ckpt[0]
		cfg.trainer.max_epochs += 1

	cfg.exp_manager.checkpoint_callback_params.dirpath = "./pretrained_models/" + \
		ptype.lower().replace(" ","")

	trainer = get_trainer(cfg)
	model = ClassificationModel(cfg=cfg, trainer=trainer)
	model.init_from_checkpoint_if_available()
	model.fit()