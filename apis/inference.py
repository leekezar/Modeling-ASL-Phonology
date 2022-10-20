import torch
import pytorch_lightning as pl
from tqdm import tqdm
import time, json
import numpy as np

from ..core.data import DataModule
from ..models.loader import get_model
from sklearn.metrics import confusion_matrix
import numpy as np

# merge with the corresponding modules in the future release.
class InferenceModel(pl.LightningModule):
    """
    This will be the general interface for running the inference across models.
    Args:
        cfg (dict): configuration set.

    """
    def __init__(self, cfg, stage="test"):
        super().__init__()
        self.cfg = cfg
        self.datamodule = DataModule(cfg.data)
        self.datamodule.setup(stage=stage)

        self.model = self.create_model(cfg.model)
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if stage == "test":
            self.model.to(self._device).eval()
    
    def create_model(self, cfg):
        """
        Creates and returns the model object based on the config.
        """
        params = { p : self.datamodule.num_param[p] for p in cfg.decoder.parameters }

        return get_model(cfg, self.datamodule.in_channels, 
            self.datamodule.num_class, params)
    
    def forward(self, x):
        """
        Forward propagates the inputs and returns the model output.
        """
        return self.model(x)
    
    def init_from_checkpoint_if_available(self, map_location=torch.device("cpu")):
        """
        Intializes the pretrained weights if the ``cfg`` has ``pretrained`` parameter.
        """
        if "pretrained" not in self.cfg.keys():
            return

        ckpt_path = self.cfg["pretrained"]
        print(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=map_location)
        self.load_state_dict(ckpt["state_dict"], strict=False)
        del ckpt

    def test_inference(self):
        """
        Calculates the time taken for inference for all the batches in the test dataloader.
        """
        # TODO: Write output to a csv
        results = []
        splits = json.load(open("/home/lee/wlasl/sl-gcn/data/merged_data.json"))
        path2gloss = { instance["video_id"] : sign["gloss"] for sign in splits for instance in sign["instances"]}
        path2split = { instance["video_id"] : instance["split"] for sign in splits for instance in sign["instances"]}
        path2islex = { instance["video_id"] : instance["Handshape"]!=-1 for sign in splits for instance in sign["instances"]}
        id2gloss = self.datamodule.test_dataset.id_to_gloss
        gloss2id = self.datamodule.test_dataset.gloss_to_id
        dataloader = self.datamodule.test_dataloader()

        results_stats = {
            "all": {
                "a1":[],
                "a3":[],
                "rank":[],
                "reciprocal_rank":[]
            },
            "asllex": {
                "a1":[],
                "a3":[],
                "rank":[],
                "reciprocal_rank":[]
            },
            "nonlex": {
                "a1":[],
                "a3":[],
                "rank":[],
                "reciprocal_rank":[]
            }
        }

        for batch in dataloader:
            y_hat = self.model(batch["frames"].to(self._device))
            y_true = [path2gloss[path.split("/")[-1].replace(".pkl","")] \
                if path.split("/")[-1].replace(".pkl","") in path2gloss.keys() else None
                for path in batch["files"] ]

            y_hat_gloss = y_hat[0].cpu()
            y_true_gloss = [gt for gt in y_true]

            # y_hat_params = { p : v.cpu() for p,v in y_hat[1].items() }
            # y_true_params = { p : [gt["params"][p] for gt in y_true] for p in y_hat[1].items() }
            
            for sample_idx, gloss_probs in enumerate(y_hat_gloss):
                if not y_true[sample_idx]: continue
                sample_preds = {id2gloss[i] : prob for i,prob in enumerate(gloss_probs)}
                rankings, probs = zip(*sorted(sample_preds.items(), key=lambda x: x[1], reverse=True))
                if y_true[sample_idx] not in rankings: continue
                row = {
                    "id" : batch["files"][sample_idx].split("/")[-1].replace(".pkl",""),
                    "true" : y_true[sample_idx],
                    # "true_i": sample_idx,
                    "pred": rankings[0],
                    # "pred_i": torch.argmax(gloss_probs, dim=-1),
                    "top10" : rankings[:10],
                    "a1" : rankings[0] == y_true[sample_idx],
                    "a3" : y_true[sample_idx] in rankings[:3],
                    "rank" : rankings.index(y_true[sample_idx])
                }
                if path2split[row["id"]] != "test" : continue

                results_stats["all"]["a1"].append(row["a1"])
                results_stats["all"]["a3"].append(row["a3"])
                results_stats["all"]["rank"].append(row["rank"]+1)
                results_stats["all"]["reciprocal_rank"].append(1/(row["rank"]+1))
                
                if path2islex[row["id"]]:
                    results_stats["asllex"]["a1"].append(row["a1"])
                    results_stats["asllex"]["a3"].append(row["a3"])
                    results_stats["asllex"]["rank"].append(row["rank"]+1)
                    results_stats["asllex"]["reciprocal_rank"].append(1/(row["rank"]+1))                    
                else:
                    results_stats["nonlex"]["a1"].append(row["a1"])
                    results_stats["nonlex"]["a3"].append(row["a3"])
                    results_stats["nonlex"]["rank"].append(row["rank"]+1)
                    results_stats["nonlex"]["reciprocal_rank"].append(1/(row["rank"]+1))


                results.append(row)
                print(row)


        for split in results_stats.keys():
            for metric in results_stats[split].keys():
                results_stats[split][metric] = np.average(results_stats[split][metric])

        print(results_stats["all"])
        print(results_stats["asllex"])
        print(results_stats["nonlex"])
        json.dump(results, open(self.cfg.data.test_pipeline.dataset.results,"w+"), indent=4)
        json.dump(results_stats, open(self.cfg.data.test_pipeline.dataset.results[:-5] + "_aggregate.json","w+"), indent=4)

    def compute_test_accuracy(self):
        """
        Computes the accuracy for the test dataloader.
        """
        # Ensure labels are loaded
        assert not self.datamodule.test_dataset.inference_mode
        # TODO: Write output to a csv
        dataloader = self.datamodule.test_dataloader()
        dataset_scores, class_scores = {}, {}
        for batch_idx, batch in tqdm(enumerate(dataloader), unit="batch"):
            y_hat = self.model(batch["frames"].to(self._device)).cpu()
            class_indices = torch.argmax(y_hat, dim=-1)
            for i, (pred_index, gt_index) in enumerate(zip(class_indices, batch["labels"])):

                dataset_name = batch["dataset_names"][i]
                score = pred_index == gt_index
                
                if dataset_name not in dataset_scores:
                    dataset_scores[dataset_name] = []
                dataset_scores[dataset_name].append(score)

                if gt_index not in class_scores:
                    class_scores[gt_index] = []
                class_scores[gt_index].append(score)
        
        
        for dataset_name, score_array in dataset_scores.items():
            dataset_accuracy = sum(score_array)/len(score_array)
            print(f"Accuracy for {len(score_array)} samples in {dataset_name}: {dataset_accuracy*100}%")


        classwise_accuracies = {class_index: sum(scores)/len(scores) for class_index, scores in class_scores.items()}
        avg_classwise_accuracies = sum(classwise_accuracies.values()) / len(classwise_accuracies)

        print(f"Average of class-wise accuracies: {avg_classwise_accuracies*100}%")
    
    def compute_test_avg_class_accuracy(self):
        """
        Computes the accuracy for the test dataloader.
        """
        #Ensure labels are loaded
        assert not self.datamodule.test_dataset.inference_mode
        # TODO: Write output to a csv
        dataloader = self.datamodule.test_dataloader()
        scores = []
        all_class_indices=[]
        all_batch_labels=[]
        for batch_idx, batch in tqdm(enumerate(dataloader),unit="batch"):
            y_hat = self.model(batch["frames"].to(self._device)).cpu()
            class_indices = torch.argmax(y_hat, dim=-1)

            for i in range(len(batch["labels"])):
                all_batch_labels.append(batch["labels"][i])
                all_class_indices.append(class_indices[i])
            for pred_index, gt_index in zip(class_indices, batch["labels"]):
                scores.append(pred_index == gt_index)
        cm = confusion_matrix(np.array(all_batch_labels), np.array(all_class_indices))
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print(f"Average Class Accuracy for {len(all_batch_labels)} samples: {np.mean(cm.diagonal())*100}%")
