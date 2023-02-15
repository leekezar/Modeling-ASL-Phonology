import torch
import torch.nn.functional as F
import torchvision
import pickle
import json
import albumentations as A
import numpy as np
import pandas as pd
import os, warnings
from ..video_transforms import *
from ..data_readers import *

PARAMS = [
    "Handshape","Selected Fingers", "Flexion", "Spread", "Spread Change",
    "Thumb Position", "Thumb Contact", "Sign Type", "Path Movement",
    "Repeated Movement", "Major Location", "Minor Location",
    "Second Minor Location", "Contact", "Nondominant Handshape", 
    "Wrist Twist", "Handshape Morpheme 2"
]

class BaseIsolatedDataset(torch.utils.data.Dataset):
    """
    This module provides the datasets for Isolated Sign Language Classification.
    Do not instantiate this class
    """

    lang_code = None
    # Get language from here:
    # https://iso639-3.sil.org/code_tables/639/data?title=&field_iso639_cd_st_mmbrshp_639_1_tid=94671&name_3=sign+language&field_iso639_element_scope_tid=All&field_iso639_language_type_tid=All&items_per_page=200

    ASSETS_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

    def __init__(
        self,
        root_dir,
        split_file=None,
        class_mappings_file_path=None,
        normalized_class_mappings_file=None,
        splits=["train"],
        modality="rgb",
        transforms="default",
        cv_resize_dims=(264, 264),
        pose_use_confidence_scores=False,
        pose_use_z_axis=False,
        inference_mode=False,
        only_metadata=False, # Does not load data files if `True`
        multilingual=False,
        languages=None,
        language_set=None,
        results=None,
        
        # Windowing
        seq_len=1, # No. of frames per window
        num_seq=1, # No. of windows
    ):
        super().__init__()

        self.split_file = split_file
        self.root_dir = root_dir
        self.class_mappings_file_path = class_mappings_file_path
        self.splits = splits
        self.modality = modality
        self.multilingual = multilingual
        self.seq_len = seq_len
        self.num_seq = num_seq
        self.languages=languages
        self.language_set=language_set

        self.normalized_class_mappings_file = normalized_class_mappings_file
        if normalized_class_mappings_file:
            df = pd.read_csv(normalized_class_mappings_file, na_filter=False) # In German, "null" means "zero"
            self.normalized_class_mappings = {df["actual_gloss"][i]: df["normalized_gloss"][i] for i in range(len(df))}
            # TODO: Also store reverse mapping for inference in original lang
        
        self.glosses = []
        self.read_glosses()
        if not self.glosses:
            raise RuntimeError("Unable to read glosses list")
        print(f"Found {len(self.glosses)} classes in {splits} splits")

        self.gloss_to_id = {gloss: i for i, gloss in enumerate(self.glosses)}
        self.id_to_gloss = {i: gloss for i, gloss in enumerate(self.glosses)}

        self.video_id_to_gloss = {}
        for sign in json.load(open(self.split_file)):
            for instance in sign["instances"]:
                self.video_id_to_gloss[instance["video_id"]] = sign["gloss"]

        self.params = {}
        self.read_params()
        if self.params:
            for p, vals in self.params.items():
                print(f"Found {len(vals)} {p}s in {splits} splits")

        self.param_to_id = {param: { val : i for i, val in enumerate(vals) } for param, vals in self.params.items() }
        self.id_to_param = {param: { i : val for i, val in enumerate(vals) } for param, vals in self.params.items() }
        
        self.inference_mode = inference_mode
        self.only_metadata = only_metadata

        if not only_metadata:
            self.data = []
            
            if inference_mode:
                # Will have null labels
                self.enumerate_data_files(self.root_dir)
            else:
                self.read_original_dataset()
            if not self.data:
                raise RuntimeError("No data found")

        self.cv_resize_dims = cv_resize_dims
        self.pose_use_confidence_scores = pose_use_confidence_scores
        self.pose_use_z_axis = pose_use_z_axis

        if "rgb" in modality:
            self.in_channels = 3
            if modality == "rgbd":
                self.in_channels += 1

            self.__getitem = self.__getitem_video

        elif modality == "pose":
            self.in_channels = 4
            if not self.pose_use_confidence_scores:
                self.in_channels -= 1
            if not self.pose_use_z_axis:
                self.in_channels -= 1

            self.__getitem = self.__getitem_pose

        elif modality == "mixed":
            self.in_channels = 7
            self.__getitem = self.__getitem_mixed

        else:
            exit(f"ERROR: Modality `{modality}` not supported")

        self.setup_transforms(modality, transforms)

    def setup_transforms(self, modality, transforms):
        if "rgb" in modality or "mixed" in modality:
            if transforms == "default":
                albumentation_transforms = A.Compose(
                    [
                        A.ShiftScaleRotate(
                            shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                        ),
                        A.ChannelDropout(p=0.1),
                        A.RandomRain(p=0.1),
                        A.GridDistortion(p=0.3),
                    ]
                )
                self.transforms = torchvision.transforms.Compose(
                    [
                        Albumentations2DTo3D(albumentation_transforms),
                        NumpyToTensor(),
                        RandomTemporalSubsample(16),
                        torchvision.transforms.Resize(
                            (self.cv_resize_dims[0], self.cv_resize_dims[1])
                        ),
                        torchvision.transforms.RandomCrop(
                            (self.cv_resize_dims[0], self.cv_resize_dims[1])
                        ),
                        torchvision.transforms.RandomHorizontalFlip(p=0.5),
                        # torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        TCHW2CTHW(),
                    ]
                )
            elif transforms:
                self.transforms = transforms
            else:
                self.transforms = torchvision.transforms.Compose(
                    [
                        NumpyToTensor(),
                        # THWC2CTHW(),
                        THWC2TCHW(),
                        torchvision.transforms.Resize(
                            (self.cv_resize_dims[0], self.cv_resize_dims[1])
                        ),
                        TCHW2CTHW(),
                    ]
                )
        elif "pose" in modality:
            if transforms == "default":
                transforms = None
            self.transforms = transforms

    @property
    def num_class(self):
        return len(self.glosses)

    def num_param(self, param):
        return len(self.params[param])

    def read_glosses(self):
        """
        Implement this method to construct `self.glosses[]`
        """
        raise NotImplementedError
    
    def read_original_dataset(self):
        """
        Implement this method to read (video_name/video_folder, classification_label)
        into self.data[]
        """
        raise NotImplementedError
    
    def enumerate_data_files(self, dir):
        """
        Lists the video files from given directory.
        - If pose modality, generate `.pkl` files for all videos in folder.
          - If no videos present, check if some `.pkl` files already exist
        """
        files = list_all_videos(dir)

        if self.modality == "pose" or self.modality == "mixed":
            holistic = None
            pose_files = []

            for video_file in files:
                pose_file = os.path.splitext(video_file)[0] + ".pkl"
                if not os.path.isfile(pose_file):
                    # If pose is not cached, generate and store it.
                    if not holistic:
                        # Create MediaPipe instance
                        from ..pipelines.generate_pose import MediaPipePoseGenerator
                        holistic = MediaPipePoseGenerator()
                    # Dump keypoints
                    frames = load_frames_from_video(video_file)
                    holistic.generate_keypoints_for_frames(frames, pose_file)
                pose_files.append(pose_file)
            
            if not pose_files:
                pose_files = list_all_files(dir, extensions=[".pkl"])
            
            files = pose_files
        
        if not files:
            raise RuntimeError(f"No files found in {dir}")
        
        self.data = [(f, -1) for f in files]
        # -1 means invalid label_id

    def __len__(self):
        return len(self.data)

    def load_pose_from_path(self, path):
        """
        Load dumped pose keypoints.
        Should contain: {
            "keypoints" of shape (T, V, C),
            "confidences" of shape (T, V)
        }
        """
        # if "videos" in path:
        #     path = path.replace("videos","poses")

        pose_data = pickle.load(open(path, "rb"))
        return pose_data

    def read_video_data(self, index):
        """
        Extend this method for dataset-specific formats
        """
        video_path = self.data[index][0]
        label = self.data[index][1]
        imgs = load_frames_from_video(video_path)

        return imgs, label, video_name

    def __getitem_video(self, index):
        if self.inference_mode:
            imgs, label, video_id, params = super().read_video_data(index)
        else:
            imgs, label, video_id, params = self.read_video_data(index)
        # imgs shape: (T, H, W, C)

        if len(imgs) == 0:
            return None

        if self.transforms is not None:
            imgs = self.transforms(imgs)

        data = {
            "frames": imgs,
            "label": torch.tensor(label, dtype=torch.long),
            "file": video_id,
            "dataset_name": data["dataset_name"] if self.multilingual else None, # Required to calc dataset-wise accuracy
        }

        for p,val in params.items():
            data[p] = torch.tensor(val, dtype=torch.long)

        # print(f"\tVideo: {video_id}, gloss: {label}")
        return data

    def __getitem_mixed(self, index):
        # print(f"Getting {index}th item (mixed)")
        vid = self.__getitem_video(index)
        if vid:
            vid["poses"] = self.__getitem_pose(index)["frames"]
        return vid


    @staticmethod
    def collate_fn(batch_list):
        # test
        # if "num_windows" in batch_list[0]:
        #     # Padding not required for windowed models
        #     frames=[x["frames"] for x in batch_list]
        # else:
        batch_list = [b for b in batch_list if b]
        if not batch_list:
            return None

        # print(batch_list[0][0])
        # print(batch_list[0][1])

        # import pdb; pdb.set_trace()
        max_frames = max([x["frames"].shape[1] for x in batch_list if x])
        # Pad the temporal dimension to `max_frames` for all videos
        # Assumes each instance of shape: (C, T, V) 
        # TODO: Handle videos (C,T,H,W)
        frames = [
            F.pad(x["frames"], (0, 0, 0, max_frames - x["frames"].shape[1], 0, 0))
            for i, x in enumerate(batch_list) if x
        ]
        frames = torch.stack(frames, dim=0)

        poses = [
            F.pad(x["poses"], (0, 0, 0, max_frames - x["poses"].shape[1], 0, 0))
            for i,x in enumerate(batch_list) if x
        ]
        poses = torch.stack(poses, dim=0)

        labels = [x["label"] for i, x in enumerate(batch_list) if x]
        labels = torch.stack(labels, dim=0)
        # print(labels)
        if 'Handshape' in batch_list[0].keys():
            params = { p : torch.stack([x[p] for x in batch_list if x], dim=0) for p in PARAMS }
        else:
            params = {}

        return dict(frames=frames, poses=poses, labels=labels, params=params, files=[x["file"] for x in batch_list if x], dataset_names=[x["dataset_name"] for x in batch_list if x])

    def read_pose_data(self, index):
        label = self.data[index][1]
        if len(self.data[index]) > 2:
            params = self.data[index][2]

        if self.inference_mode:
            pose_path = self.data[index][0]
        else:
            video_name = self.data[index][0]
            
            video_path = os.path.join(self.root_dir, video_name)
            # print("--------------279",self.root_dir)
            # print("---------280",video_name)
            # If `video_path` is folder of frames from which pose was dumped, keep it as it is.
            # Otherwise, just remove the video extension
            pose_path = (
                video_path if os.path.isdir(video_path) else os.path.splitext(video_path)[0]
            )
            pose_path = pose_path.replace("videos","poses") + ".pkl"
        #print(pose_path)
        pose_data = self.load_pose_from_path(pose_path)

        pose_data["label"] = torch.tensor(label, dtype=torch.long)
        
        if not self.inference_mode:
            for p in self.params.keys():
                pose_data[p] = torch.tensor(params[p], dtype=torch.long)

        if self.multilingual:
            # if `ConcatDataset` is used, it has extra entries for following:
            pose_data["lang_code"] = self.data[index][2]
            pose_data["dataset_name"] = self.data[index][3]
        return pose_data, pose_path

    def __getitem_pose(self, index):
        """
        Returns
        C - num channels
        T - num frames
        V - num vertices
        """
        data, path = self.read_pose_data(index)
        # imgs shape: (T, V, C)
        kps = data["keypoints"]
        scores = data["confidences"]

        if not self.pose_use_z_axis:
            kps = kps[:, :, :2]

        if self.pose_use_confidence_scores:
            kps = np.concatenate([kps, np.expand_dims(scores, axis=-1)], axis=-1)

        kps = np.asarray(kps, dtype=np.float32)
        formatted_data = {
            "frames": torch.tensor(kps).permute(2, 0, 1),  # (C, T, V)
            "label": data["label"],
            "file": path,
            "lang_code": data["lang_code"] if self.multilingual else None, # Required for lang_token prepend
            "dataset_name": data["dataset_name"] if self.multilingual else None, # Required to calc dataset-wise accuracy
        }
        
        if not self.inference_mode:
            for p in self.params.keys():
                formatted_data[p] = data[p]
        data = formatted_data

        # temporarily remove this bc transforms is currently defined to be for videos
        # if self.transforms is not None:
        #     data = self.transforms(data)
        
        if self.seq_len > 1 and self.num_seq > 1:
            data["num_windows"] = self.num_seq
            kps = data["frames"].permute(1, 2, 0).numpy() # CTV->TVC
            if kps.shape[0] < self.seq_len * self.num_seq:
                pad_kps = np.zeros(
                    ((self.seq_len * self.num_seq) - kps.shape[0], *kps.shape[1:])
                )
                kps = np.concatenate([pad_kps, kps])

            elif kps.shape[0] > self.seq_len * self.num_seq:
                kps = kps[: self.seq_len * self.num_seq, ...]

            SL = kps.shape[0]
            clips = []
            i = 0
            while i + self.seq_len <= SL:
                clips.append(torch.tensor(kps[i : i + self.seq_len, ...], dtype=torch.float32))
                i += self.seq_len

            t_seq = torch.stack(clips, 0)
            data["frames"] = t_seq.permute(0, 3, 1, 2) # WTVC->WCTV

        label = data["label"]
        vid = path.split("/")[-1]
        # print(f"\tPose: {vid}, gloss: {label}")

        return data

    def __getitem__(self, index):
        return self.__getitem(index)
