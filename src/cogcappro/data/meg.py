import argparse
import json
import torch, os
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import logging
import open_clip
import gc
from tqdm import tqdm
import itertools
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from pytorch_lightning import seed_everything

from ..utils import instantiate_from_config, get_device, update_config


def load_data(config, shuffle_train=True):
    exp_setting = config.get('exp_setting', 'intra-subject')
    num_gpus = len(config['devices'])  # Get number of GPUs
    rank = getattr(pl.utilities.rank_zero.rank_zero_only, 'rank', 0)

    if exp_setting == 'intra-subject':
        test_dataset = MEGDataset(config, mode='test')
        if num_gpus > 1:
            test_dataset = MEGDatasetDistributed(test_dataset, num_gpus)
        print('init test_dataset success')

        train_dataset = MEGDataset(config, mode='train')
        print('init train_dataset success')

        train_sampler = DistributedSampler(train_dataset, num_replicas=num_gpus, rank=rank, shuffle=shuffle_train)

        # Use distributed sampler to create DataLoader
        test_loader = DataLoader(test_dataset, batch_size=200,
                                 drop_last=False, num_workers=0, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['per_gpu_train_batch_size'],
                                  sampler=train_sampler,
                                  drop_last=False, num_workers=0, pin_memory=True)
        return train_loader, test_loader, test_loader

    elif exp_setting == 'inter-subject':
        subjects = config['data']['subjects']
        test_dataset = MEGDataset(config, mode='test')  # 这个最后是新的
        print('init test_dataset success')

        all_subjects = [f'sub-{i:02}' for i in range(1, 5)]
        leave_one_subjects = list(set(all_subjects) - set(subjects))
        leave_one_subjects_config = config
        leave_one_subjects_config['data']['subjects'] = leave_one_subjects
        val_dataset = MEGDataset(leave_one_subjects_config, mode='test')  # This one is new at the end
        print('init val_dataset success')
        train_dataset = MEGDataset(leave_one_subjects_config, mode='train')
        print('init train_dataset success')


        train_sampler = DistributedSampler(train_dataset, num_replicas=num_gpus, rank=rank, shuffle=shuffle_train)

        # Use distributed sampler to create DataLoader
        test_loader = DataLoader(test_dataset, batch_size=config['data']['per_gpu_test_batch_size'],
                                 drop_last=False, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['data']['per_gpu_val_batch_size'],
                                drop_last=False, num_workers=0, pin_memory=True)
        train_loader = DataLoader(train_dataset, batch_size=config['data']['per_gpu_train_batch_size'],
                                  sampler=train_sampler,
                                  drop_last=False, num_workers=0, pin_memory=True)
        return train_loader, val_loader, test_loader


class MEGDataset(Dataset):
    def __init__(self, config, mode):
        self.config = config
        self.data_dir = config['data']['data_dir']
        self.data_path = Path(self.data_dir).expanduser()
        self.repo_root = Path(__file__).resolve().parents[3]
        # self.img_directory = os.path.join(self.data_dir,'../','Image_set_Resize',f'{mode}_images')
        # self.all_class_names = [d.split('_',1)[-1] for d in os.listdir(self.img_directory) if os.path.isdir(os.path.join(self.img_directory, d))]
        # self.all_class_names.sort()
        self.subjects = config['data']['subjects']
        print(f'subjects:{self.subjects}')
        self.mode = mode
        self.name = config['name']
        self.model_type = config['data']['model_type']
        self.selected_ch = config['data']['selected_ch']
        self.channels = ['MLC11-1609', 'MLC12-1609', 'MLC13-1609', 'MLC14-1609', 'MLC15-1609', 'MLC16-1609',
                         'MLC17-1609', 'MLC21-1609', 'MLC22-1609', 'MLC23-1609', 'MLC24-1609', 'MLC25-1609',
                         'MLC31-1609', 'MLC32-1609', 'MLC41-1609', 'MLC42-1609', 'MLC51-1609', 'MLC52-1609',
                         'MLC53-1609', 'MLC54-1609', 'MLC55-1609', 'MLC61-1609', 'MLC62-1609', 'MLC63-1609',
                         'MLF11-1609', 'MLF12-1609', 'MLF13-1609', 'MLF14-1609', 'MLF21-1609', 'MLF22-1609',
                         'MLF23-1609', 'MLF24-1609', 'MLF31-1609', 'MLF32-1609', 'MLF33-1609', 'MLF34-1609',
                         'MLF35-1609', 'MLF41-1609', 'MLF42-1609', 'MLF43-1609', 'MLF44-1609', 'MLF45-1609',
                         'MLF46-1609', 'MLF51-1609', 'MLF52-1609', 'MLF53-1609', 'MLF54-1609', 'MLF55-1609',
                         'MLF56-1609', 'MLF61-1609', 'MLF62-1609', 'MLF63-1609', 'MLF64-1609', 'MLF65-1609',
                         'MLF66-1609', 'MLF67-1609', 'MLO11-1609', 'MLO12-1609', 'MLO13-1609', 'MLO14-1609',
                         'MLO21-1609', 'MLO22-1609', 'MLO23-1609', 'MLO24-1609', 'MLO31-1609', 'MLO32-1609',
                         'MLO33-1609', 'MLO34-1609', 'MLO41-1609', 'MLO42-1609', 'MLO43-1609', 'MLO44-1609',
                         'MLO51-1609', 'MLO52-1609', 'MLO53-1609', 'MLP11-1609', 'MLP12-1609', 'MLP21-1609',
                         'MLP22-1609', 'MLP23-1609', 'MLP31-1609', 'MLP32-1609', 'MLP33-1609', 'MLP34-1609',
                         'MLP35-1609', 'MLP41-1609', 'MLP42-1609', 'MLP43-1609', 'MLP44-1609', 'MLP45-1609',
                         'MLP51-1609', 'MLP52-1609', 'MLP53-1609', 'MLP54-1609', 'MLP55-1609', 'MLP56-1609',
                         'MLP57-1609', 'MLT11-1609', 'MLT12-1609', 'MLT13-1609', 'MLT14-1609', 'MLT15-1609',
                         'MLT16-1609', 'MLT21-1609', 'MLT22-1609', 'MLT23-1609', 'MLT24-1609', 'MLT25-1609',
                         'MLT26-1609', 'MLT27-1609', 'MLT31-1609', 'MLT32-1609', 'MLT33-1609', 'MLT34-1609',
                         'MLT35-1609', 'MLT36-1609', 'MLT37-1609', 'MLT41-1609', 'MLT42-1609', 'MLT43-1609',
                         'MLT44-1609', 'MLT45-1609', 'MLT46-1609', 'MLT47-1609', 'MLT51-1609', 'MLT52-1609',
                         'MLT53-1609', 'MLT54-1609', 'MLT55-1609', 'MLT56-1609', 'MLT57-1609', 'MRC11-1609',
                         'MRC12-1609', 'MRC13-1609', 'MRC14-1609', 'MRC15-1609', 'MRC16-1609', 'MRC17-1609',
                         'MRC21-1609', 'MRC22-1609', 'MRC23-1609', 'MRC24-1609', 'MRC25-1609', 'MRC31-1609',
                         'MRC32-1609', 'MRC41-1609', 'MRC42-1609', 'MRC51-1609', 'MRC52-1609', 'MRC53-1609',
                         'MRC54-1609', 'MRC55-1609', 'MRC61-1609', 'MRC62-1609', 'MRC63-1609', 'MRF11-1609',
                         'MRF12-1609', 'MRF13-1609', 'MRF14-1609', 'MRF21-1609', 'MRF22-1609', 'MRF23-1609',
                         'MRF24-1609', 'MRF25-1609', 'MRF31-1609', 'MRF32-1609', 'MRF33-1609', 'MRF34-1609',
                         'MRF35-1609', 'MRF41-1609', 'MRF42-1609', 'MRF44-1609', 'MRF45-1609', 'MRF46-1609',
                         'MRF51-1609', 'MRF52-1609', 'MRF53-1609', 'MRF54-1609', 'MRF55-1609', 'MRF56-1609',
                         'MRF61-1609', 'MRF62-1609', 'MRF63-1609', 'MRF64-1609', 'MRF65-1609', 'MRF66-1609',
                         'MRF67-1609', 'MRO12-1609', 'MRO14-1609', 'MRO21-1609', 'MRO22-1609', 'MRO23-1609',
                         'MRO24-1609', 'MRO31-1609', 'MRO32-1609', 'MRO33-1609', 'MRO34-1609', 'MRO41-1609',
                         'MRO42-1609', 'MRO43-1609', 'MRO44-1609', 'MRO51-1609', 'MRO52-1609', 'MRO53-1609',
                         'MRP11-1609', 'MRP12-1609', 'MRP21-1609', 'MRP22-1609', 'MRP23-1609', 'MRP31-1609',
                         'MRP32-1609', 'MRP33-1609', 'MRP34-1609', 'MRP35-1609', 'MRP41-1609', 'MRP42-1609',
                         'MRP43-1609', 'MRP44-1609', 'MRP45-1609', 'MRP51-1609', 'MRP52-1609', 'MRP53-1609',
                         'MRP54-1609', 'MRP55-1609', 'MRP56-1609', 'MRP57-1609', 'MRT11-1609', 'MRT12-1609',
                         'MRT13-1609', 'MRT14-1609', 'MRT15-1609', 'MRT16-1609', 'MRT21-1609', 'MRT22-1609',
                         'MRT23-1609', 'MRT24-1609', 'MRT25-1609', 'MRT26-1609', 'MRT27-1609', 'MRT31-1609',
                         'MRT32-1609', 'MRT33-1609', 'MRT34-1609', 'MRT35-1609', 'MRT36-1609', 'MRT37-1609',
                         'MRT41-1609', 'MRT42-1609', 'MRT43-1609', 'MRT44-1609', 'MRT45-1609', 'MRT46-1609',
                         'MRT47-1609', 'MRT51-1609', 'MRT52-1609', 'MRT53-1609', 'MRT54-1609', 'MRT55-1609',
                         'MRT56-1609', 'MRT57-1609', 'MZC01-1609', 'MZC02-1609', 'MZC03-1609', 'MZC04-1609',
                         'MZF01-1609', 'MZF02-1609', 'MZF03-1609', 'MZO01-1609', 'MZO02-1609', 'MZO03-1609',
                         'MZP01-1609']

        categorized_channels = categorize_channels_by_third_letter(self.channels)
        for letter, channel_list in categorized_channels.items():
            print(f"Number of channels with third letter {letter}: {len(channel_list)}")
            print(f"Channel list: {channel_list}")

        frontal_ch = categorized_channels['F']
        temporal_ch = categorized_channels['T']
        central_ch = categorized_channels['C']
        parietal_ch = categorized_channels['P']
        occipital_ch = categorized_channels['O']
        self_ch = []    # select your own channel here

        if self.selected_ch is True:
            self.selected_ch = ['TODO']
        elif isinstance(self.selected_ch, str):
            # If selected_ch is a string, it represents selecting a specific brain region
            region_map = {
                'frontal': frontal_ch,
                'temporal': temporal_ch,
                'central': central_ch,
                'parietal': parietal_ch,
                'occipital': occipital_ch,
                'self': self_ch
            }
            if self.selected_ch in region_map:
                self.selected_ch = region_map[self.selected_ch]
            else:
                self.selected_ch = self.channels
        else:
            self.selected_ch = self.channels

        self.avg = config['data'][f"{mode}_avg"]

        self.blur_type = config['data']['blur_type']

        self.timesteps = config['data']['timesteps']

        self.n_cls = 1654 if self.mode == 'train' else 200
        self.per_trials = 4 if self.mode == 'train' else 80

        self.data_paths = [os.path.join(self.data_dir, subject, f'{mode}.pt') for subject in self.subjects]
        self.loaded_data = [self.load_data(data_path) for data_path in self.data_paths]

        self.img_path_to_idx = {}  # Dictionary mapping path to index
        self.idx_counter = 0  # Index counter
        # Traverse all subjects and trials to collect img_path
        for subject_data in self.loaded_data:
            img_paths = subject_data['img']  # Get all trial img paths for this subject
            for path in img_paths:
                if path not in self.img_path_to_idx:  # Assign index for new paths
                    self.img_path_to_idx[path] = self.idx_counter
                    self.idx_counter += 1

        self.trial_subject = self.loaded_data[0]['eeg'].shape[0]
        self.trial_all_subjects = self.trial_subject * len(self.subjects)

        data_dir = self.data_path.parent / 'Image_feature' / 'FoveaBlur'
        data_dir.mkdir(parents=True, exist_ok=True)

        # Note:
        # f"{self.name}_{mode}.pt" is normalized
        # f"{self.name}_{mode}_unnorm.pt" is unnormalized

        # Use the same feature file for different brain encoders
        base_name = self.name.replace(f"_{self.config['brain_backbone']}", "")
        features_filename = data_dir / f"{base_name}_{mode}.pt"

        self.c = config['c']
        if self.config['data']['uncertainty_aware']:
            self.blur_transform = {}
            for shift, tag in zip([-self.c, 0, self.c], ['low', 'medium', 'high']):
                blur_param = config['data']['blur_type']
                blur_param['params']['blur_kernel_size'] = blur_param['params']['blur_kernel_size'] + shift
                self.blur_transform[tag] = instantiate_from_config(blur_param)
        else:
            self.blur_transform = instantiate_from_config(config['data']['blur_type'])
        process_term = [transforms.ToTensor(), transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(
            0.26862954, 0.26130258, 0.27577711))]  # transforms.Resize(pretrain_map[self.model_type]['resize']),
        self.process_transform = transforms.Compose(process_term)

        self.match_label = {'image': np.ones(self.trial_all_subjects, dtype=int),
                            'text': np.ones(self.trial_all_subjects, dtype=int),
                            'depth': np.ones(self.trial_all_subjects, dtype=int),
                            'edge': np.ones(self.trial_all_subjects, dtype=int)}

        # New: Build mapping from image path to sample indices (one image may correspond to multiple samples)
        self.img_to_sample_indices = {}  # img_path -> list of sample indices
        self._build_img_to_sample_mapping()

        # New: If in training mode, load similarity data required for clustering
        self.clusters = None
        self.similarity_dir = config.get('similarity_dir', None)

        if features_filename.exists():
            saved_features = torch.load(features_filename)
            self.img_features = saved_features['img_features']  # Organization: dict: 16540
            self.depth_features = saved_features['depth_features']
            self.edge_features = saved_features['edge_features']
            self.text_features = saved_features['text_features']
        else:
            device = get_device('auto')

            pre_trained_pth = dict(self.config.get('paths', {}).get('clip_weights', {}))
            if self.model_type not in pre_trained_pth:
                raise KeyError(f"Missing clip weight path for model type: {self.model_type}")
            self.vlmodel, self.preprocess, _ = open_clip.create_model_and_transforms(
                self.model_type,
                device=f"cuda:{device}",
                pretrained=pre_trained_pth[self.model_type])

            for param in self.vlmodel.parameters():
                param.requires_grad = False
            self.vlmodel.eval()
            self.text_features = self.Textencoder(self.loaded_data[0]['text'])

            if self.config['data']['uncertainty_aware']:
                self.img_features = {}
                self.depth_features = {}
                self.edge_features = {}
                for tag in ['low', 'medium', 'high']:
                    self.img_features[tag] = self.ImageEncoder(self.loaded_data[0]['img'], self.blur_transform[tag])
                    self.depth_features[tag] = self.ImageEncoder(self.loaded_data[0]['img'], self.blur_transform[tag],
                                                                 modality='depth_')
                    self.edge_features[tag] = self.ImageEncoder(self.loaded_data[0]['img'], self.blur_transform[tag],
                                                                modality='edge_')

                self.img_features['avg'] = {k: (sum(self.img_features[tag][k] for tag in ['low', 'medium', 'high']) / 3)
                                            for k in self.img_features['medium']}
                self.depth_features['avg'] = {
                    k: (sum(self.depth_features[tag][k] for tag in ['low', 'medium', 'high']) / 3)
                    for k in self.depth_features['medium']}
                self.edge_features['avg'] = {
                    k: (sum(self.edge_features[tag][k] for tag in ['low', 'medium', 'high']) / 3)
                    for k in self.edge_features['medium']}
            else:
                self.img_features = self.ImageEncoder(self.loaded_data[0]['img'])
                self.depth_features = self.ImageEncoder(self.loaded_data[0]['img'], modality='depth_')
                self.edge_features = self.ImageEncoder(self.loaded_data[0]['img'], modality='edge_')
            torch.save({
                'text_features': self.text_features,
                'img_features': self.img_features,
                'depth_features': self.depth_features,
                'edge_features': self.edge_features
            }, str(features_filename))  # 2025.4.16 Currently the features here are all after blur

            del self.vlmodel
            torch.cuda.empty_cache()
            gc.collect()

    def _find_blip2_dir(self):
        data_root = self.config.get('paths', {}).get('data_root')
        candidates = [
            self.repo_root / 'weights' / 'texts' / 'meg',
        ]
        if data_root:
            candidates.append(Path(data_root) / 'THINGS-MEG' / 'Image_text_description')

        for candidate in candidates:
            if (candidate / f'texts_BLIP2_{self.mode}.npy').exists():
                return candidate
        return None

    def load_data(self, data_path):
        logging.info(f"----load {data_path.rsplit('1000HZ', 1)[-1]}----")
        loaded_data = torch.load(data_path)
        # Change 'eeg' to 'eeg' to match MEG data
        loaded_data['eeg'] = torch.from_numpy(loaded_data['eeg'])

        if self.selected_ch:
            selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
            loaded_data['eeg'] = loaded_data['eeg'][:, :, selected_idx]

        ### add blip2 generated text    
        blip2_path = self._find_blip2_dir()
        if blip2_path is not None:
            self.texts_BLIP2 = np.load(blip2_path / f'texts_BLIP2_{self.mode}.npy')
            self.texts_BLIP2 = self.texts_BLIP2[::2]
            
        if self.avg:  # Take average across trials
            avg_data = {}
            avg_data['eeg'] = loaded_data['eeg'].mean(axis=1)
            avg_data['label'] = loaded_data['label']
            avg_data['img'] = np.array(loaded_data['img'])
            avg_data['text'] = self.texts_BLIP2 if hasattr(self, 'texts_BLIP2') else loaded_data['text'][:,
                                                                                     0] 
            avg_data['session'] = loaded_data['session']
            avg_data['times'] = loaded_data['times']
            loaded_data = avg_data
        else:
            _data = {}
            _data['eeg'] = loaded_data['eeg'].reshape(-1, *loaded_data['eeg'].shape[2:])
            _data['meg_avg'] = loaded_data['eeg'].mean(axis=1)
            _data['label'] = loaded_data['label'].reshape(-1)
            _data['img'] = loaded_data['img'].reshape(-1)
            _data['text'] = np.tile(self.texts_BLIP2, self.per_trials) if hasattr(self, 'texts_BLIP2') else loaded_data[
                'text'].reshape(-1) 
            _data['session'] = loaded_data['session'].reshape(-1)
            _data['times'] = loaded_data['times']
            loaded_data = _data

        for k, v in loaded_data.items():
            if k in ['eeg', 'label', 'img', 'text', 'session']:
                logging.info(f"{k}: {v.shape}")

        return loaded_data

    def _build_img_to_sample_mapping(self):
        """Build mapping from image path to sample indices in dataset"""
        self.img_to_sample_indices = {}
        # Traverse all samples and record the sample indices corresponding to each image
        for sample_idx in range(self.__len__()):
            # Get the image path for this sample
            subject = sample_idx // self.trial_subject
            trial_index = sample_idx % self.trial_subject
            img_path = self.loaded_data[subject]['img'][trial_index]

            if img_path not in self.img_to_sample_indices:
                self.img_to_sample_indices[img_path] = []
            self.img_to_sample_indices[img_path].append(sample_idx)

    def image_generator(self, img_paths, blur_transform, modality=''):
        """Generator: Open, process and close images one by one"""
        for img_path in img_paths:
            base_path = self.data_path.parent / f'Image_{modality}set_Resize'
            full_path = base_path / img_path
            
            # Try different image formats
            if not full_path.exists():
                # Try replacing extension
                name, ext = os.path.splitext(img_path)
                if ext.lower() in ['.jpg', '.jpeg']:
                    # Try png format
                    full_path = base_path / f'{name}.png'
                elif ext.lower() == '.png':
                    # Try jpg format
                    full_path = base_path / f'{name}.jpg'
                # If still doesn't exist, try jpeg format
                if not full_path.exists():
                    full_path = base_path / f'{name}.jpeg'

            # Open, process and close image
            try:
                img = Image.open(full_path).convert("RGB")
                # When using DirectT, no additional transformation is needed
                if hasattr(blur_transform, '__class__') and blur_transform.__class__.__name__ == 'DirectT':
                    processed_img = self.process_transform(img)
                else:
                    processed_img = self.process_transform(blur_transform(img))
                img.close()  # Explicitly close image
                yield processed_img
            except Exception as e:
                logging.error(f"Failed to process image: {img_path}, error: {e}")
                yield None  # Or return default value

    @torch.no_grad()
    def ImageEncoder(self, images, blur_transform=None, modality=''):
        """
        Add modality. Options are: 'depth_', 'edge_'
        """
        if modality not in ['', 'depth_', 'edge_']:
            raise ValueError('modality must be empty or depth_ or edge_')
        if blur_transform == None:
            blur_transform = self.blur_transform
        self.vlmodel.eval()

        set_images = list(set(images))
        set_images.sort()
        batch_size = 256
        image_features_list = []
        for i in tqdm(range(0, len(set_images), batch_size)):
            batch_images = set_images[i:i + batch_size]

            device = next(self.vlmodel.parameters()).device

            ele = list(self.image_generator(batch_images, blur_transform, modality))
            # ele = [self.process_transform(
            #     blur_transform(
            #         Image.open(os.path.join(self.data_dir, f'../Image_{modality}set_Resize', img)).convert("RGB"))) for
            #     img in batch_images]  # Blur operation is performed here

            image_inputs = torch.stack(ele).to(device)

            batch_image_features = self.vlmodel.encode_image(image_inputs)
            # batch_image_features = batch_image_features / batch_image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(batch_image_features)
        image_features = torch.cat(image_features_list, dim=0)
        image_features_dict = {set_images[i]: image_features[i].float().cpu() for i in range(len(set_images))}
        return image_features_dict

    @torch.no_grad()
    def Textencoder(self, text):  # todo change this for more batch train_set are one img one text

        set_text = list(set(text))
        text_inputs = torch.cat([open_clip.tokenize(f"{t}") for t in set_text])

        device = next(self.vlmodel.parameters()).device
        text_inputs = text_inputs.to(device)

        batch_size = 1024
        text_features_list = []
        for i in tqdm(range(0, len(text_inputs), batch_size)):
            batch_texts = text_inputs[i:i + batch_size]
            batch_text_features = self.vlmodel.encode_text(batch_texts)
            # batch_text_features = batch_text_features / batch_text_features.norm(dim=-1, keepdim=True)
            text_features_list.append(batch_text_features)

        text_features = torch.cat(text_features_list, dim=0)
        text_features_dict = {set_text[i]: text_features[i].float().cpu() for i in range(len(set_text))}
        return text_features_dict

    def __getitem__(self, index):
        subject = index // self.trial_subject
        trial_index = index % self.trial_subject

        meg = self.loaded_data[subject]['eeg'][trial_index].float()
        if self.avg:
            meg_mean = meg
        else:
            meg_mean = self.loaded_data[subject]['meg_avg'][trial_index // self.per_trials].float()

        label = self.loaded_data[subject]['label'][trial_index]
        img_path = self.loaded_data[subject]['img'][trial_index]
        img_index = self.img_path_to_idx[img_path]  # Get from pre-generated mapping
        img_index_tensor = torch.tensor(img_index, dtype=torch.long)

        img = 'None'  # Image.open(os.path.join(self.data_dir,'../Image_set_Resize',img_path)).convert("RGB")

        if self.config['data']['uncertainty_aware']:

            if type(self.match_label) is dict:
                tag = {}
                for key in self.match_label:
                    match_label = self.match_label[key][index]

                    if self.mode == 'train':
                        if match_label == 0:
                            tag[key] = 'low'
                        elif match_label == 2:
                            tag[key] = 'high'
                        else:
                            tag[key] = 'medium'
                    else:
                        tag[key] = 'medium'  # Default to medium during test

                img_features = self.img_features[tag['image']][img_path]
                depth_features = self.depth_features[tag['depth']][img_path]
                edge_features = self.edge_features[tag['edge']][img_path]

            else:
                match_label = self.match_label[index]
                if self.mode == 'train':
                    if match_label == 0:
                        tag = 'low'
                    elif match_label == 2:
                        tag = 'high'
                    else:
                        tag = 'medium'
                else:
                    tag = 'medium'
                img_features = self.img_features[tag][img_path]
                depth_features = self.depth_features[tag][img_path]
                edge_features = self.edge_features[tag][img_path]


        else:
            img_features = self.img_features[img_path]
            depth_features = self.depth_features[img_path]
            edge_features = self.edge_features[img_path]

        text = f"{self.loaded_data[subject]['text'][trial_index]}"
        text_features = self.text_features[self.loaded_data[subject]['text'][trial_index]] 
        session = self.loaded_data[subject]['session'][trial_index]

        sample = {
            'idx': index,
            'eeg': meg[:, self.timesteps[0]:self.timesteps[1]],
            # channel left 17 only 
            'img_path': img_path,
            'img': img,
            'img_index': img_index_tensor,
            'img_features': img_features,
            'depth_features': depth_features,
            'edge_features': edge_features,
            'text': text,
            'text_features': text_features,
            'session': session,
            'subject': subject,
            'meg_mean': meg_mean[:, self.timesteps[0]:self.timesteps[1]],
        }
        return sample

    def __len__(self):
        return self.trial_all_subjects


class MEGDatasetDistributed(Dataset):
    """
    For ddp's val set
    """

    def __init__(self, MEGDataset, gpu_num) -> None:
        self.MEGDataset = MEGDataset
        self.gpu_num = gpu_num
        self.length = len(MEGDataset) * gpu_num

    def __getitem__(self, index):
        actual_index = int(index / self.gpu_num)
        return self.MEGDataset[actual_index]

    def __len__(self):
        return self.length


def load_similarity_data(similarity_dir):
    """Load previously calculated similarity-related files"""
    import numpy as np
    import json
    import torch

    sim_matrix = np.load(os.path.join(similarity_dir, "similarity_matrix.npy"))
    with open(os.path.join(similarity_dir, "image_names.json"), "r") as f:
        image_names = json.load(f)
    image_features = torch.load(os.path.join(similarity_dir, "image_features.pt"))

    # Convert to feature matrix (N×D)
    feature_matrix = torch.stack([image_features[name] for name in image_names]).squeeze().numpy()
    return sim_matrix, image_names, feature_matrix, image_features



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cogcappro_meg.yaml",
        help="path to config which constructs model",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--subjects",
        type=str,
        default='sub-01',
        help="the subjects",
    )
    parser.add_argument(
        "--exp_setting",
        type=str,
        default='intra-subject',
        help="the exp_setting",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=50,
        help="train epoch",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="lr",
    )
    parser.add_argument(
        "--brain_backbone",
        type=str,
        default="EEGProjectLayer",
        help="brain_backbone",
    )
    parser.add_argument(
        "--vision_backbone",
        type=str,
        default="ViT-H-14",
        help="vision_backbone",
    )
    parser.add_argument(
        "--c",
        type=int,
        default=6,
        help="c",
    )
    parser.add_argument(
        "--similarity_dir",
        type=str,
        default=None,
        help="Directory containing similarity matrix and features",
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)
    config = OmegaConf.load(f"{opt.config}")
    config = update_config(opt, config)

    config['devices'] = [0, 1]
    config['data']['subjects'] = [opt.subjects]  # opt parser + yaml file
    config['similarity_dir'] = opt.similarity_dir

    pretrain_map = {
        'RN50': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 1024},
        'RN101': {'pretrained': 'openai', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-16': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-B-32': {'pretrained': 'laion2b_s34b_b79k', 'resize': (224, 224), 'z_dim': 512},
        'ViT-L-14': {'pretrained': 'laion2b_s32b_b82k', 'resize': (224, 224), 'z_dim': 768},
        'ViT-H-14': {'pretrained': 'laion2b_s32b_b79k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-g-14': {'pretrained': 'laion2b_s34b_b88k', 'resize': (224, 224), 'z_dim': 1024},
        'ViT-bigG-14': {'pretrained': 'laion2b_s39b_b160k', 'resize': (224, 224), 'z_dim': 1280}
    }

    config['z_dim'] = pretrain_map[opt.vision_backbone]['z_dim']

    test_dataset = MEGDataset(config, mode='train')
    print(test_dataset[0])

    train_loader, val_loader, test_loader = load_data(config)
    print('over')


def analyze_third_letters(channels):
    """
    Analyze the distribution of the third letter in MEG channel names

    Args:
        channels: List of channel names

    Returns:
        dict: Dictionary of third letters and their occurrence counts
    """
    third_letter_count = {}

    for channel in channels:
        if len(channel) >= 3:
            third_letter = channel[2]
            third_letter_count[third_letter] = third_letter_count.get(third_letter, 0) + 1

    # Sort by alphabetical order
    sorted_count = dict(sorted(third_letter_count.items()))
    return sorted_count


def categorize_channels_by_third_letter(channels):
    """
    Categorize channels into different lists based on the third letter
    
    Args:
        channels: List of channel names
        
    Returns:
        dict: Dictionary of channel lists categorized by third letter
    """
    categories = {
        'C': [],
        'F': [],
        'O': [],
        'P': [],
        'T': []
    }
    
    for channel in channels:
        if len(channel) >= 3:
            third_letter = channel[2]
            if third_letter in categories:
                categories[third_letter].append(channel)
    
    return categories


if __name__ == '__main__':
    main()
