import gc
import logging
import os
from pathlib import Path

import numpy as np
import open_clip
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

from ..utils import get_device, instantiate_from_config

def load_data(config, shuffle_train=True):
    exp_setting = config.get('exp_setting', 'intra-subject')
    num_gpus = len(config['devices'])
    rank = getattr(pl.utilities.rank_zero.rank_zero_only, 'rank', 0)

    if exp_setting == 'intra-subject':
        test_dataset = EEGDataset(config, mode='test')
        if num_gpus > 1:
            test_dataset = EEGDatasetDistributed(test_dataset, num_gpus)
        print('init test_dataset success')

        train_dataset = EEGDataset(config, mode='train')
        print('init train_dataset success')

        train_sampler = DistributedSampler(train_dataset, num_replicas=num_gpus, rank=rank, shuffle=shuffle_train)

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['data']['val_batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['per_gpu_train_batch_size'],
            sampler=train_sampler,
            shuffle=(train_sampler is None and shuffle_train),
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )
        return train_loader, test_loader, test_loader

    if exp_setting == 'inter-subject':
        subjects = config['data']['subjects']
        test_dataset = EEGDataset(config, mode='test')
        print('init test_dataset success')

        all_subjects = [f'sub-{i:02}' for i in range(1, 11)]
        leave_one_subjects = list(set(all_subjects) - set(subjects))
        leave_one_subjects_config = config
        leave_one_subjects_config['data']['subjects'] = leave_one_subjects
        val_dataset = EEGDataset(leave_one_subjects_config, mode='test')
        print('init val_dataset success')
        train_dataset = EEGDataset(leave_one_subjects_config, mode='train')
        print('init train_dataset success')

        if num_gpus > 1:
            train_sampler = DistributedSampler(train_dataset, num_replicas=num_gpus, rank=rank, shuffle=shuffle_train)
        else:
            train_sampler = None

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['data']['test_batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['data']['val_batch_size'],
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['data']['per_gpu_train_batch_size'],
            sampler=train_sampler,
            shuffle=(train_sampler is None and shuffle_train),
            drop_last=False,
            num_workers=0,
            pin_memory=True,
        )
        return train_loader, val_loader, test_loader

    raise ValueError(f"Unsupported exp_setting: {exp_setting}")


class EEGDataset(Dataset):
    def __init__(self, config, mode):
        self.config = config
        self.data_dir = config['data']['data_dir']
        self.data_path = Path(self.data_dir).expanduser()
        self.repo_root = Path(__file__).resolve().parents[3]
        self.subjects = config['data']['subjects']
        self.mode = mode
        self.name = config['name']
        self.model_type = config['data']['model_type']
        self.selected_ch = config['data']['selected_ch']
        self.channels = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3',
                         'F1', 'F2', 'F4', 'F6', 'F8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1',
                         'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'T7', 'C5', 'C3', 'C1',
                         'Cz', 'C2', 'C4', 'C6', 'T8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                         'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'P7', 'P5', 'P3', 'P1',
                         'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                         'O1', 'Oz', 'O2']

        frontal_ch = ['Fp1', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6',
                      'F8']
        temporal_ch = ['FT9', 'FT7', 'FT8', 'FT10', 'T7', 'T8', 'TP9', 'TP7', 'TP8', 'TP10']
        central_ch = ['FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                      'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']
        parietal_ch = ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8']
        occipital_ch = ['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
        self_ch = ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1',
                   'Oz', 'O2']

        if self.selected_ch is True:
            self.selected_ch = self_ch
        elif isinstance(self.selected_ch, str):
            region_map = {
                'frontal': frontal_ch,
                'temporal': temporal_ch,
                'central': central_ch,
                'parietal': parietal_ch,
                'occipital': occipital_ch,
                'self': self_ch,
            }
            self.selected_ch = region_map.get(self.selected_ch, self.channels)
        else:
            self.selected_ch = self.channels

        self.avg = config['data'][f'{mode}_avg']
        self.blur_type = config['data']['blur_type']
        self.timesteps = config['data']['timesteps']

        self.n_cls = 1654 if self.mode == 'train' else 200
        self.per_trials = 4 if self.mode == 'train' else 80
        self.texts_BLIP2 = self._load_blip2_texts()

        self.data_paths = [os.path.join(self.data_dir, subject, f'{mode}.pt') for subject in self.subjects]
        self.loaded_data = [self.load_data(data_path) for data_path in self.data_paths]

        self.img_path_to_idx = {}
        self.idx_counter = 0
        for subject_data in self.loaded_data:
            img_paths = subject_data['img']
            for path in img_paths:
                if path not in self.img_path_to_idx:
                    self.img_path_to_idx[path] = self.idx_counter
                    self.idx_counter += 1

        self.trial_subject = self.loaded_data[0]['eeg'].shape[0]
        self.trial_all_subjects = self.trial_subject * len(self.subjects)

        data_dir = self.data_path.parent / 'Image_feature_new' / f"{config['data']['blur_type']['target'].rsplit('.', 1)[-1]}"
        data_dir.mkdir(parents=True, exist_ok=True)

        base_name = self.name.replace(f"_{self.config['brain_backbone']}", '')
        features_filename = data_dir / f'{base_name}_{mode}.pt'

        self.c = config['c']
        if self.config['data']['uncertainty_aware']:
            self.blur_transform = {}
            for shift, tag in zip([-self.c, 0, self.c], ['low', 'medium', 'high']):
                blur_param = config['data']['blur_type']
                blur_param['params']['blur_kernel_size'] = blur_param['params']['blur_kernel_size'] + shift
                self.blur_transform[tag] = instantiate_from_config(blur_param)
        else:
            self.blur_transform = instantiate_from_config(config['data']['blur_type'])

        process_term = [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ]
        self.process_transform = transforms.Compose(process_term)

        self.match_label = {
            'image': np.ones(self.trial_all_subjects, dtype=int),
            'text': np.ones(self.trial_all_subjects, dtype=int),
            'depth': np.ones(self.trial_all_subjects, dtype=int),
            'edge': np.ones(self.trial_all_subjects, dtype=int),
        }

        if features_filename.exists():
            saved_features = torch.load(features_filename)
            self.img_features = saved_features['img_features']
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
                device=f'cuda:{device}',
                pretrained=pre_trained_pth[self.model_type],
            )

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
                    self.depth_features[tag] = self.ImageEncoder(
                        self.loaded_data[0]['img'], self.blur_transform[tag], modality='depth_'
                    )
                    self.edge_features[tag] = self.ImageEncoder(
                        self.loaded_data[0]['img'], self.blur_transform[tag], modality='edge_'
                    )

                self.img_features['avg'] = {
                    k: (sum(self.img_features[tag][k] for tag in ['low', 'medium', 'high']) / 3)
                    for k in self.img_features['medium']
                }
                self.depth_features['avg'] = {
                    k: (sum(self.depth_features[tag][k] for tag in ['low', 'medium', 'high']) / 3)
                    for k in self.depth_features['medium']
                }
                self.edge_features['avg'] = {
                    k: (sum(self.edge_features[tag][k] for tag in ['low', 'medium', 'high']) / 3)
                    for k in self.edge_features['medium']
                }
            else:
                self.img_features = self.ImageEncoder(self.loaded_data[0]['img'])
                self.depth_features = self.ImageEncoder(self.loaded_data[0]['img'], modality='depth_')
                self.edge_features = self.ImageEncoder(self.loaded_data[0]['img'], modality='edge_')

            torch.save(
                {
                    'text_features': self.text_features,
                    'img_features': self.img_features,
                    'depth_features': self.depth_features,
                    'edge_features': self.edge_features,
                },
                str(features_filename),
            )

            del self.vlmodel
            torch.cuda.empty_cache()
            gc.collect()

    def _load_blip2_texts(self):
        blip2_path = self.repo_root / 'weights' / 'texts' / 'eeg'
        blip2_file = blip2_path / f'texts_BLIP2_{self.mode}.npy'
        if not blip2_file.exists():
            raise FileNotFoundError(f"Missing EEG BLIP2 texts: {blip2_file}")
        return np.load(blip2_file)[::2]

    def load_data(self, data_path):
        logging.info(f"----load {data_path}----")
        loaded_data = torch.load(data_path)
        loaded_data['eeg'] = torch.from_numpy(loaded_data['eeg'])

        if self.selected_ch:
            selected_idx = [self.channels.index(ch) for ch in self.selected_ch]
            loaded_data['eeg'] = loaded_data['eeg'][:, :, selected_idx]

        if self.avg:
            avg_data = {}
            avg_data['eeg'] = loaded_data['eeg'].mean(axis=1)
            avg_data['label'] = loaded_data['label'][:, 0]
            avg_data['img'] = loaded_data['img'][:, 0]
            avg_data['text'] = self.texts_BLIP2
            avg_data['session'] = loaded_data['session']
            avg_data['times'] = loaded_data['times']
            loaded_data = avg_data
        else:
            _data = {}
            _data['eeg'] = loaded_data['eeg'].reshape(-1, *loaded_data['eeg'].shape[2:])
            _data['eeg_avg'] = loaded_data['eeg'].mean(axis=1)
            _data['label'] = loaded_data['label'].reshape(-1)
            _data['img'] = loaded_data['img'].reshape(-1)
            _data['text'] = np.tile(self.texts_BLIP2, self.per_trials)
            _data['session'] = loaded_data['session'].reshape(-1)
            _data['times'] = loaded_data['times']
            loaded_data = _data

        return loaded_data

    def image_generator(self, img_paths, blur_transform, modality=''):
        for img_path in img_paths:
            full_path = self.data_path.parent / f'Image_{modality}set_Resize' / img_path
            try:
                img = Image.open(full_path).convert('RGB')
                if hasattr(blur_transform, '__class__') and blur_transform.__class__.__name__ == 'DirectT':
                    processed_img = self.process_transform(img)
                else:
                    processed_img = self.process_transform(blur_transform(img))
                img.close()
                yield processed_img
            except Exception as error:
                logging.error(f'处理图像失败: {img_path}, 错误: {error}')
                yield None

    @torch.no_grad()
    def ImageEncoder(self, images, blur_transform=None, modality=''):
        if modality not in ['', 'depth_', 'edge_']:
            raise ValueError('modality must be empty or depth_ or edge_')
        if blur_transform is None:
            blur_transform = self.blur_transform

        self.vlmodel.eval()

        set_images = list(set(images))
        set_images.sort()
        batch_size = 256
        image_features_list = []
        for i in tqdm(range(0, len(set_images), batch_size)):
            batch_images = set_images[i:i + batch_size]
            device = next(self.vlmodel.parameters()).device
            image_inputs = torch.stack(list(self.image_generator(batch_images, blur_transform, modality))).to(device)
            batch_image_features = self.vlmodel.encode_image(image_inputs)
            image_features_list.append(batch_image_features)

        image_features = torch.cat(image_features_list, dim=0)
        image_features_dict = {set_images[i]: image_features[i].float().cpu() for i in range(len(set_images))}
        return image_features_dict

    @torch.no_grad()
    def Textencoder(self, text):
        set_text = list(set(text))
        text_inputs = torch.cat([open_clip.tokenize(f'{t}') for t in set_text])

        device = next(self.vlmodel.parameters()).device
        text_inputs = text_inputs.to(device)

        batch_size = 1024
        text_features_list = []
        for i in tqdm(range(0, len(text_inputs), batch_size)):
            batch_texts = text_inputs[i:i + batch_size]
            batch_text_features = self.vlmodel.encode_text(batch_texts)
            text_features_list.append(batch_text_features)

        text_features = torch.cat(text_features_list, dim=0)
        text_features_dict = {set_text[i]: text_features[i].float().cpu() for i in range(len(set_text))}
        return text_features_dict

    def __getitem__(self, index):
        subject = index // self.trial_subject
        trial_index = index % self.trial_subject

        eeg = self.loaded_data[subject]['eeg'][trial_index].float()
        if self.avg:
            eeg_mean = eeg
        else:
            eeg_mean = self.loaded_data[subject]['eeg_avg'][trial_index // self.per_trials].float()

        label = self.loaded_data[subject]['label'][trial_index]
        img_path = self.loaded_data[subject]['img'][trial_index]
        img_index = self.img_path_to_idx[img_path]
        img_index_tensor = torch.tensor(img_index, dtype=torch.long)

        img = 'None'

        if self.config['data']['uncertainty_aware']:
            if isinstance(self.match_label, dict):
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
                        tag[key] = 'medium'

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
            'eeg': eeg[:, self.timesteps[0]:self.timesteps[1]],
            'label': label,
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
            'eeg_mean': eeg_mean[:, self.timesteps[0]:self.timesteps[1]],
        }
        return sample

    def __len__(self):
        return self.trial_all_subjects


class EEGDatasetDistributed(Dataset):
    def __init__(self, dataset, gpu_num) -> None:
        self.dataset = dataset
        self.gpu_num = gpu_num
        self.length = len(dataset) * gpu_num

    def __getitem__(self, index):
        actual_index = int(index / self.gpu_num)
        return self.dataset[actual_index]

    def __len__(self):
        return self.length
