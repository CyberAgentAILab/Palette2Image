import json
import cv2
from PIL import Image
import numpy as np
import ast
import torch

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from color.utils.color_data_generator import ColorGenerator

class MyPaletteDataset(Dataset):
    def __init__(self, data_tag='test'):
        self.data_tag = data_tag
        self.data = []

        # get palette emb from palette_only model
        # self.palette_only_embeddings = torch.load(f'pretrain/image-palette/color/palette_emb_ponly_768d/palette_emb_{data_tag}.pt')

        # get palette emb from text2palette model
        # self.palette_t2p_embeddings = torch.load(f'pretrain/image-palette/color/palette_emb_t2p_768d_m0/palette_emb_{data_tag}.pt')

        # get palette emb from palette_image model
        self.palette_i2p_embeddings = torch.load(f'pretrain/image-palette/color/palette_emb_by_image_768d/palette_emb_{data_tag}.pt')
        
        with open(f'./pretrain/image-palette/data_{data_tag}.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        # read pre-created palette emb from pretrained color model
        # palette_emb = self.palette_only_embeddings[idx]
        # palette_emb = torch.cat([palette_emb, self.palette_t2p_embeddings[idx]], dim=0)
        palette_emb = self.palette_i2p_embeddings[idx]
        
        # get source, target and prompt
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']
        palette = ast.literal_eval(item['palette'])

        # source is the palette RGB color list, target is the colorful image
        source = cv2.imread('pretrain/image-palette/' + source_filename)
        target = cv2.imread('pretrain/image-palette/' + target_filename)

        # resize image into same dimensions
        source = cv2.resize(source, (512, 512))
        target = cv2.resize(target, (512, 512))
        
        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        
        if self.data_tag == 'train':
            # Normalize source images to [0, 1].
            source = source.astype(np.float32) / 255.0
            # Normalize target images to [-1, 1].
            target = (target.astype(np.float32) / 127.5) - 1.0
            return dict(jpg=target, txt=prompt, plt=palette_emb, hint=source)

        return dict(jpg=target, txt=prompt, plt_v=palette, plt=palette_emb, hint=source)

