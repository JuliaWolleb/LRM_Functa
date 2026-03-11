import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import pandas as pd
import os
import nibabel
import cv2
from torch.utils.data import DataLoader, Dataset
from imutils import paths
import random
import sys
import matplotlib.pyplot as plt 
import torch.nn.functional as F
from scipy.ndimage import shift, zoom

fourcc = cv2.VideoWriter_fourcc(*'mp4v')



class LVH_video(torch.utils.data.Dataset):

    def __init__(
        self,
        datapath='./data/POCUS',
        img_size=112,
        num_frames=4,
        selection=True,
        sobel=False,
        high_pass=False
    ):

        self.datapath = datapath
        self.img_size = img_size
        self.num_frames = num_frames
        self.selection = selection
        self.sobel = sobel
        self.high_pass = high_pass

        # Collect all mp4 files
        self.video_paths = sorted([
            os.path.join(datapath, f)
            for f in os.listdir(datapath)
            if f.endswith(".mp4")
        ])

        print("Found videos:", len(self.video_paths))

    def __len__(self):
        return len(self.video_paths)

    def _load_labels(self, video_path):
        """
        Load ES and ED frame numbers from the corresponding CSV
        """
        base = os.path.splitext(os.path.basename(video_path))[0]  # ID_04
        csv_path = os.path.join(self.datapath, f"{base}.csv")
       

        df = pd.read_csv(csv_path, usecols=["ImageNo"])
        values = df["ImageNo"].dropna().astype(int).values

      
       
        unique, counts = np.unique(values, return_counts=True)
     #   print('unique', unique)
        try:
            # End systole = appears once
            es_frame = unique[counts == 1][0]

            # End diastole = appears multiple times
            ed_frame = unique[counts > 1][0]   # or keep all if you want
   
      
            return int(es_frame), int(ed_frame)
        except: 
            return 1,2

    def __getitem__(self, idx):

        video_path = self.video_paths[idx]
        filename = os.path.basename(video_path)

        # ---- Load labels ----
        es_frame, ed_frame = self._load_labels(video_path)

        # ---- Load video ----
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)

        cap.release()

        out = np.asarray(frames)  # [T, H, W]
        length = out.shape[0]

        # ---- Pad if too short ----
        if length < self.num_frames:
            out = np.tile(out, (self.num_frames // length + 1, 1, 1))[:self.num_frames]

        # ---- Frame selection ----
        if self.selection:
            selected = sorted(random.sample(range(out.shape[0]), self.num_frames))
            out = out[selected]

            # adjust labels to selected frames if needed
            es_rel = min(range(len(selected)), key=lambda i: abs(selected[i] - es_frame))
            ed_rel = min(range(len(selected)), key=lambda i: abs(selected[i] - ed_frame))
        else:
            es_rel = es_frame
            ed_rel = ed_frame

        # ---- Normalize ----
        out = np.clip(out, np.quantile(out, 0.001), np.quantile(out, 0.999))
        out = (out - out.min()) / (out.max() - out.min())
        vid = torch.tensor(out, dtype=torch.float32)

        if self.img_size == 112:
            vid = nn.AvgPool2d(2, 2)(vid)

        label = {
            "ES": es_rel,
            "ED": ed_rel
        }

        return vid, label, fps, filename


class LUS_video(torch.utils.data.Dataset):

        def __init__(self, datapath= './data/lung/', labelsfile = './data/lung/Callinformation.csv', img_size=112, transform=None, clip = 'all', num_frames = 4, selection = True, sobel = False, high_pass = False):
           
            self.datapath= datapath
            self.high_pass = high_pass
            self.transform = transform
            self.sobel = sobel


            file_paths = []
            for dirpath, dirnames, filenames in os.walk(self.datapath):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    file_paths.append(file_path)
            self.image_paths = file_paths

            
            self.img_size = img_size
            self.clip = clip
            self.num_frames = num_frames
            self.selection = selection
          
          
            self.labels_df = pd.read_csv(labelsfile)

            try:
                self.labels_df.dropna(subset=['Label'], inplace=True)
            except:
                pass


            
        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            image_names=self.image_paths[idx]
           
            filename = image_names.split('/')[-1].split('.')[0][:4]
           
            
            L= self.labels_df.loc[self.labels_df['ID'] == filename, 'Label'].values
            label_map = {'Negative': 0, 'Positive': 1}

            # Convert
            label = [label_map[l] for l in L]


            
            curr_vid_path= os.path.join( image_names)
           
            cap = cv2.VideoCapture(curr_vid_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

          
            current_data = []
            
            while cap.isOpened():
                frame_id = cap.get(1)

                ret, frame = cap.read()
                if (ret != True):
                    break
                image = cv2.resize(frame, (224,224))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
          
                numpyimage = np.asarray(image)
               
               
                current_data.append(numpyimage)
               
            cap.release()

            out= np.asarray(current_data)

            length = out.shape[0]
            if length < self.num_frames:
             
              out_repeated = np.tile(out, (4, 1, 1))[:12]
              out = out_repeated
            


            if self.selection: 
                try:
                 
                    start_t = random.randint(0, out.shape[0] - self.num_frames*2)
                    selected=random.sample(range(0, out.shape[0]), self.num_frames)
                    out =out[selected,...]
                    timestep =0
                    
                except:
                    out=out[:self.num_frames]
                    timestep =(torch.arange(0, length)/ length )[...,None]
            else:
                timestep = 0
            
            try:
             # Clip and normalize the images
                out_clipped = np.clip(out, np.quantile(out, 0.001), np.quantile(out, 0.999))
                out_normalized = (out_clipped - np.min(out_clipped)) / (np.max(out_clipped) - np.min(out_clipped))
                out = torch.tensor(out_normalized)
                if self.img_size == 112:
                    downsample = nn.AvgPool2d(kernel_size=2, stride=2)
                    vid = downsample(out)


                
            except:
                vid=torch.nan

            output= {
                    "vid": vid,
                    'time': timestep,
                    'label': label  
                }

            return  vid, label, fps, filename
        
        print('got loaders cristiana')




def get_dataset(args, only_test=False, all=False, double = False):
    train_set = None
    val_set = None
    test_set = None
    print('args', args.dataset)



    if args.dataset == 'lusvideo' or args.dataset== 'autoregressive':
       

        if args.option =='lung':
            print('got lung dataset')
            dataset = LUS_video(img_size=args.img_size, clip=args.clip, num_frames=args.num_frames, selection=args.selection, sobel = args.sobel, high_pass = args.high_pass)
       
        elif args.option =='LVH':
            print('got LVH dataset')
            dataset = LVH_video( datapath='./data/POCUS', img_size=args.img_size,  num_frames=args.num_frames, selection=args.selection, sobel = args.sobel, high_pass = args.high_pass)
       

        
        train_size = int(0.8 * len(dataset))  # 80% for training
        val_size = int(0.1 * len(dataset)) +1 # 10% for validation
        test_size = len(dataset) - train_size - val_size # 10% for testing
        generator = torch.Generator().manual_seed(42)
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=generator)
        print(f'Training set containing {len(train_set)} images.')
        print(f'Test set containing {len(test_set)} images.')



    elif  args.dataset =='framewise':
        
            
            dataset = LUS_video(img_size=args.img_size, clip=args.clip)
            train_size = int(0.8 * len(dataset))  # 80% for training
            val_size = int(0.1 * len(dataset)) +1 # 10% for validation
            test_size = len(dataset) - train_size - val_size # 10% for testing
            generator = torch.Generator().manual_seed(42)
            train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=generator)
            
            print(f'Training set containing {len(train_set)} images.')
            print(f'Test set containing {len(test_set)} images.')
            args.data_type = 'img'
            args.in_size, args.out_size = 2, 1
            args.data_size = (1, args.img_size, args.img_size)



    else:
        raise NotImplementedError()

    if only_test:
        return test_set

    elif all:
        return train_set, val_set, test_set

    else:
        return train_set, test_set
