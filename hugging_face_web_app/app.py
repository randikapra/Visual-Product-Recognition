import gradio
import torch
from torchvision import transforms 

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import math
import yaml

from PIL import Image 
import io
import base64
import requests

class CFG:
    model_name = 'ViT-H-14'           #Neural network model architecture
    model_data = 'laion2b_s32b_b79k'  #Pretrained model
    samples_per_class = 50            #Class balancing
    n_classes = 0                     
    min_samples = 4                   
    image_size = 224                  #pixel 224 x 224
    hidden_layer = 1024                #number of neurons in a hidden layer
    seed = 5                         
    workers = 12                      #number of CPU cores ; parallel tasks
    train_batch_size = 4              
    valid_batch_size = 8             
    emb_size = 512                    
    vit_bb_lr = {'10': 1.25e-6, '20': 2.5e-6, '26': 5e-6, '32': 10e-6}   #learning rates of backbone
    vit_bb_wd = 1e-3                  #weight dacay of backbone
    hd_lr = 3e-4                      
    hd_wd = 1e-5                      
    autocast = True                   
    n_warmup_steps = 1000             
    n_epochs = 10                     
    device = torch.device('cuda')     
    s=30.                             
    m=0.45                            
    m_min=0.05                        
    acc_steps = 4                     
    global_step = 0                   
    reduce_lr = 0.1                   
    crit = 'ce'                       #loss function cross entropy


class utilities():
    class ArcMarginProduct(nn.Module):
        
        #Softmax Loss function extensiion - "Additive Angular Margin Loss."
        '''def __init__(self, dimension_of_input_features,  size_of_each_output_sample, scaling_factor_for_cosine_similarity, 
                     margin, easy_margin=False, ls_eps=0.0,computation-cuda)):
        '''    
        """Implement of large margin arc distance: :
            Args:
                in_features: size of each input sample
                out_features: size of each output sample
                s: norm of input feature
                m: margin
                cos(theta + m)
            """
        """
        exteded softmax loss fn. 
        simultaneously enhance the intra-class compactness and inter-class discrepancy
        """
        def __init__(self, in_features, out_features, s=30.0, 
                     m=0.50, easy_margin=False, ls_eps=0.0, device=torch.device('cuda')):
            super(ArcMarginProduct, self).__init__()
            self.device = device
            self.in_features = in_features
            self.out_features = out_features
            self.s = s
            self.m = m
            self.ls_eps = ls_eps  # label smoothing
            self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
            nn.init.xavier_uniform_(self.weight)

            self.easy_margin = easy_margin
            self.cos_m = math.cos(m)
            self.sin_m = math.sin(m)
            self.th = math.cos(math.pi - m)
            self.mm = math.sin(math.pi - m) * m
        
        #forward pass
        def forward(self, input, label):
            # --------------------------- cos(theta) & phi(theta) ---------------------
            cosine = F.linear(F.normalize(input), F.normalize(self.weight))
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            # --------------------------- convert label to one-hot ---------------------
            # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
            # create a mask for the correct class
            one_hot = torch.zeros(cosine.size(), device=self.device)
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            if self.ls_eps > 0:
                one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features
            # -------------torch.where(out_i = {x_i if condition_i else y_i) ------------
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s

            return output
    
    # Dense (per-class) cross-entropy loss for a classification task. 
    # loss fn for the classification between prediction and true. SGD based loss fn.
    # DCE = -logprobs * target
    class DenseCrossEntropy(nn.Module):
        def forward(self, x, target):
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim=-1)

            loss = -logprobs * target
            loss = loss.sum(-1)
            return loss.mean()
    
    # loss fn in class imbalanced prob. 
    # FL(pt) = -target(1-probs)^(gamma)*(logprobs)
    class FocalLoss(nn.Module):
        def __init__(self, gamma=2):
            super(FocalLoss, self).__init__()
            self.gamma = gamma

        def forward(self, x, target):
            x = x.float()
            target = target.float()
            probs = torch.nn.functional.softmax(x, dim=-1)
            logprobs = torch.log(probs)

            loss = -logprobs * target * (1 - probs) ** self.gamma
            loss = loss.sum(-1)
            return loss.mean()
    
    #compute the cosine similarity scores between input features and a learnable set of "center" vectors.
    class ArcMarginProduct_subcenter(nn.Module):
        def __init__(self, in_features, out_features, k=3):
            super().__init__()
            self.weight = nn.Parameter(torch.FloatTensor(out_features*k, in_features))
            self.reset_parameters()
            self.k = k
            self.out_features = out_features

        def reset_parameters(self):
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)

        def forward(self, features):
            cosine_all = F.linear(F.normalize(features), F.normalize(self.weight))
            cosine_all = cosine_all.view(-1, self.out_features, self.k)
            cosine, _ = torch.max(cosine_all, dim=2)
            return cosine   
    
    class ArcFaceLossAdaptiveMargin(nn.modules.Module):
        def __init__(self, margins, s=30.0, crit='ce'):
            super().__init__()
            if crit == 'ce':
                self.crit = utilities.DenseCrossEntropy()
            else:
                self.crit = utilities.FocalLoss()
            self.s = s
            self.margins = margins
            
        def forward(self, logits, labels, out_dim):
            ms = []
            ms = self.margins[labels.cpu().numpy()]
            cos_m = torch.from_numpy(np.cos(ms)).float().cuda()
            sin_m = torch.from_numpy(np.sin(ms)).float().cuda()
            th = torch.from_numpy(np.cos(math.pi - ms)).float().cuda()
            mm = torch.from_numpy(np.sin(math.pi - ms) * ms).float().cuda()
            labels = F.one_hot(labels, out_dim).float()
            logits = logits.float()
            cosine = logits
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * cos_m.view(-1,1) - sine * sin_m.view(-1,1)
            phi = torch.where(cosine > th.view(-1,1), phi, cosine - mm.view(-1,1))
            output = (labels * phi) + ((1.0 - labels) * cosine)
            output *= self.s
            loss = self.crit(output, labels)
            return loss     

    def set_seed(seed):
        '''Sets the seed of the entire notebook so results are the same every time we run.
        This is for REPRODUCIBILITY.'''
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Set a fixed value for the hash seed
        os.environ['PYTHONHASHSEED'] = str(seed)


    def get_similiarity_hnsw(embeddings_gallery, emmbeddings_query, k):
        
        print('Processing indices...')

        s = time.time()
        index = faiss.IndexHNSWFlat(embeddings_gallery.shape[1], 32)
        index.add(embeddings_gallery)

        scores, indices = index.search(emmbeddings_query, k) 
        e = time.time()

        print(f'Finished processing indices, took {e - s}s')
        return scores, indices
    
    #Ecucledian Distance Similarity measure
    def get_similiarity_l2(embeddings_gallery, emmbeddings_query, k):
        print('Processing indices...')

        s = time.time()
        index = faiss.IndexFlatL2(embeddings_gallery.shape[1])
        index.add(embeddings_gallery)

        scores, indices = index.search(emmbeddings_query, k) 
        e = time.time()

        print(f'Finished processing indices, took {e - s}s')
        return scores, indices


    def get_similiarity_IP(embeddings_gallery, emmbeddings_query, k):
        print('Processing indices...')

        s = time.time()
        index = faiss.IndexFlatIP(embeddings_gallery.shape[1])
        index.add(embeddings_gallery)

        scores, indices = index.search(emmbeddings_query, k) 
        e = time.time()

        print(f'Finished processing indices, took {e - s}s')
        return scores, indices

    def get_similiarity(embeddings, k):
        print('Processing indices...')

        index = faiss.IndexFlatL2(embeddings.shape[1])

        res = faiss.StandardGpuResources()

        index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(embeddings)

        scores, indices = index.search(embeddings, k) 
        print('Finished processing indices')

        return scores, indices
    @staticmethod
    def map_per_image(label, predictions, k=5): 
        try:
            return 1 / (predictions[:k].index(label) + 1)
        except ValueError:
            return 0.0
    @staticmethod
    def map_per_set(labels, predictions, k=5):
        return np.mean([utilities.map_per_image(l, p, k) for l,p in zip(labels, predictions)])
    
    class AverageMeter(object):
        """Computes and stores the average and current value"""

        def __init__(self, window_size=None):
            self.length = 0
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0
            self.window_size = window_size

        def reset(self):
            self.length = 0
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            if self.window_size and (self.count >= self.window_size):
                self.reset()
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    def get_lr_groups(param_groups):
            groups = sorted(set([param_g['lr'] for param_g in param_groups]))
            groups = ["{:2e}".format(group) for group in groups]
            return groups

    def convert_indices_to_labels(indices, labels):
        indices_copy = copy.deepcopy(indices)
        for row in indices_copy:
            for j in range(len(row)):
                row[j] = labels[row[j]]
        return indices_copy

    class Multisample_Dropout(nn.Module):
        def __init__(self, dropout_rate=0.1):
            super(Multisample_Dropout, self).__init__()
            self.dropout = nn.Dropout(dropout_rate)
            self.dropouts = nn.ModuleList([nn.Dropout((i+1)*.1) for i in range(5)])

        def forward(self, x, module):
            x = self.dropout(x)
            return torch.mean(torch.stack([module(dropout(x)) for dropout in self.dropouts],dim=0),dim=0) 
    
    #Data augmentation
    def transforms_auto_augment(image_path, image_size):
        image = Image.open(image_path).convert('RGB')
        train_transforms = transforms.Compose([transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET), transforms.PILToTensor()])
        return train_transforms(image)

    def transforms_cutout(image_path, image_size):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
        train_transforms = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.ImageCompression(quality_lower=99, quality_upper=100),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=10, border_mode=0, p=0.7),
                A.Resize(image_size, image_size),
                A.Cutout(max_h_size=int(image_size * 0.4), max_w_size=int(image_size * 0.4), num_holes=1, p=0.5),
                ToTensorV2(),
            ])
        return train_transforms(image=image)['image']

    def transforms_happy_whale(image_path, image_size):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
        aug8p3 = A.OneOf([
                A.Sharpen(p=0.3),
                A.ToGray(p=0.3),
                A.CLAHE(p=0.3),
            ], p=0.5)

        train_transforms = A.Compose([
                A.ShiftScaleRotate(rotate_limit=15, scale_limit=0.1, border_mode=cv2.BORDER_REFLECT, p=0.5),
                A.Resize(image_size, image_size),
                aug8p3,
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
                ToTensorV2(),
            ])
        return train_transforms(image=image)['image']

    def transforms_valid(image_path, image_size):
        image = Image.open(image_path).convert('RGB')
        valid_transforms = transforms.Compose([transforms.PILToTensor()]) 
        return valid_transforms(image)
  

class Model(nn.Module):
    def __init__(self, vit_backbone, head_size, version='v1', k=3):
        super(Model, self).__init__()
        if version == 'v1':
            self.head = Head(head_size, k)
        elif version == 'v2':
            self.head = HeadV2(head_size, k)
        elif version == 'v3':
            self.head = HeadV3(head_size, k)
        else:
            self.head = Head(head_size, k)
        
        self.encoder = vit_backbone.visual
    def forward(self, x):
        x = self.encoder(x)
        return self.head(x)

    def get_parameters(self):

        parameter_settings = [] 
        parameter_settings.extend(
            self.get_parameter_section(
                [(n, p) for n, p in self.encoder.named_parameters()], 
                lr=CFG.vit_bb_lr, 
                wd=CFG.vit_bb_wd
            )
        ) 

        parameter_settings.extend(
            self.get_parameter_section(
                [(n, p) for n, p in self.head.named_parameters()], 
                lr=CFG.hd_lr, 
                wd=CFG.hd_wd
            )
        ) 

        return parameter_settings

    def get_parameter_section(self, parameters, lr=None, wd=None): 
        parameter_settings = []


        lr_is_dict = isinstance(lr, dict)
        wd_is_dict = isinstance(wd, dict)

        layer_no = None
        for no, (n,p) in enumerate(parameters):
            
            for split in n.split('.'):
                if split.isnumeric():
                    layer_no = int(split)
            
            if not layer_no:
                layer_no = 0
            
            if lr_is_dict:
                for k,v in lr.items():
                    if layer_no < int(k):
                        temp_lr = v
                        break
            else:
                temp_lr = lr

            if wd_is_dict:
                for k,v in wd.items():
                    if layer_no < int(k):
                        temp_wd = v
                        break
            else:
                temp_wd = wd

            weight_decay = 0.0 if 'bias' in n else temp_wd

            parameter_setting = {"params" : p, "lr" : temp_lr, "weight_decay" : temp_wd}

            parameter_settings.append(parameter_setting)

            #print(f'no {no} | params {n} | lr {temp_lr} | weight_decay {weight_decay} | requires_grad {p.requires_grad}')

        return parameter_settings

class Head(nn.Module):
    def __init__(self, hidden_size, k=3):
        super(Head, self).__init__()
        self.emb = nn.Linear(hidden_size, CFG.emb_size, bias=False)
        self.dropout = utilities.Multisample_Dropout()
        self.arc = utilities.ArcMarginProduct_subcenter(CFG.emb_size, CFG.n_classes, k)
        
    def forward(self, x):
        embeddings = self.dropout(x, self.emb)
        output = self.arc(embeddings)
        return output, F.normalize(embeddings)
    
class HeadV2(nn.Module):
    def __init__(self, hidden_size, k=3):
        super(HeadV2, self).__init__()
        self.arc = utilities.ArcMarginProduct_subcenter(hidden_size, CFG.n_classes, k)
        
    def forward(self, x):
        output = self.arc(x)
        return output, F.normalize(x)
    
class HeadV3(nn.Module):
    def __init__(self, hidden_size, k=3):
        super(HeadV3, self).__init__()        
        self.emb = nn.Linear(hidden_size, CFG.emb_size, bias=False)
        self.dropout = nn.Dropout1d(0.2)
        self.arc = utilities.ArcMarginProduct_subcenter(CFG.emb_size, CFG.n_classes, k)
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.emb(x)
        output = self.arc(x)
        return output, F.normalize(x)


import torch
from torchvision import transforms
from PIL import Image
import base64
import io
import json
import numpy as np

import gradio as gr
from gradio import Interface, components
import requests
import json

import torch
from transformers import AutoModelForImageClassification

from gradio.data_classes import FileData

model = torch.load("model.pt")


def predict(image):
    """Generates a prediction for the given image.""" 
    # image = FileData(image)
    # image = components.Image(image)
    image = Image.fromarray(np.uint8(image))

    model.eval()

    # # Convert the base64 image to a PIL Image
    # image_binary = base64.b64decode(image)
    # image = Image.open(io.BytesIO(image_binary))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    img = transform(image).unsqueeze(0)  # Add a batch dimension
    
    # Move the image tensor to the same device as the model
    image_tensor = img.to(torch.device('cuda'), dtype=torch.float32)

    # Obtain embeddings using the model
    with torch.no_grad():
        _, embedding = model(image_tensor)

    embedding_array = embedding.detach().cpu().numpy()

    json_data = json.dumps({"predictions": embedding_array.tolist()})

    return json_data


image = components.Image()

# interface = gr.Interface(fn=predict, inputs=[image], outputs="json")
interface = gr.Interface(fn=predict, inputs="image", outputs="json")

# Add a button to the Gradio interface to send the prediction to the Flask backend.
# interface.add_component("button", "Send to Backend", send_fn=send_prediction, inputs=["json"])


# Launch the Gradio interface.
interface.launch(debug=True)

