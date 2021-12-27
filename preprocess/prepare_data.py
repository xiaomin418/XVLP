
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
import torchvision.transforms as transforms
import os
import numpy as np
import json
import glob

class VggEncoder(nn.Module):

    def __init__(self,tk_dim = 49, hidden_dim = 512, train_CNN=False):
        super(VggEncoder, self).__init__()
        self.tk_dim = tk_dim
        self.hidden_dim = hidden_dim
        self.train_CNN = train_CNN
        self.vgg19 = models.vgg19(pretrained=True)
        self.vgg19=self.vgg19.eval()
        self.W_h = nn.Linear(self.hidden_dim , self.hidden_dim, bias=False)
        # self.dropout = nn.Dropout(config.dropout)

    def forward(self, images):
        # Fine tuning, we don't want to train
        local_features = self.vgg19.features(images)
        local_features = local_features.detach()
        x = self.vgg19.avgpool(local_features)
        x = torch.flatten(x,1)
        global_features = self.vgg19.classifier[:6](x)
        global_features = global_features.detach()


        local_outputs = local_features.view(-1, self.tk_dim, self.hidden_dim) #B x t_k x hidden_dim
        local_features = local_outputs.view(-1, self.hidden_dim)  # B * t_k x 2*hidden_dim
        local_features = self.W_h(local_features)
        # local_features = self.dropout(local_features)


        return local_outputs, local_features,global_features

def get_img_det(img_path, transform, img_encoder, device):
    img = Image.open(os.path.join(img_path)).convert("RGB")
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    local_outputs, local_features, global_features = img_encoder(img)
    return local_outputs, local_features, global_features

def write_one_sentence(sentence_str, id):
    """
    {'sentids': [0, 1, 2, 3, 4], 'imgid': 0, 'sentences': [{'tokens': ['two', 'young', 'guys', 'with', 'shaggy', 'hair', 'look', 'at', 'their', 'hands', 'while', 'hanging', 'out', 'in', 'the', 'yard'], 'raw': 'Two young guys with shaggy hair look at their hands while hanging out in the yard.', 'imgid': 0, 'sentid': 0, 'deptree': [[2, 0], [2, 1], [8, 2], [7, 3], [3, 4], [7, 5], [7, 6], [2, 7], [-1, 8], [11, 9], [11, 10], [8, 11], [13, 12], [8, 13], [13, 14], [17, 15], [17, 16], [13, 17], [8, 18]]}, {'tokens': ['two', 'young', 'white', 'males', 'are', 'outside', 'near', 'many', 'bushes'], 'raw': 'Two young, White males are outside near many bushes.', 'imgid': 0, 'sentid': 1, 'deptree': [[1, 0], [-1, 1], [1, 2], [4, 3], [6, 4], [6, 5], [1, 6], [9, 7], [9, 8], [6, 9], [1, 10]]}, {'tokens': ['two', 'men', 'in', 'green', 'shirts', 'are', 'standing', 'in', 'a', 'yard'], 'raw': 'Two men in green shirts are standing in a yard.', 'imgid': 0, 'sentid': 2, 'deptree': [[1, 0], [6, 1], [4, 2], [4, 3], [1, 4], [6, 5], [-1, 6], [9, 7], [9, 8], [6, 9], [6, 10]]}, {'tokens': ['a', 'man', 'in', 'a', 'blue', 'shirt', 'standing', 'in', 'a', 'garden'], 'raw': 'A man in a blue shirt standing in a garden.', 'imgid': 0, 'sentid': 3, 'deptree': [[1, 0], [-1, 1], [5, 2], [5, 3], [5, 4], [1, 5], [1, 6], [9, 7], [9, 8], [6, 9], [1, 10]]}, {'tokens': ['two', 'friends', 'enjoy', 'time', 'spent', 'together'], 'raw': 'Two friends enjoy time spent together.', 'imgid': 0, 'sentid': 4, 'deptree': [[1, 0], [2, 1], [-1, 2], [2, 3], [3, 4], [4, 5], [2, 6]]}], 'split': 'train', 'filename': '1000092795.jpg'}

    """
    data_dict = {}
    data_dict["sentids"]=[id]
    data_dict["imgid"] = id

    data_dict['sentences'] = []
    one_sent = {}
    one_sent['raw'] = sentence_str
    one_sent['imgid'] = id
    one_sent['sentid'] = id
    data_dict['sentences'].append(one_sent)

    data_dict['split'] = 'train'
    data_dict['filename'] = str(id)+'.jpg'

    return data_dict

def write_img_det():

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    device = "cuda:0"
    img_encoder = VggEncoder()
    img_encoder = img_encoder.to(device)
    mode = 'train'
    if mode == 'dev':
        src_sent_file = '/home/meihuan2/document/MMSS4.0/corpus/dev_sent.txt'
        src_img_dir = '/home/meihuan2/document/MMSS4.0/corpus/images_dev'
        tgt_img_dir = '/data/meihuan2/dataset/SS_MLM/images_dev'
        tgt_sentence_file = '/data/meihuan2/dataset/SS_MLM/dev_sent.json'
        tgt_valid_file = '/data/meihuan2/dataset/SS_MLM/dev_valid.json'
    elif mode == 'train':
        src_sent_file = '/home/meihuan2/document/MMSS4.0/corpus/train_sent.txt'
        src_img_dir = '/home/meihuan2/document/MMSS4.0/corpus/images_train'
        tgt_img_dir = '/data/meihuan2/dataset/SS_MLM/images_train'
        tgt_sentence_file = '/data/meihuan2/dataset/SS_MLM/train_sent.json'
        tgt_valid_file = '/data/meihuan2/dataset/SS_MLM/train_valid.json'
    else:
        src_sent_file = '/home/meihuan2/document/MMSS4.0/corpus/test_sent.txt'
        src_img_dir = '/home/meihuan2/document/MMSS4.0/corpus/images_test'
        tgt_img_dir = '/data/meihuan2/dataset/SS_MLM/images_test'
        tgt_sentence_file = '/data/meihuan2/dataset/SS_MLM/test_sent.json'
        tgt_valid_file = '/data/meihuan2/dataset/SS_MLM/test_valid.json'


    pts = glob.glob(src_img_dir+'/*.jpg')
    src_sentences = open(src_sent_file,'r').readlines()
    alldata = []
    valid_data_dict = {}

    for i, img_pth in enumerate(pts):
        if i%1000==0:
            print("{}/{} and cur image path is:{}".format(i,len(pts),img_pth))
        img_id = img_pth.split('/')[-1].split('.')[0]
        img_id = int(img_id)
        sentence_str = src_sentences[img_id-1]
        local_outputs, _, _ = get_img_det(img_pth, transform, img_encoder, device)
        local_outputs = local_outputs.squeeze(0).cpu().numpy()
        # print(local_outputs.shape)
        cur_sent = write_one_sentence(sentence_str, img_id)

        alldata.append(cur_sent)
        valid_data_dict[str(img_id + 1) + '.jpg'] = str(img_id + 1) + '.jpg'
        np.save(tgt_img_dir+'/'+str(img_id)+'.npy', local_outputs)

    with open(tgt_sentence_file , 'w') as f:
        json.dump({'images':alldata,'dataset':'SS'},f)
        f.close()

    with open(tgt_valid_file, 'w') as f:
        json.dump(valid_data_dict, f)
        f.close()


if __name__ == '__main__':
    write_img_det()


