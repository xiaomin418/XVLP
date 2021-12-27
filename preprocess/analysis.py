from configs import config as myconfig
import pickle
from nltk.corpus import stopwords
import numpy as np
import math
en_stops = stopwords.words('english')

def write_multimodal_better_pic():
    res_root = '/data/meihuan2/dataset/SS_MLM/'  # /data/meihuan2/dataset/SS_MLM/1213-testnpy
    # devnpy_list = ['devnpy', 'devnpy2', 'testnpy', 'testnpy2']
    devnpy_list = ['1213-testnpy', '1213-testnpy2', '1213-devnpy3']  #
    words_proj_img = {}
    for dev in devnpy_list:
        multi_file = res_root + dev + '/result_multimodal.pickle'
        text_file = res_root + dev + '/result_textonly.pickle'
        multiresult = pickle.load(open(multi_file, 'rb'))
        textresult = pickle.load(open(text_file, 'rb'))
        if 'dev' in dev:
            cur_root = './images_dev/'
        else:
            cur_root = './images_test/'

        for mr, tr in zip(multiresult, textresult):
            assert mr['img_id'] == tr['img_id']
            assert mr['mask_lm'] == tr['mask_lm']
            for mw, tw, gw in zip(mr['predict_lm'], tr['predict_lm'], mr['mask_lm']):
                if mw == gw and tw != gw:
                    if mw in words_proj_img:
                        words_proj_img[mw].append(cur_root + str(mr['img_id']) + '.jpg')
                    else:
                        words_proj_img[mw] = [cur_root + str(mr['img_id']) + '.jpg']

    for k in en_stops:
        if k in words_proj_img:
            words_proj_img.pop(k)

    dev_better_imgs = []
    print(sorted(words_proj_img.items(), key=lambda x: len(x[1]), reverse=True))
    print("len of img words ", len(words_proj_img))
    for k, v in words_proj_img.items():
        for pth in v:
            if 'dev' not in pth:
                dev_better_imgs.append(pth)
    print(dev_better_imgs)

def save_multimodal_better_pic(devnpy_dir, better_path):
    words_proj_img = {}

    multi_file =  devnpy_dir + '/result_multimodal.pickle'
    text_file =  devnpy_dir + '/result_textonly.pickle'
    multiresult = pickle.load(open(multi_file, 'rb'))
    textresult = pickle.load(open(text_file, 'rb'))

    for mr, tr in zip(multiresult, textresult):
        assert mr['img_id'] == tr['img_id']
        assert mr['mask_lm'] == tr['mask_lm']
        for mw, tw, gw in zip(mr['predict_lm'], tr['predict_lm'], mr['mask_lm']):
            if mw == gw and tw != gw:
                if mw in words_proj_img:
                    words_proj_img[mw].append(str(mr['img_id']) + '.jpg')
                else:
                    words_proj_img[mw] = [str(mr['img_id']) + '.jpg']

    for k in en_stops:
        if k in words_proj_img:
            words_proj_img.pop(k)

    print("len of img words ", len(words_proj_img))
    dev_better_imgs = []
    for k,v in words_proj_img.items():
        for ig in v:
            dev_better_imgs.append(ig)
    print("len of images: ",len(dev_better_imgs))
    with open(better_path, 'wb') as fb:
        pickle.dump(dev_better_imgs, fb)
        fb.close()

def merge_batch(cand_batch_result_files):
    # 合并所有cand_mask结果，获得multimodal(或textonly)对每个样本预测的总分数
    data0 = pickle.load(open(cand_batch_result_files[0], 'rb'))
    data1 = pickle.load(open(cand_batch_result_files[1], 'rb'))
    scores_dict = dict()
    nums_dict = dict()

    merge_scores = dict()
    for data in [data0, data1]:
        for d in data:
            img_id = d['img_id']
            if img_id not in scores_dict:
                scores_dict[img_id] = [d['score']]
                nums_dict[img_id] = [d['mask_num']]
            else:
                scores_dict[img_id].append(d['score'])
                nums_dict[img_id].append(d['mask_num'])

    for k,sc in scores_dict.items():
        num = np.array(nums_dict[k])
        sc = np.array(sc)
        rc = np.sum(sc*num)/np.sum(num)
        merge_scores[k] = rc

    return merge_scores

def merge_contri_scores(merge_multi, merget_text):
    last_scs = dict()
    for k,v in merge_multi.items():
        if k in merget_text:
            soft_s = soft_method(v - merget_text[k])
            last_scs[str(k)+'.jpg'] = soft_s
    return last_scs

def soft_method(x):
    t = (-1)*x
    return 1/(1+math.exp(t))
# analysis
# write_multimodal_better_pic()


# to generate training data
### 1. for training data
# devnpy_dir = '/data/meihuan2/dataset/SS_MLM/1213-trainnpy'
# better_path = devnpy_dir + '/useful_pic_path.pickle'
# save_multimodal_better_pic(devnpy_dir,better_path)

### 2. for dev data
# devnpy_dir = '/data/meihuan2/dataset/SS_MLM/1213-devnpy3'
# better_path = devnpy_dir + '/useful_pic_path.pickle'
# save_multimodal_better_pic(devnpy_dir,better_path)

### 3. for test data
# devnpy_dir = '/data/meihuan2/dataset/SS_MLM/1213-testnpy'
# better_path = devnpy_dir + '/useful_pic_path.pickle'
# save_multimodal_better_pic(devnpy_dir,better_path)

# merge all cand masks results
### 1. for training data
cand_batch_textonly_files = [
    '/data/meihuan2/dataset/SS_MLM/1227-trainnpy1/result_textonly1.pickle',
    '/data/meihuan2/dataset/SS_MLM/1227-trainnpy1/result_textonly2.pickle',
    '/data/meihuan2/dataset/SS_MLM/1227-trainnpy1/result_textonly3.pickle'
]
cand_batch_multimodal_files = [
    '/data/meihuan2/dataset/SS_MLM/1227-trainnpy1/result_multimodal1.pickle',
    '/data/meihuan2/dataset/SS_MLM/1227-trainnpy1/result_multimodal2.pickle',
    '/data/meihuan2/dataset/SS_MLM/1227-trainnpy1/result_multimodal3.pickle'
]
textonly_merge_score = merge_batch(cand_batch_textonly_files)
multimodal_merge_score = merge_batch(cand_batch_multimodal_files)
image_contri_score = merge_contri_scores(multimodal_merge_score, textonly_merge_score)
image_contri_path = '/data/meihuan2/dataset/SS_MLM/1227-trainnpy1/image_contri.pickle'
with open(image_contri_path, 'wb') as f:
    pickle.dump(image_contri_score, f)
    f.close()

### 2. for dev data
cand_batch_textonly_files = [
    '/data/meihuan2/dataset/SS_MLM/1227-devnpy/result_textonly1.pickle',
    '/data/meihuan2/dataset/SS_MLM/1227-devnpy/result_textonly2.pickle',
    '/data/meihuan2/dataset/SS_MLM/1227-devnpy/result_textonly3.pickle'
]
cand_batch_multimodal_files = [
    '/data/meihuan2/dataset/SS_MLM/1227-devnpy/result_multimodal1.pickle',
    '/data/meihuan2/dataset/SS_MLM/1227-devnpy/result_multimodal2.pickle',
    '/data/meihuan2/dataset/SS_MLM/1227-devnpy/result_multimodal3.pickle'
]
textonly_merge_score = merge_batch(cand_batch_textonly_files)
multimodal_merge_score = merge_batch(cand_batch_multimodal_files)
image_contri_score = merge_contri_scores(multimodal_merge_score, textonly_merge_score)
image_contri_path = '/data/meihuan2/dataset/SS_MLM/1227-devnpy/image_contri.pickle'
with open(image_contri_path, 'wb') as f:
    pickle.dump(image_contri_score, f)
    f.close()
