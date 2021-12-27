# Prepare your own data

To preprocess our own data to the this model's format, we write the ./preprocess/*. Run:

```
python ./preprocess/prepare_data.py
```

the preprocessing data is in  '/data/meihuan2/dataset/SS_MLM/devnpy/' (dev_npy_dir)



# Pre-training

1. For textonly training:
```angular2html
set the ./configs/config 
vis_input=False
len_vis_input = 49
vis_embedding_dim 512
return_the_img_path = False
>> python vlp/run_img2txt_dist.py --output_dir /data/meihuan2/SS_MLM/1213textonly \
    --model_recover_path /data/meihuan2/dataset/flickr30k_g8_lr3e-5_batch512_ft_from_s0.75_b0.25/model.21.bin \
    --do_train --learning_rate 1e-4 --new_segment_ids --always_truncate_tail --amp \
    --src_file /data/meihuan2/dataset/SS_MLM/train_sent.json \
    --dataset cc --split train --file_valid_jpgs /data/meihuan2/dataset/SS_MLM/train_valid.json \
    --local_rank -1 --global_rank -1 --world_size 1 --enable_butd \
    --s2s_prob 0.75 --bi_prob 0.25 --image_root /data/meihuan2/dataset/SS_MLM/images_train \
    --region_det_file_prefix ''
```
2. For multimodal training:
```angular2html
set the ./configs/config 
vis_input=True
len_vis_input = 49
vis_embedding_dim 512
return_the_img_path = False
>> python vlp/run_img2txt_dist.py --output_dir /data/meihuan2/SS_MLM/1213multimodal \
    --model_recover_path /data/meihuan2/SS_MLM/model.4.bin \
    --do_train --learning_rate 1e-4 --new_segment_ids --always_truncate_tail --amp \
    --src_file /data/meihuan2/dataset/SS_MLM/train_sent.json \
    --dataset cc --split train --file_valid_jpgs /data/meihuan2/dataset/SS_MLM/train_valid.json \
    --local_rank -1 --global_rank -1 --world_size 1 --enable_butd \
    --s2s_prob 0.75 --bi_prob 0.25 --image_root /data/meihuan2/dataset/SS_MLM/images_train \
    --region_det_file_prefix ''
```
# Pre-training test
1. First generate the fixed testing data: set the ./configs/config 
```angular2html
pretrain_mode = 'test'
dev_data_gen = True
```
And remember to revise the ./pytorch_pretrained_bert/modeling.py in +1156
```angular2html
Origin: return masked_lm_loss, vis_pretext_loss, masked_lm_loss.new(1).fill_(0)
Revise to: return masked_lm_loss, torch.argmax(prediction_scores_masked,dim=2), masked_lm_loss.new(1).fill_(0)
```
2. textonly test:
```angular2html
>>python vlp/eval_img2txt_dist.py --output_dir /data/meihuan2/SS_MLM \
    --model_recover_path /data/meihuan2/SS_MLM/1213textonly/model.5.bin \
    --do_train --learning_rate 1e-4 --new_segment_ids --always_truncate_tail --amp \
    --src_file /data/meihuan2/dataset/SS_MLM/dev_sent.json \
    --dataset cc --split train --file_valid_jpgs /data/meihuan2/dataset/SS_MLM/dev_valid.json \
    --local_rank -1 --global_rank -1 --world_size 1 --enable_butd \
    --s2s_prob 0.75 --bi_prob 0.25 --image_root /data/meihuan2/dataset/SS_MLM/images_dev \
    --region_det_file_prefix ''  --num_train_epochs 1
>>python vlp/eval_img2txt_dist.py --output_dir /data/meihuan2/SS_MLM \
    --model_recover_path /data/meihuan2/SS_MLM/1213textonly/model.5.bin \
    --do_train --learning_rate 1e-4 --new_segment_ids --always_truncate_tail --amp \
    --src_file /data/meihuan2/dataset/SS_MLM/test_sent.json \
    --dataset cc --split train --file_valid_jpgs /data/meihuan2/dataset/SS_MLM/test_valid.json \
    --local_rank -1 --global_rank -1 --world_size 1 --enable_butd \
    --s2s_prob 0.75 --bi_prob 0.25 --image_root /data/meihuan2/dataset/SS_MLM/images_test \
    --region_det_file_prefix ''  --num_train_epochs 1
```
3. multimodal test:
```angular2html
>>python vlp/eval_img2txt_dist.py --output_dir /data/meihuan2/SS_MLM \
    --model_recover_path /data/meihuan2/SS_MLM/1209multimodal/model.28.bin \
    --do_train --learning_rate 1e-4 --new_segment_ids --always_truncate_tail --amp \
    --src_file /data/meihuan2/dataset/SS_MLM/dev_sent.json \
    --dataset cc --split train --file_valid_jpgs /data/meihuan2/dataset/SS_MLM/dev_valid.json \
    --local_rank -1 --global_rank -1 --world_size 1 --enable_butd \
    --s2s_prob 0.75 --bi_prob 0.25 --image_root /data/meihuan2/dataset/SS_MLM/images_dev \
    --region_det_file_prefix ''  --num_train_epochs 1
>>python vlp/eval_img2txt_dist.py --output_dir /data/meihuan2/SS_MLM \
    --model_recover_path /data/meihuan2/SS_MLM/1209multimodal/model.28.bin \
    --do_train --learning_rate 1e-4 --new_segment_ids --always_truncate_tail --amp \
    --src_file /data/meihuan2/dataset/SS_MLM/test_sent.json \
    --dataset cc --split train --file_valid_jpgs /data/meihuan2/dataset/SS_MLM/test_valid.json \
    --local_rank -1 --global_rank -1 --world_size 1 --enable_butd \
    --s2s_prob 0.75 --bi_prob 0.25 --image_root /data/meihuan2/dataset/SS_MLM/images_test \
    --region_det_file_prefix ''  --num_train_epochs 1
```
4. You can also use the 'test_textonly.sh' and 'test_multimodal.sh' to test 
from model.1.bin to model.30.bin. And the test_textonly.sh is:
```angular2html
#!/bin/bash
 
for i in $(seq 1 30)
do 
python vlp/eval_img2txt_dist.py --output_dir /data/meihuan2/SS_MLM \
    --model_recover_path /data/meihuan2/SS_MLM/1209textonly/model.${i}.bin \
    --do_train --learning_rate 1e-4 --new_segment_ids --always_truncate_tail --amp \
    --src_file /data/meihuan2/dataset/SS_MLM/dev_sent.json \
    --dataset cc --split train --file_valid_jpgs /data/meihuan2/dataset/SS_MLM/dev_valid.json \
    --local_rank -1 --global_rank -1 --world_size 1 --enable_butd \
    --s2s_prob 0.75 --bi_prob 0.25 --image_root /data/meihuan2/dataset/SS_MLM/images_dev \
    --region_det_file_prefix ''  --num_train_epochs 1;
done
```



# Finetune for vqa

1. Finetune

   ```
   python vlp/run_img2txt_dist.py --output_dir /data/meihuan2/SS_finetune \
       --model_recover_path /data/meihuan2/SS_MLM/model.5.bin \
       --do_train --scst --learning_rate 1e-6 --new_segment_ids --always_truncate_tail --amp \
       --num_train_epochs 20 --enable_butd --s2s_prob 0 --bi_prob 1 \
       --image_root /data/meihuan2/dataset/coco/detectron_fix_100/fc6 \
       --tasks vqa2 --src_file /data/meihuan2/dataset/vqa/imdb_train2014.npy \
       --file_valid_jpgs /data/meihuan2/dataset/flickr30k/annotations/flickr30k_valid_jpgs.json \
       --mask_prob 0 --max_pred 0 --region_det_file_prefix ''
   ```





The vocabulary dictionary is in the "/data/meihuan2/SS_MLM/.pretrained_model_-1/5e8a2b4893d13790ed4150ca1906be5f7a03d6c4ddf62296c383f6db42814db2.e13dbb970cb325137104fb2e5f36fe865f27746c6b526f6352861b1980eb80b1"



#日志：

# 12.09日训练

结果保存于/data/meihuan2/SS_MLM/1209multimodal/和/data/meihuan2/SS_MLM/1209textonly/

采用二者model.进行测试28.bin

测试时，（1）用eval_img2txt.py测试，首先进行多模态测试

```
vis_input = True
pretrain_mode = 'test'
dev_data_gen = True
dev_data_file_load = True
return_the_img_path = True # return_the_img_path = True when pretrain_mode == 'test'
dev_npy_dir = '/data/meihuan2/dataset/SS_MLM/$testnpyxxx$/'
result_multimodal_file = '/data/meihuan2/dataset/SS_MLM/$testnpyxxx$/result_multimodal.pickle'
result_textonly_file = '/data/meihuan2/dataset/SS_MLM/$testnpyxxx$/result_textonly.pickle'
dev_sent_file = '/home/meihuan2/document/MMSS4.0/corpus/test_sent.txt'

>>python vlp/eval_img2txt_dist.py --output_dir /data/meihuan2/SS_MLM \
    --model_recover_path /data/meihuan2/SS_MLM/1209multimodal/model.28.bin \
    --do_train --learning_rate 1e-4 --new_segment_ids --always_truncate_tail --amp \
    --src_file /data/meihuan2/dataset/SS_MLM/test_sent.json \
    --dataset cc --split train --file_valid_jpgs /data/meihuan2/dataset/SS_MLM/test_valid.json \
    --local_rank -1 --global_rank -1 --world_size 1 --enable_butd \
    --s2s_prob 0.75 --bi_prob 0.25 --image_root /data/meihuan2/dataset/SS_MLM/images_test \
    --region_det_file_prefix ''  --num_train_epochs 1
```

测试结果保存于”myconfig.result_multimodal_file“ 和 ”myconfig.result_textonly_file“

（2）然后进行纯文本测试：

```
vis_input = False
pretrain_mode = 'test'
dev_data_gen = False
dev_data_file_load = True
return_the_img_path = True # return_the_img_path = True when pretrain_mode == 'test'
dev_npy_dir = '/data/meihuan2/dataset/SS_MLM/$testnpyxxx$/'
result_multimodal_file = '/data/meihuan2/dataset/SS_MLM/$testnpyxxx$/result_multimodal.pickle'
result_textonly_file = '/data/meihuan2/dataset/SS_MLM/$testnpyxxx$/result_textonly.pickle'
dev_sent_file = '/home/meihuan2/document/MMSS4.0/corpus/test_sent.txt'

>>python vlp/eval_img2txt_dist.py --output_dir /data/meihuan2/SS_MLM \
    --model_recover_path /data/meihuan2/SS_MLM/1209testonly/model.28.bin \
    --do_train --learning_rate 1e-4 --new_segment_ids --always_truncate_tail --amp \
    --src_file /data/meihuan2/dataset/SS_MLM/test_sent.json \
    --dataset cc --split train --file_valid_jpgs /data/meihuan2/dataset/SS_MLM/test_valid.json \
    --local_rank -1 --global_rank -1 --world_size 1 --enable_butd \
    --s2s_prob 0.75 --bi_prob 0.25 --image_root /data/meihuan2/dataset/SS_MLM/images_test \
    --region_det_file_prefix ''  --num_train_epochs 1
```



3. 如果希望获得更多的测试mask单词，手动修改”seq2seq_loader.py“ +265 行：

```
origin: masked_pos = cand_pos[:2*n_pred]
extending1: masked_pos = cand_pos[n_pred:2*n_pred]
extending2: masked_pos = cand_pos[2*n_pred:3*n_pred]
```



分割线

-----



# 12.13日训练

textonly训练 pid(2952) 1213textonly.log

```
set the ./configs/config 
vis_input=False
len_vis_input = 49
vis_embedding_dim 512
return_the_img_path = False
>> python vlp/run_img2txt_dist.py --output_dir /data/meihuan2/SS_MLM/1213textonly \
    --model_recover_path /data/meihuan2/dataset/flickr30k_g8_lr3e-5_batch512_ft_from_s0.75_b0.25/model.21.bin \
    --do_train --learning_rate 1e-4 --new_segment_ids --always_truncate_tail --amp \
    --src_file /data/meihuan2/dataset/SS_MLM/train_sent.json \
    --dataset cc --split train --file_valid_jpgs /data/meihuan2/dataset/SS_MLM/train_valid.json \
    --local_rank -1 --global_rank -1 --world_size 1 --enable_butd \
    --s2s_prob 0.75 --bi_prob 0.25 --image_root /data/meihuan2/dataset/SS_MLM/images_train \
    --region_det_file_prefix ''
```



multimodal训练 pid(4694) 1213multimodal.log

```
set the ./configs/config 
vis_input=True
len_vis_input = 49
vis_embedding_dim 512
return_the_img_path = False

>> python vlp/run_img2txt_dist.py --output_dir /data/meihuan2/SS_MLM/1213multimodal \
    --model_recover_path /data/meihuan2/dataset/flickr30k_g8_lr3e-5_batch512_ft_from_s0.75_b0.25/model.21.bin \
    --do_train --learning_rate 1e-4 --new_segment_ids --always_truncate_tail --amp \
    --src_file /data/meihuan2/dataset/SS_MLM/train_sent.json \
    --dataset cc --split train --file_valid_jpgs /data/meihuan2/dataset/SS_MLM/train_valid.json \
    --local_rank -1 --global_rank -1 --world_size 1 --enable_butd \
    --s2s_prob 0.75 --bi_prob 0.25 --image_root /data/meihuan2/dataset/SS_MLM/images_train \
    --region_det_file_prefix ''
```



textonly测试：

```
vis_embedding_dim = 512
len_vis_input = 49
vis_input = False
pretrain_mode = 'test'
dev_data_gen = True
dev_data_file_load = False
return_the_img_path = True # return_the_img_path = True when pretrain_mode == 'test'
dev_npy_dir = '/data/meihuan2/dataset/SS_MLM/1213-testnpy/'
result_multimodal_file = '/data/meihuan2/dataset/SS_MLM/1213-testnpy/result_multimodal.pickle'
result_textonly_file = '/data/meihuan2/dataset/SS_MLM/1213-testnpy/result_textonly.pickle'
dev_sent_file = '/home/meihuan2/document/MMSS4.0/corpus/test_sent.txt'

#tokenizer
vocab_size = 28996
tokenizer_name = 'mytokenizer' # use 'mytokenizer' or 'bertokenizer'

(test dataset)
>>python vlp/eval_img2txt_dist.py --output_dir /data/meihuan2/SS_MLM     --model_recover_path /data/meihuan2/SS_MLM/1213textonly/model.30.bin     --do_train --learning_rate 1e-4 --new_segment_ids --always_truncate_tail --amp     --src_file /data/meihuan2/dataset/SS_MLM/test_sent.json     --dataset cc --split train --file_valid_jpgs /data/meihuan2/dataset/SS_MLM/test_valid.json     --local_rank -1 --global_rank -1 --world_size 1 --enable_butd     --s2s_prob 0.75 --bi_prob 0.25 --image_root /data/meihuan2/dataset/SS_MLM/images_test     --region_det_file_prefix ''  --num_train_epochs 1

(dev dataset)
>> python vlp/eval_img2txt_dist.py --output_dir /data/meihuan2/SS_MLM     --model_recover_path /data/meihuan2/SS_MLM/1213textonly/model.30.bin     --do_train --learning_rate 1e-4 --new_segment_ids --always_truncate_tail --amp     --src_file /data/meihuan2/dataset/SS_MLM/dev_sent.json     --dataset cc --split train --file_valid_jpgs /data/meihuan2/dataset/SS_MLM/dev_valid.json     --local_rank -1 --global_rank -1 --world_size 1 --enable_butd     --s2s_prob 0.75 --bi_prob 0.25 --image_root /data/meihuan2/dataset/SS_MLM/images_dev     --region_det_file_prefix ''  --num_train_epochs 1

(train dataset)
>> python vlp/eval_img2txt_dist.py --output_dir /data/meihuan2/SS_MLM     --model_recover_path /data/meihuan2/SS_MLM/1213textonly/model.30.bin     --do_train --learning_rate 1e-4 --new_segment_ids --always_truncate_tail --amp     --src_file /data/meihuan2/dataset/SS_MLM/train_sent.json     --dataset cc --split train --file_valid_jpgs /data/meihuan2/dataset/SS_MLM/train_valid.json     --local_rank -1 --global_rank -1 --world_size 1 --enable_butd     --s2s_prob 0.75 --bi_prob 0.25 --image_root /data/meihuan2/dataset/SS_MLM/images_train    --region_det_file_prefix ''  --num_train_epochs 1
```



multimodal测试：

```
vis_embedding_dim = 512
len_vis_input = 49
vis_input = True
pretrain_mode = 'test'
dev_data_gen = False
dev_data_file_load = True
return_the_img_path = True # return_the_img_path = True when pretrain_mode == 'test'
dev_npy_dir = '/data/meihuan2/dataset/SS_MLM/1213-testnpy/'
result_multimodal_file = '/data/meihuan2/dataset/SS_MLM/1213-testnpy/result_multimodal.pickle'
result_textonly_file = '/data/meihuan2/dataset/SS_MLM/1213-testnpy/result_textonly.pickle'
dev_sent_file = '/home/meihuan2/document/MMSS4.0/corpus/test_sent.txt'

#tokenizer
vocab_size = 28996
tokenizer_name = 'mytokenizer' # use 'mytokenizer' or 'bertokenizer'

(test dataset)
>> python vlp/eval_img2txt_dist.py --output_dir /data/meihuan2/SS_MLM     --model_recover_path /data/meihuan2/SS_MLM/1213multimodal/model.30.bin     --do_train --learning_rate 1e-4 --new_segment_ids --always_truncate_tail --amp     --src_file /data/meihuan2/dataset/SS_MLM/test_sent.json     --dataset cc --split train --file_valid_jpgs /data/meihuan2/dataset/SS_MLM/test_valid.json     --local_rank -1 --global_rank -1 --world_size 1 --enable_butd     --s2s_prob 0.75 --bi_prob 0.25 --image_root /data/meihuan2/dataset/SS_MLM/images_test     --region_det_file_prefix ''  --num_train_epochs 1

(dev dataset)
>> python vlp/eval_img2txt_dist.py --output_dir /data/meihuan2/SS_MLM     --model_recover_path /data/meihuan2/SS_MLM/1213multimodal/model.30.bin     --do_train --learning_rate 1e-4 --new_segment_ids --always_truncate_tail --amp     --src_file /data/meihuan2/dataset/SS_MLM/dev_sent.json     --dataset cc --split train --file_valid_jpgs /data/meihuan2/dataset/SS_MLM/dev_valid.json     --local_rank -1 --global_rank -1 --world_size 1 --enable_butd     --s2s_prob 0.75 --bi_prob 0.25 --image_root /data/meihuan2/dataset/SS_MLM/images_dev     --region_det_file_prefix ''  --num_train_epochs 1

(train dataset)
>> python vlp/eval_img2txt_dist.py --output_dir /data/meihuan2/SS_MLM     --model_recover_path /data/meihuan2/SS_MLM/1213multimodal/model.30.bin     --do_train --learning_rate 1e-4 --new_segment_ids --always_truncate_tail --amp     --src_file /data/meihuan2/dataset/SS_MLM/train_sent.json     --dataset cc --split train --file_valid_jpgs /data/meihuan2/dataset/SS_MLM/train_valid.json     --local_rank -1 --global_rank -1 --world_size 1 --enable_butd     --s2s_prob 0.75 --bi_prob 0.25 --image_root /data/meihuan2/dataset/SS_MLM/images_train    --region_det_file_prefix ''  --num_train_epochs 1
```

