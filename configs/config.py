
vis_embedding_dim = 512
len_vis_input = 49
vis_input = True
pretrain_mode = 'test' # 'test'/'train'/'test_td'(test train data)
dev_data_gen = False
dev_data_file_load = True
return_the_img_path = True # return_the_img_path = True when pretrain_mode == 'test' or pretrain_mode == 'test_td'
dev_npy_dir = '/data/meihuan2/dataset/SS_MLM/1227-devnpy/'
result_multimodal_file = '/data/meihuan2/dataset/SS_MLM/1227-devnpy/result_multimodal.pickle'
result_textonly_file = '/data/meihuan2/dataset/SS_MLM/1227-devnpy/result_textonly.pickle'
dev_sent_file = '/home/meihuan2/document/MMSS4.0/corpus/dev_sent.txt'

#tokenizer
vocab_size = 28996
tokenizer_name = 'mytokenizer' # use 'mytokenizer' or 'bertokenizer'
