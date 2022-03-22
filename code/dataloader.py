#!/home/mtran/anaconda3/bin/python
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import os
import random
import gensim
from gensim import corpora
import pandas as pd
import ast
from multiprocessing import Pool, Process, Manager

class DiscreteDataset(Dataset):
    def __init__(self, data_path, vocab_size, seq_len, p_mask, consecutive):
        # self.vocab = [str(i) for i in range(vocab_size)]
        # self.vocab += ['<ignore>', '<oov>', '<mask>']

        # self.vocab = {e:i for i, e in enumerate(self.vocab)}
        # self.rvocab = {v:k for k,v in self.vocab.items()}
        self.IGNORE_IDX = vocab_size + 1 #replacement tag for tokens to ignore
        self.OUT_OF_VOCAB_IDX = vocab_size + 2 #replacement tag for unknown words
        self.MASK_IDX = vocab_size + 3 #replacement tag for the masked word prediction task

        self.vocab = {str(i):i for i in range(vocab_size)}
        self.vocab['<ignore>'] = self.IGNORE_IDX
        self.vocab['<oov>'] = self.OUT_OF_VOCAB_IDX
        self.vocab['<mask>'] = self.MASK_IDX
        self.rvocab = {v:k for k,v in self.vocab.items()}

        self.seq_len = seq_len
        self.p_mask = p_mask
        self.consecutive = consecutive

        #special tags
        # self.IGNORE_IDX = self.vocab['<ignore>'] #replacement tag for tokens to ignore
        # self.OUT_OF_VOCAB_IDX = self.vocab['<oov>'] #replacement tag for unknown words
        # self.MASK_IDX = self.vocab['<mask>'] #replacement tag for the masked word prediction task

        #load full text dataset
        with open(data_path) as f:
            content = f.readlines()

        self.data = [x.strip() for x in content]


    def __getitem__(self, index):
        current_seq = self.data[index].split(' ')
        s = []
        for i in range(self.seq_len):
            if(i <= len(current_seq) - 1):
                try:
                    s.append(self.vocab[current_seq[i]])
                except:
                    s.append(self.OUT_OF_VOCAB_IDX)
            else:
                s.append(self.IGNORE_IDX) #padding
        #apply random mask
        sp, idx = [], 0
        while(idx < len(s)):
            w = s[idx]
            if(random.random() < self.p_mask / self.consecutive):
                for j in range(self.consecutive):
                    sp.append((self.MASK_IDX, w))
                    if(len(sp) == self.seq_len):
                        break
                idx += self.consecutive
            else:
                sp.append((w, self.IGNORE_IDX))
                idx += 1

        # sp = [(self.MASK_IDX, w) if random.random() < self.p_mask else (w, self.IGNORE_IDX) for w in s]

        return {'input': torch.Tensor([w[0] for w in sp]).long(),
                'target': torch.Tensor([w[1] for w in sp]).long()}

    def __len__(self):
        return len(self.data)

class DiscreteAVDatasetNew(Dataset):
    def __init__(self, audio_path, video_path, seq_len, nvocab, p_mask=0.15):
        if(not os.path.isfile('../data/dictionary_token2id_audio'+ str(nvocab) +'.csv') or not os.path.isfile('../data/dictionary_token2id_video'+ str(nvocab) +'.csv')):
            with open(audio_path) as f:
                content_audio = f.readlines() 
            tokens_audio = [[token for token in sentence.replace('\n', '').split()] for sentence in content_audio]
            tokens_audio.insert(0, ['<ignore>', '<cls>', '<mask>'])
            dictionary_audio = corpora.Dictionary(tokens_audio)

            with open(video_path) as f:
                content_video = f.readlines()
            tokens_video = [[token for token in sentence.replace('\n', '').split()] for sentence in content_video]
            tokens_video.insert(0, ['<ignore>', '<cls>', '<mask>'])
            dictionary_video = corpora.Dictionary(tokens_video)

            assert len(content_audio) == len(content_video) # numbers of videos processed in audio and video are the same
            self.seq_len = seq_len
            self.p_mask = p_mask

            audio_data, video_data = [], []
            for i in range(len(content_audio)):
                current_a, current_v = content_audio[i], content_video[i]
                items_a = [int(x) for x in current_a.replace('\n', '').split()]
                items_v = [int(x) for x in current_v.replace('\n', '').split()]
                keep = True
                for item_a in items_a:
                    if(item_a > nvocab):
                        keep = False
                for item_v in items_v:
                    if(item_v > nvocab):
                        keep = False
                if(keep):
                    audio_data.append(current_a)
                    video_data.append(current_v)

            self.audio_data, self.video_data = audio_data, video_data
            #saving dictionary
            output_a, output_v = [], []
            for k, v in dictionary_audio.items():
                output_a.append([k, v])
            pd.DataFrame(output_a).to_csv('../data/dictionary_token2id_audio'+ str(nvocab) +'.csv', header=None, index=False)
            for k, v in dictionary_video.items():
                output_v.append([k, v])        
            pd.DataFrame(output_v).to_csv('../data/dictionary_token2id_video'+ str(nvocab) +'.csv', header=None, index=False)
            self.dictionary_audio, self.dictionary_video = {v: k for k, v in dict(dictionary_audio).items()}, {v: k for k, v in dict(dictionary_video).items()}
        else:
            audio_token2id, video_token2id = pd.read_csv('../data/dictionary_token2id_audio'+ str(nvocab) +'.csv', header=None).values, \
                pd.read_csv('../data/dictionary_token2id_video'+ str(nvocab) +'.csv', header=None).values
            self.dictionary_audio, self.dictionary_video = {}, {}
            for row in audio_token2id:
                self.dictionary_audio[str(row[1])] = row[0]
            for row in video_token2id:
                self.dictionary_video[str(row[1])] = row[0]  
            self.seq_len = seq_len
            self.p_mask = p_mask  
            with open(audio_path) as f:
                content_audio = f.readlines()  
            with open(video_path) as f:
                content_video = f.readlines()  

            audio_data, video_data = [], []
            for i in range(len(content_audio)):
                current_a, current_v = content_audio[i], content_video[i]
                items_a = [int(x) for x in current_a.replace('\n', '').split()]
                items_v = [int(x) for x in current_v.replace('\n', '').split()]
                keep = True
                for item_a in items_a:
                    if(item_a > nvocab):
                        keep = False
                for item_v in items_v:
                    if(item_v > nvocab):
                        keep = False
                if(keep):
                    audio_data.append(current_a)
                    video_data.append(current_v)

            self.audio_data, self.video_data = audio_data, video_data

        self.IGNORE_IDX = self.dictionary_audio['<ignore>']
        self.nvocab = nvocab

    def __getitem__(self, idx):
        a, v = self.audio_data[idx].replace('\n', '').split(), self.video_data[idx].replace('\n', '').split()
        clen = min(self.seq_len, min(len(a), len(v)))
        input_a, target_a, input_v, target_v = [self.dictionary_audio['<cls>']], [self.dictionary_audio['<ignore>']], \
            [self.dictionary_video['<cls>']], [self.dictionary_video['<ignore>']]
        for i in range(clen):
            r = random.random()
            if(r > self.p_mask):
                input_a.append(self.dictionary_audio[a[i]])
                input_v.append(self.dictionary_video[v[i]])
                target_a.append(self.dictionary_audio['<ignore>'])
                target_v.append(self.dictionary_video['<ignore>'])
            else:
                input_a.append(self.dictionary_audio['<mask>'])
                input_v.append(self.dictionary_video['<mask>'])
                target_a.append(self.dictionary_audio[a[i]])
                target_v.append(self.dictionary_video[v[i]])


        while(len(input_a) < self.seq_len):
            input_a.append(self.dictionary_audio['<ignore>'])
            input_v.append(self.dictionary_video['<ignore>'])
            target_a.append(self.dictionary_audio['<ignore>'])
            target_v.append(self.dictionary_video['<ignore>'])  

        return {'input_audio': torch.LongTensor(input_a),
                'target_audio': torch.LongTensor(target_a),
                'input_video': torch.LongTensor(input_v),
                'target_video': torch.LongTensor(target_v)
                }     

    def __len__(self):
        return len(self.audio_data)

class SentimentDataset(Dataset):
    def __init__(self, data_path, seq_len, nvocab):
        self.data = pd.read_csv(data_path, header=None).values
        audio_token2id, video_token2id = pd.read_csv('../data/dictionary_token2id_audio'+ str(nvocab) +'.csv', header=None).values, \
            pd.read_csv('../data/dictionary_token2id_video'+ str(nvocab) +'.csv', header=None).values
        self.dictionary_audio, self.dictionary_video = {}, {}
        for row in audio_token2id:
            self.dictionary_audio[str(row[1])] = row[0]
        for row in video_token2id:
            self.dictionary_video[str(row[1])] = row[0]  
        self.seq_len = seq_len
        self.IGNORE_IDX = self.dictionary_audio['<ignore>']  

    def __getitem__(self, index):
        current_file, current_video, current_audio = self.data[index]
        a, v = current_audio.replace('\n', '').split(), current_video.replace('\n', '').split()
        clen = min(self.seq_len, min(len(a), len(v)))    
        input_a, input_v = [self.dictionary_audio['<cls>']], [self.dictionary_video['<ignore>']]        
        for i in range(clen):
            input_a.append(self.dictionary_audio[a[i]])
            input_v.append(self.dictionary_video[v[i]])


        while(len(input_a) < self.seq_len):
            input_a.append(self.dictionary_audio['<ignore>'])
            input_v.append(self.dictionary_video['<ignore>'])

        return {'input_audio': torch.LongTensor(input_a),
                'input_video': torch.LongTensor(input_v),
                'file_name': current_file
                }     

    def __len__(self):
        return len(self.data)    

class DiscreteAVDataset(Dataset):
    def __init__(self, data_path, vocab_size, seq_len, p_mask, consecutive):
        # self.vocab = [str(i) for i in range(vocab_size)]
        # self.vocab += ['<ignore>', '<oov>', '<mask>']

        # self.vocab = {e:i for i, e in enumerate(self.vocab)}
        # self.rvocab = {v:k for k,v in self.vocab.items()}
        self.IGNORE_IDX = vocab_size + 1 #replacement tag for tokens to ignore
        self.OUT_OF_VOCAB_IDX = vocab_size + 2 #replacement tag for unknown words
        self.MASK_IDX = vocab_size + 3 #replacement tag for the masked word prediction task

        self.vocab = {str(i):i for i in range(vocab_size)}
        self.vocab['<ignore>'] = self.IGNORE_IDX
        self.vocab['<oov>'] = self.OUT_OF_VOCAB_IDX
        self.vocab['<mask>'] = self.MASK_IDX
        self.rvocab = {v:k for k,v in self.vocab.items()}

        self.seq_len = seq_len
        self.p_mask = p_mask
        self.consecutive = consecutive

        #special tags
        # self.IGNORE_IDX = self.vocab['<ignore>'] #replacement tag for tokens to ignore
        # self.OUT_OF_VOCAB_IDX = self.vocab['<oov>'] #replacement tag for unknown words
        # self.MASK_IDX = self.vocab['<mask>'] #replacement tag for the masked word prediction task

        #load full text dataset
        with open(data_path) as f:
            content = f.readlines()

        data = [x.strip() for x in content]
        self.audio_data, self.video_data = [], []
        for row in data:
            audio_part = row.split('\t')[1].strip()
            video_part = row.split('\t')[0].strip()
            self.audio_data.append(audio_part)
            self.video_data.append(video_part)


    def __getitem__(self, index):
        current_seq_a, current_seq_v = self.audio_data[index].split(' '), self.video_data[index].split(' ')
        s_a, s_v = [], []
        for i in range(self.seq_len):
            if(i <= min(len(current_seq_a), len(current_seq_v)) - 1):
                try:
                    s_a.append(self.vocab[current_seq_a[i]])
                except:
                    s_a.append(self.OUT_OF_VOCAB_IDX)
                try:
                    s_v.append(self.vocab[current_seq_a[i]])
                except:
                    s_v.append(self.OUT_OF_VOCAB_IDX)
            else:
                s_a.append(self.IGNORE_IDX) #padding
                s_v.append(self.IGNORE_IDX)
        #apply random mask
        sp, idx = [], 0
        while(idx < len(s_a)):
            w_a, w_v = s_a[idx], s_v[idx]
            if(random.random() < self.p_mask / self.consecutive):
                for j in range(self.consecutive):
                    sp.append((self.MASK_IDX, self.MASK_IDX, w_a, w_v))
                    if(len(sp) == self.seq_len):
                        break
                idx += self.consecutive
            else:
                sp.append((w_a, w_v, self.IGNORE_IDX, self.IGNORE_IDX))
                idx += 1

        # sp = [(self.MASK_IDX, w) if random.random() < self.p_mask else (w, self.IGNORE_IDX) for w in s]

        return {'input_audio': torch.Tensor([w[0] for w in sp]).long(),
                'target_audio': torch.Tensor([w[2] for w in sp]).long(),
                'input_video': torch.Tensor([w[1] for w in sp]).long(),
                'target_video': torch.Tensor([w[3] for w in sp]).long()
                }

    def __len__(self):
        return len(self.audio_data)


class ContinuousDataset(Dataset):
    def __init__(self, datapath, seq_len, p_mask, consecutive):
        #input datapath for large-scale processing
        #TODO: not masking padding frames
        self.datapath = datapath
        self.file_ids = os.listdir(datapath)
        self.seq_len = seq_len
        self.p_mask = p_mask
        self.consecutive = consecutive

    def __getitem__(self, index):
        current_file = os.path.join(self.datapath, self.file_ids[index])
        current_data = pd.read_csv(current_file, header=None).values # TxD
        T, D = current_data.shape

        # ignore_frame = np.array([1 for _ in range(D)])
        ignore_frame = np.zeros(D)
        mask_frame = np.zeros(D)
        s = []
        actual_frame = 0
        for i in range(self.seq_len):
            if(i <= T - 1):
                s.append(current_data[i, :])
                actual_frame += 1
            else:
                s.append(ignore_frame) #padding

        #apply random mask
        sp, idx = [], 0
        while(idx < len(s)):
            w = s[idx]
            if(random.random() < self.p_mask / self.consecutive and idx < actual_frame):
                for j in range(self.consecutive):
                    sp.append((mask_frame, w))
                    if(len(sp) == self.seq_len):
                        break
                idx += self.consecutive
            else:
                sp.append((w, ignore_frame))
                idx += 1
        return {'input': torch.Tensor([w[0] for w in sp]).float(),
                'target': torch.Tensor([w[1] for w in sp]).float()}

    def __len__(self):
        return len(self.file_ids)
    
def read_file(file_info):
    a_abs_path, v_abs_path, seq_len, masking_list = file_info
    audio_data = pd.read_csv(a_abs_path.replace('data','shares'), header=None).values[::1]
    video_data = pd.read_csv(v_abs_path.replace('data','shares'), error_bad_lines=False).values[::6]
    video_data = video_data[:, -35:-18]
    
    audio_data = audio_data[~np.isnan(audio_data).any(axis=1)] #S1 X 512
    video_data = video_data[~np.isnan(video_data).any(axis=1)] #S2 x 17   
    
    audio_data, video_data = np.float16(audio_data), np.float16(video_data)
    return [audio_data, video_data, seq_len, masking_list]

class ContinuousAVDataset(Dataset):
    def __init__(self, metadata_path, seq_len, limit=None):
        self.meta_datapath = metadata_path
        self.seq_len = seq_len
        self.data = pd.read_csv(metadata_path, header=None).values
        self.d_a, self.d_v = 512, 17
        self.ds_a, self.ds_v = 1, 6
        self.a_data, self.v_data = [], []
        self.masking_prop = 0.05 # 3 consecutive so masking 15% of the frames
        self.mask_consecutive = 3
        #cnt = 0
        #for i in range(self.data.shape[0]):
            #a_abs_path, v_abs_path, seq_len, masking_list = self.data[i]
            #audio_data = pd.read_csv(a_abs_path, header=None).values[::self.ds_a]
            #video_data = pd.read_csv(v_abs_path, error_bad_lines=False).values[::self.ds_v]
            #video_data = video_data[:, -35:-18]
            
            #audio_data = audio_data[~np.isnan(audio_data).any(axis=1)] #S1 X 512
            #video_data = video_data[~np.isnan(video_data).any(axis=1)] #S2 x 17   
            
            #audio_data, video_data = np.float16(audio_data), np.float16(video_data)
            #self.a_data.append(audio_data)
            #self.v_data.append(video_data)
            #if(limit and cnt > limit):
                #break
        file_list = [x for x in self.data]
        pool = Pool(40)
        self.all_data = pool.map(read_file, file_list)
        pool.close()
        pool.join()       
        self.data_length = len(self.all_data)
            
        
    def __getitem__(self, index):
        #current_data = self.data[index]
        #a_abs_path, v_abs_path, seq_len, masking_list = current_data
        #masking_list = ast.literal_eval(masking_list)
        #audio_data, video_data = self.a_data[index], self.v_data[index]
        current_data = self.all_data[index]
        audio_data, video_data, seq_len, masking_list = current_data
        masking_list = ast.literal_eval(masking_list)
        #norm openface
        #video_data = video_data/5 - 0.5
        x_a_masked, x_v_masked = torch.zeros(self.seq_len, self.d_a), torch.zeros(self.seq_len, self.d_v)
        x_a_target, x_v_target = torch.zeros(self.seq_len, self.d_a), torch.zeros(self.seq_len, self.d_v)
        
        #dynamic masking technique
        new_masking_list = [0] * self.seq_len
        E_nmask = int(np.ceil(seq_len*self.masking_prop))
        leading_masks = np.random.randint(0, seq_len, (E_nmask, ))
        random_frame = False
        for i in range(E_nmask):
            current_leading_mask_position = leading_masks[i]
            for j in range(self.mask_consecutive):
                if(current_leading_mask_position+j < self.seq_len):
                    p = random.random()
                    if(p < 0.8):
                        new_masking_list[current_leading_mask_position+j] = 1  #normal masking
                    elif(p < 0.9):
                        new_masking_list[current_leading_mask_position+j] = 2 #replace random frame
                        random_frame = True
                    else:
                        new_masking_list[current_leading_mask_position+j] = 3 #keep unchanged
        if(random_frame):
            file_num = random.randint(0, self.data_length-1)
            random_audio_data, random_video_data, seq_len_r, masking_list_r = self.all_data[file_num]
            frame_num = random.randint(0, min(random_audio_data.shape[0], random_video_data.shape[0])-1)
        for i in range(min(audio_data.shape[0], self.seq_len)):
            if(new_masking_list[i] == 0 or new_masking_list[i] == 3):
                x_a_masked[i] = torch.FloatTensor(audio_data[i])
            elif(new_masking_list[i] == 1):
                x_a_masked[i] = torch.zeros(self.ds_a)
            elif(new_masking_list[i] == 2):
                x_a_masked[i] = torch.FloatTensor(random_audio_data[frame_num]) #replace with random frames
            x_a_target[i] = torch.FloatTensor(audio_data[i])
            
        for i in range(min(video_data.shape[0], self.seq_len)):
            if(new_masking_list[i] == 0 or new_masking_list[i] == 3):
                x_v_masked[i] = torch.FloatTensor(video_data[i])
            elif(new_masking_list[i] == 1):
                x_v_masked[i] = torch.zeros(self.d_v)
            elif(new_masking_list[i] == 2):
                x_v_masked[i] = torch.FloatTensor(random_video_data[frame_num]) #replace with random frames
            x_v_target[i] = torch.FloatTensor(video_data[i])        
        
        bool_masking = [True if x > 0 else False for x in new_masking_list]
        return {'input_audio': x_a_masked,
                'target_audio': x_a_target,
                'input_video': x_v_masked,
                'target_video': x_v_target,
                'mask': torch.BoolTensor(bool_masking)
                }        
    
    def __len__(self):
        return len(self.all_data)    

if __name__ == '__main__':
    #testing discrete dataset class
    #os.chdir('/home/mtran/transformer/code/')
    testing_mode = "mm"
    if(testing_mode == "discrete"):
        print('testing discrete dataset...')
        data_path, vocab_size, seq_len, p_mask, consecutive = "../data/dev_av_5k.txt", 5000, 150, 0.15, 10
        dataset = DiscreteAVDataset(data_path, vocab_size, seq_len, p_mask, consecutive)
        kwargs = {'num_workers':4, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':8}
        data_loader = torch.utils.data.DataLoader(dataset, **kwargs)

        for _, batch_data in enumerate(data_loader):
            print(batch_data['input_audio'].size(), batch_data['input_video'].size(), batch_data['target_audio'].size(), batch_data['target_video'].size())
    elif(testing_mode == 'continuous'):
        print("testing continuous dataset ...")
        datapath, seq_len, p_mask, consecutive = "../../openface_toydata/", 150, 0.15, 4
        dataset = ContinuousDataset(datapath, seq_len, p_mask, consecutive)
        kwargs = {'num_workers':4, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':32}
        data_loader = torch.utils.data.DataLoader(dataset, **kwargs)

        for _, batch_data in enumerate(data_loader):
            print(batch_data['input'].size(), batch_data['target'].size())
    else:
        print("testing multimodal dataset ...")
        dataset = DiscreteAVDatasetNew("../data/dev.txt", "../data/dev.txt", 159)
        kwargs = {'num_workers':4, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':32}
        data_loader = torch.utils.data.DataLoader(dataset, **kwargs)

        for _, batch_data in enumerate(data_loader):
            print(batch_data['input_audio'].size(), batch_data['target_audio'].size())        

