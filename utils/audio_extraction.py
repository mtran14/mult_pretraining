import librosa
import numpy as np
from vectorhub.encoders.audio.tfhub import Trill2Vec
import os
import pandas as pd
import random

def split_given_size(a, size):
    return np.split(a, np.arange(size,len(a),size))

def extract_from_wav(file, trill_model):
    sr = 16000
    y, current_sr=librosa.load(file, dtype='float32')
    frame_rate = 5 #fps
    frame_size = int(sr/frame_rate)

    y = np.array(librosa.resample(y.T, current_sr, sr))
    frames = split_given_size(y, frame_size)

    output_trill = []
    for frame in frames:
        trill_emb = trill_model.encode(frame)
        output_trill.append(trill_emb)
    output_trill = np.array(output_trill)
    return output_trill

if __name__ == '__main__':
    file_name = '../../00148.wav'
    trill_model = Trill2Vec()
    trill_features = extract_from_wav(file_name, trill_model)
    print(trill_features.shape)


# files = pd.read_csv("/shares/perception-working/minh/file_paths_fm.csv", header=None).values[:, 0]
# random.shuffle(files)
# output_path = "/shares/perception-temp/voxceleb2/trill/train/"

# trill_model = Trill2Vec()
# sr = 16000

# for file in files:
#     file_path_split = file.split("/")
#     id1, id2, fname = file_path_split[-3], file_path_split[-2], file_path_split[-1]
#     output_file_name = id1 + '_' + id2 + '_' + fname.split('.')[0] + '.csv'   
#     out_trill = os.path.join(output_path, output_file_name)
#     if(os.path.isfile(out_trill)):
#         continue
#     try:
#         y, current_sr=librosa.load(file, dtype='float32')
#     except:
#         continue
#     frame_rate = 5 #fps
#     frame_size = int(sr/frame_rate)

#     y = np.array(librosa.resample(y.T, current_sr, sr))
#     frames = split_given_size(y, frame_size)

#     output_trill = []
#     for frame in frames:
#         trill_emb = trill_model.encode(frame)
#         output_trill.append(trill_emb)
#     output_trill = np.array(output_trill)

    
#     pd.DataFrame(output_trill).to_csv(out_trill, header=None, index=False)
#     print(out_trill)


