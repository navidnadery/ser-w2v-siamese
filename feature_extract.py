# python3
# Extract and Save wav2vec 2.0 representation features as pytorch

# In[1]
## Import necessary packages and initialize path and other vars
from glob import glob
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
import os
import torch
import torchaudio

speakers = ['03', '08', '09', '10', '11', '12', '13', '14', '15', '16']

path = input("please enter the path where wav files exist(example: /home/user/Downloads/Berlin/wav/):\n")
assert os.path.exists(path) and len(glob(os.path.join(path, "*.wav"))) > 0, "path is not correct, please enter the path where all wav files exist in"
data_dir = Path(path)
    
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# In[2]:
# Load the model for extracting representations
bundle = torchaudio.pipelines.WAV2VEC2_BASE
model = bundle.get_model().to(device)

## Extract/Save features
# In[3]:
# Remove old saved feature files, if exist
os.system(f"rm {path}/*/*.pt")

# In[4]:
# Extract and save features for Berlin dataset
ses = defaultdict(list)
for f, g in enumerate(speakers):
    if not os.path.exists(data_dir.joinpath(g)):
        os.mkdir(data_dir.joinpath(g))
    ses[f] = list(data_dir.glob(f"{g}*.wav"))
    for wav_file in tqdm(ses[f], disable=True):
        waveform, sample_rate = torchaudio.load(wav_file)
        waveform = waveform.to(device)
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        with torch.inference_mode():
            conv_feats, _ = model.feature_extractor(waveform, waveform.shape[1])
            torch.save(conv_feats.detach().cpu(), os.path.join(wav_file.parent, g, wav_file.stem+'.pt'))