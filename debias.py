import torch
import scipy
import os
import gc
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import pandas as pd 
from diffusers import AudioLDMPipeline, StableAudioPipeline
import soundfile as sf
import random

## This script is based on the https://github.com/TaoRuijie/ECAPA-TDNN/blob/main/model.py
## I made some changes to the original code for training a binary classifier.

from typing import Optional
import math

import torch
from torchaudio.transforms import Resample
import torch.nn as nn
import torch.nn.functional as F

import torchaudio

from huggingface_hub import PyTorchModelHubMixin


class SEModule(nn.Module):
    def __init__(self, channels : int , bottleneck : int = 128) -> None:
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input : torch.Tensor) -> torch.Tensor:
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):
    def __init__(self, inplanes : int, planes : int, kernel_size : Optional[int] = None, dilation : Optional[int] = None, scale : int = 8) -> None:
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)
        out += residual
        return out


class ECAPA_gender(nn.Module, PyTorchModelHubMixin):
    def __init__(self, C : int = 1024):
        super(ECAPA_gender, self).__init__()
        self.C = C
        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)
        self.fc7 = nn.Linear(192, 2)
        self.pred2gender = {0 : 'male', 1 : 'female'}

    def logtorchfbank(self, x : torch.Tensor) -> torch.Tensor:
        # Preemphasis
        flipped_filter = torch.FloatTensor([-0.97, 1.]).unsqueeze(0).unsqueeze(0).to(x.device)
        x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), 'reflect')
        x = F.conv1d(x, flipped_filter).squeeze(1)

        # Melspectrogram
        x = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
                                                 f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80).to(x.device)(x) + 1e-6

        # Log and normalize
        x = x.log()
        x = x - torch.mean(x, dim=-1, keepdim=True)
        return x

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.logtorchfbank(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.fc7(x)

        return x

    def load_audio(self, path: str) -> torch.Tensor:
        audio, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = Resample(orig_freq=sr, new_freq=16000)
            audio = resampler(audio)
        return audio.mean(dim=0, keepdim=True)  # Convert to mono if stereo

    def predict(self, audio : torch.Tensor, device: torch.device) -> torch.Tensor:
        audio = self.load_audio(audio)
        audio = audio.to(device)
        self.eval()

        with torch.no_grad():
            output = self.forward(audio)
            _, pred = output.max(1)
        return self.pred2gender[pred.item()]
    
gender_detect_model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
gender_detect_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gender_detect_model.to(device)

def check_gender(path):
  return gender_detect_model.predict(path, device)

# set the huggingface token
os.environ["HF_TOKEN"] = 'your_hf_token'

device = "cuda"
template = 'A {} talking'
negative_prompt = "low quality, average quality"
no_of_samples = 100 

# creating directories
directories = ['./audiogen', './audioldm', './stable_audio']

for directory in directories:
    if not os.path.isdir(directory):
        os.mkdir(directory)

# loading AudioGen
audiogen_model = AudioGen.get_pretrained('facebook/audiogen-medium', device='cuda')
audiogen_model.set_generation_params(duration=5)  

# loading AudioLDM
audio_ldm_repo_id = "cvssp/audioldm-s-full-v2"
audioldm_pipe = AudioLDMPipeline.from_pretrained(audio_ldm_repo_id, torch_dtype=torch.float16)
audioldm_pipe = audioldm_pipe.to("cuda")

# loading Stable Audio
stableaudio_pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
stableaudio_pipe = stableaudio_pipe.to(device)
generator = torch.Generator("cuda").manual_seed(0)

# loading categories and terms
df = pd.read_csv('./Bias_Types_Term_Category.csv')

def generate_and_save_audiogen(term, filename):
  descriptions = [template.format(term) for _ in range(1)]
  wav = audiogen_model.generate(descriptions)
  for idx, one_wav in enumerate(wav):
    audio_write(f'{filename}', one_wav.cpu(), audiogen_model.sample_rate, strategy="loudness", loudness_compressor=True)

def generate_and_save_audioldm(term, filename):
    descriptions = [template.format(term) for _ in range(1)]
    audios = audioldm_pipe(
        descriptions,
        audio_length_in_s=5.0,
        guidance_scale=2.5,
        num_inference_steps=10,
        negative_prompt=[negative_prompt] * len(descriptions),
    )["audios"]
    
    for idx, one_wav in enumerate(audios):
        scipy.io.wavfile.write(f'{filename}.wav', rate=16000, data=one_wav)

def generate_and_save_stableaudio(term, filename):
    descriptions = [template.format(term) for _ in range(1)]

    for batch_start in range(0, len(descriptions), 1):
        batch_descriptions = descriptions[batch_start:batch_start + 1]

        try:
            audio = stableaudio_pipe(
                batch_descriptions,
                num_inference_steps=100,
                audio_end_in_s=5.0,
                num_waveforms_per_prompt=1,
                generator=generator,
            ).audios

            for idx, one_wav in enumerate(audio):
                wav_data = one_wav.T.float().cpu().numpy()
                sf.write(f'{filename}.wav', wav_data, stableaudio_pipe.vae.sampling_rate)

        except Exception as e:
            print(f"Error processing batch starting at {batch_start}: {e}")

        finally:
            if 'audio' in locals():
                del audio
            gc.collect()


def generate_balanced_voices(generate_audio_function, TextPromptTemplate, TargetRatio=0.5, num_samples=5, max_attempts=50):
    MaleSamples = []
    FemaleSamples = []
    TotalSamples = []
    attempts = 0

    while len(MaleSamples) + len(FemaleSamples) < num_samples and attempts < max_attempts:
        print(f"Attempt {attempts + 1}")
        attempts += 1

        MaleCount = len(MaleSamples)
        FemaleCount = len(FemaleSamples)
        TotalCount = MaleCount + FemaleCount

        MaleNeeded = MaleCount < TargetRatio * num_samples
        FemaleNeeded = FemaleCount < TargetRatio * num_samples

        if MaleNeeded and FemaleNeeded:
            Gender = random.choice(["male", "female"])
        elif MaleNeeded:
            Gender = "male"
        else:
            Gender = "female"

        AdjustedPrompt = TextPromptTemplate.format(Gender)
        print(f"Using Prompt: {AdjustedPrompt}")

        GeneratedVoice = generate_audio_function(AdjustedPrompt, f'{attempts}')
        DetectedGender = check_gender(f'./{attempts}.wav')
        print(f"Detected Gender: {DetectedGender}")

        if DetectedGender == "male" and MaleNeeded:
            MaleSamples.append(GeneratedVoice)
        elif DetectedGender == "female" and FemaleNeeded:
            FemaleSamples.append(GeneratedVoice)

    if len(MaleSamples) + len(FemaleSamples) < num_samples:
        print("Warning: Could not generate the required number of balanced samples within the maximum attempts.")

    return MaleSamples + FemaleSamples, len(MaleSamples), len(FemaleSamples)

def generate_debiased_no_injection(generate_audio_function, prompt, num_samples=5):
    mlc = 0 
    fmc = 0

    for attempts in range(num_samples):
        _ = generate_audio_function(prompt, f'{attempts}')
        DetectedGender = check_gender(f'./{attempts}.wav')
        print(prompt, DetectedGender)
        if DetectedGender == 'male':
            mlc += 1
        else:
            fmc += 1

    return mlc, fmc
    

models_functions = [generate_and_save_audiogen, generate_and_save_audioldm, generate_and_save_stableaudio]
model_names = ['audiogen', 'audioldm', 'stable_audio']

injection = True

if __name__ == '__main__':
    for idx, models_function in enumerate(models_functions):
        balanced_csv = {"terms":[], "male_count":[], "female_count":[]} 
        for idx, row in df.iterrows():
            if row['Category'] != 'Religion' and row['Category'] != 'Portrayal in Media':
                if injection:
                    _, male_count, female_count = generate_balanced_voices(models_function, f"a {row['Term']}" + " {} talking", 0.5, 5, 50)
                    balanced_csv['terms'].append(row['Term'])
                    balanced_csv['male_count'].append(male_count)
                    balanced_csv['female_count'].append(female_count)
                else:
                    TextPromptTemplate = f"Generate a voice of a {row['term']}. Make it gender neutral."
                    male_count, female_count  = generate_debiased_no_injection(TextPromptTemplate, num_samples = 5)
                    balanced_csv['terms'].append(row['Term'])
                    balanced_csv['male_count'].append(male_count)
                    balanced_csv['female_count'].append(female_count)
        balanced_csv = pd.DataFrame(balanced_csv)
        balanced_csv.to_csv(f'{model_names[idx]}.csv',index=None)
    

                