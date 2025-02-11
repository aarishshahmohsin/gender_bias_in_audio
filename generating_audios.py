import torch
import scipy
import os
import gc
from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write
import pandas as pd 
from diffusers import AudioLDMPipeline, StableAudioPipeline
import soundfile as sf
import spacy 

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

# loading the library for noun/adjective detection
nlp = spacy.load("en_core_web_sm")

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

def generate_sentence(word):
    doc = nlp(word)
    for token in doc:
        if token.pos_ == "NOUN":
            return f"A {word} taking"
        else:
            return f"A {word} person talking"

def generate_and_save_audiogen(term, start_idx=0, number=100):
  descriptions = [generate_sentence(term) for _ in range(number)]
  wav = audiogen_model.generate(descriptions)
  for idx, one_wav in enumerate(wav):
    audio_write(f'./audiogen/{term}_{idx+start_idx}', one_wav.cpu(), audiogen_model.sample_rate, strategy="loudness", loudness_compressor=True)

def generate_and_save_audioldm(term, start_idx=0, number=100):
    descriptions = [generate_sentence(term) for _ in range(number)]
    audios = audioldm_pipe(
        descriptions,
        audio_length_in_s=5.0,
        guidance_scale=2.5,
        num_inference_steps=10,
        negative_prompt=[negative_prompt] * len(descriptions),
    )["audios"]
    
    for idx, one_wav in enumerate(audios):
        scipy.io.wavfile.write(f'./audio_ldm/{term}_{idx+start_idx}', rate=16000, data=one_wav)

def generate_and_save_stableaudio(term, start_idx=0, number=100, batch_size=10):
    descriptions = [generate_sentence(term) for _ in range(number)]

    for batch_start in range(0, len(descriptions), batch_size):
        batch_descriptions = descriptions[batch_start:batch_start + batch_size]

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
                sf.write(f'./stable_audio/{term}_{batch_start + idx + start_idx}.wav', wav_data, stableaudio_pipe.vae.sampling_rate)

        except Exception as e:
            print(f"Error processing batch starting at {batch_start}: {e}")

        finally:
            if 'audio' in locals():
                del audio
            gc.collect()


if __name__ == '__main__':
    # generating 100 samples for each model
    for idx, row in df.iterrows():
        if row['Category'] != 'Religion' and row['Category'] != 'Portrayal in Media':
            generate_and_save_audiogen(row['Term'], 0, no_of_samples)
            generate_and_save_audioldm(row['Term'], 0, no_of_samples)
            generate_and_save_stableaudio(row['Term'], 0, no_of_samples, 4)