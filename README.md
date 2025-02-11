# Detecting Gender Bias in Audio

## Installation Steps: 

1. Install ffmpeg
```
apt-get install -y ffmpeg
```

2. Run the following command to install spacy 
```
python -m spacy download en_core_web_sm
```

2. Install the following python packages using pip 
```
pip install git+https://github.com/facebookresearch/audiocraft.git
pip install ffmpeg-python
pip install diffusers torchsde spacy 
```

3. Set the huggingface token in the `generating_audios.py` file
```
os.environ["HF_TOKEN"] = 'your_hf_token'
```

## Running 

1. Run the `generating_audios.py` file to generate audios.
2. Run the `count_and_calculate_baas.py` file to generate the CSVs for the final scores
3. Run the `debias.py` file to generated debiased samples.

