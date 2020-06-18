import os
import time
import torch
import json
from datetime import datetime
import string
from glob import glob
import numpy as np
import re
from pathlib import Path
import sys
import yaml
import random


from TTS_lib.utils.synthesis import synthesis
from TTS_lib.utils.generic_utils import setup_model
from TTS_lib.utils.io import load_config, load_checkpoint
from TTS_lib.utils.text.symbols import make_symbols, symbols, phonemes
from TTS_lib.utils.audio import AudioProcessor

from TTS_lib.vocoder.utils.generic_utils import setup_generator 


def tts(model,
        vocoder_model,
        C,
        VC,
        text,
        ap,
        ap_vocoder,
        use_cuda,
        batched_vocoder,
        speaker_id=None,
        style_input=None,
        figures=False):
    use_vocoder_model = vocoder_model is not None

    waveform, alignment, _, postnet_output, stop_tokens, _ = synthesis(
        model, text, C, use_cuda, ap, speaker_id, style_input=style_input,
        truncated=False, enable_eos_bos_chars=C.enable_eos_bos_chars,
        use_griffin_lim=(not use_vocoder_model), do_trim_silence=True)


    if C.model == "Tacotron" and use_vocoder_model:
        postnet_output = ap.out_linear_to_mel(postnet_output.T).T
    # correct if there is a scale difference b/w two models
    
    if use_vocoder_model:
        vocoder_input = torch.FloatTensor(postnet_output.T).unsqueeze(0)
        waveform = vocoder_model.inference(vocoder_input)
        if use_cuda:
            waveform = waveform.cpu()
        waveform = waveform.detach().numpy()
        waveform = waveform.flatten()
        

    # if use_vocoder_model:
    #     postnet_output = ap._denormalize(postnet_output)
    #     postnet_output = ap_vocoder._normalize(postnet_output)
    #     vocoder_input = torch.FloatTensor(postnet_output.T).unsqueeze(0)
    #     waveform = vocoder_model.generate(
    #         vocoder_input.cuda() if use_cuda else vocoder_input,
    #         batched=batched_vocoder,
    #         target=8000,
    #         overlap=400)

    return alignment, postnet_output, stop_tokens, waveform


def load_melgan(lib_path, model_file, model_config, use_cuda):
    sys.path.append(lib_path) # set this if ParallelWaveGAN is not installed globally
    #pylint: disable=import-outside-toplevel
    C = load_config(model_config)
    model_gen = setup_generator(C)
    checkpoint = torch.load(model_file, map_location='cpu')
    model_gen.load_state_dict(checkpoint['model'])
    ap_vocoder = AudioProcessor(**C.audio)

    return model_gen.eval(), ap_vocoder

def split_into_sentences(text):
    text = text.replace('.', '.<stop>')
    text = text.replace('!', '!<stop>')
    text = text.replace('?', '?<stop>')
    sentences = text.split("<stop>")
    sentences = list(filter(None, [s.strip() for s in sentences]))  # remove empty sentences
    return sentences


def main(**kwargs):
    global symbols, phonemes
    start_time = time.time()

    # read passed variables from gui
    text = kwargs['text']                           # text to generate speech from
    use_cuda = kwargs['use_cuda']                   # if gpu exists default is true
    project = kwargs['project']                     # path to project folder
    vocoder_type = kwargs['vocoder']                # vocoder type, default is GL
    vocoder_model_file = ""                         # path to vocoder model file
    vocoder_config = ""                             # path to vocoder config file
    use_gst = kwargs['use_gst']                     # use style_wave for prosody
    style_dict = kwargs['style_int']                 # use style_wave for prosody
    use_style_dict = style_dict is not None
    speakers_json = kwargs['speaker_config']        # has to be the speakers file
    speaker_name = kwargs['speaker_name']           # name of the selected speaker
    sentence_file = kwargs['sentence_file']         # path to file if generate from file
    
    
    batched_vocoder = True
    
    # create output directory if it doesn't exist
    os.makedirs(str(Path(project + '/output/' + speaker_name)), exist_ok=True)
    
    # load the config
    config_path = Path(project + "/config.json")
    C = load_config(config_path)
    #C.forward_attn_mask = True

    if use_gst:
        if use_style_dict:
            style_wav = style_dict
            print(style_wav)
        else:
            if speaker_name != 'Default':
                #prosody_waves = glob(str(Path(project + '/prosody_style/*.wav')))
                prosody_waves = glob(str(Path(C.datasets[0]['path']+speaker_name+'/*/*.wav')))
            else:
                            #prosody_waves = glob(str(Path(project + '/prosody_style/*.wav')))
                prosody_waves = glob(str(Path(C.datasets[0]['path']+'/*/*.wav')))
            #style_wav = None #'svm_xardas_handsoff_g3.wav'
            style_wav_id = random.randrange(0, len(prosody_waves), 1)
            style_wav = prosody_waves[style_wav_id]
    else:
        style_wav = None

    # load the audio processor
    ap = AudioProcessor(**C.audio)

    # if the vocabulary was passed, replace the default
    if 'characters' in C.keys():
        symbols, phonemes = make_symbols(**C.characters)


    # load speakers
    speaker_id = None
    if speakers_json != '':
        speakers = json.load(open(speakers_json, 'r'))
        num_speakers = len(speakers)
        #get the speaker id for selected speaker
        speaker_id = [id for speaker, id in speakers.items() if speaker_name in speaker][0]
    else:
        num_speakers = 0

    # find the tts model file in project folder
    try:
        tts_model_file = glob(str(Path(project + '/*.pth.tar')))
        if not tts_model_file:
            raise FileNotFoundError('[!] TTS Model not found in path: "{}"'.format(project))
        model_path = tts_model_file[0]
    except FileNotFoundError:
        raise

    # load the model
    num_chars = len(phonemes) if C.use_phonemes else len(symbols)
    model = setup_model(num_chars, num_speakers, C)

    # if gpu is not available use cpu
    model, _ = load_checkpoint(model, model_path, use_cuda=use_cuda)
    # if not use_cuda:
    #     cp = torch.load(model_path, map_location=torch.device('cpu'))
    # else:
    #     cp = torch.load(model_path)

    # model.load_state_dict(cp['model'])
    #print(model)
    model.decoder.max_decoder_steps = 2000
    
    model.eval()
    # if use_cuda:
    #     model.cuda()
    # model.decoder.set_r(cp['r'])

    # load vocoder
    if vocoder_type is 'GAN':
        model_file = glob(str(Path("/media/alexander/LinuxFS/Documents/PycharmProjects/GothicTTS/TTS_lib/vocoder/Trainings/multiband-melgan-rwd-Juni-15-2020_02+07-9d7cb1e/*.pth.tar")))
        if model_file:
            print(model_file[0])
            vocoder, ap_vocoder = load_melgan(str(Path('TTS_lib')), 
            str(model_file[0]), 
            str("/media/alexander/LinuxFS/Documents/PycharmProjects/GothicTTS/TTS_lib/vocoder/Trainings/multiband-melgan-rwd-Juni-15-2020_02+07-9d7cb1e/config.json"), 
            use_cuda)
        else:
            vocoder_type = 'GriffinLim'
    elif vocoder_type is 'WaveRNN':
        model_file = glob(str(Path(project + '/*.pkl')))
        if model_file:
            vocoder, ap_vocoder = load_melgan(str(Path('TTS_lib')), str(model_file[0]), str(Path(project + '/config.yml')), use_cuda)
        else:
            vocoder_type = 'GriffinLim'
    
    if vocoder_type is 'GriffinLim':
        vocoder, ap_vocoder = None, None

    print(" > Vocoder: {}".format(vocoder_type))

    # if files with sentences was passed -> read them
    if sentence_file != '':
        with open(sentence_file, "r", encoding='utf8') as f:
            list_of_sentences = [s.strip() for s in f.readlines()]
    else:
        list_of_sentences = [text.strip()]

    print(' > Using style wav: {}\n'.format(style_wav))

    # iterate over every passed sentence and synthesize
    for _, tts_sentence in enumerate(list_of_sentences):
        print(" > Text: {}".format(tts_sentence))
        wav_list = []

        # build filename
        current_time = datetime.now().strftime("%m%d%Y_%H%M%S")
        out_path = Path(project + '/output/' + speaker_name)
        speaker_name_file = speaker_name.split(' ')
        speaker_name_file = '_'.join(speaker_name_file)
        file_name = ''

        # if multiple sentences in one line -> split them
        tts_sentence = split_into_sentences(tts_sentence)

        # if sentence was split in sub-sentences -> iterate over them
        for sentence in tts_sentence:
            # synthesize voice
            _, _, _, wav = tts(model,
                               vocoder,
                               C,
                               None,
                               sentence,
                               ap,
                               ap_vocoder,
                               use_cuda,
                               batched_vocoder,
                               speaker_id=speaker_id,
                               style_input=style_wav,
                               figures=False)

            # join sub-sentences back together and add a filler between them
            wav_list += list(wav)
            wav_list += [0] * 10000
            file_name += sentence.replace(" ", "_")
        wav = np.array(wav_list)

        # finalize filename
        file_name = '_'.join([str(current_time), str(speaker_id), speaker_name_file, file_name[:30]])
        file_name = file_name.translate(
            str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
        out_path = os.path.join(out_path, file_name)

        # save generated wav to disk
        ap.save_wav(wav, out_path)
        end_time = time.time()
        print(" > Run-time: {}".format(end_time - start_time))
        print(" > Saving output to {}\n".format(out_path))
