import PySimpleGUI as sg
import threading
from TTS_lib import synthesize
from torch.cuda import is_available
import json
from glob import glob
from pathlib import Path
import os, subprocess, platform
import webbrowser
import random

# Global Variables
status = False
_platform = platform.system()

def synthesize_tts(text, 
                   use_cuda, 
                   use_gst, 
                   style_int, 
                   project, 
                   speaker_config, 
                   speaker_name, 
                   vocoder_type, 
                   sentence_file):
    
    global status
    synthesize.main(text=text,
                    use_cuda=use_cuda,
                    use_gst=use_gst,
                    style_int=style_int,
                    project=project,
                    speaker_config=speaker_config,
                    speaker_name=speaker_name,
                    vocoder=vocoder_type,
                    sentence_file=sentence_file)
    status = True


def open_output_folder(speaker_path):
    try:
        if _platform == 'Darwin':       # macOS
            subprocess.call(('open', str(speaker_path)))
        elif _platform == 'Windows':    # Windows
            os.startfile(str(speaker_path))
        else:   # linux variants                      
            #webbrowser.open(str(speaker_path))
            subprocess.Popen(['xdg-open', str(speaker_path)])
    except FileNotFoundError as er:
        print(er)


def main_gui():

    # paths
    root_path = os.path.dirname(os.path.abspath(__file__))
    print(root_path)
    if root_path:
        os.chdir(root_path)
    path_loading_gif = str(Path(root_path, "media/loading.gif"))
    path_icon = str(Path(root_path, "media/g"))
    project_path = str(Path(root_path, "TTS_lib/trained_models/*/"))

    # Variables
    thread = None
    sentence_file = ''
    text_memory = ''
    cuda_available = is_available()
    gst_dict = {}
    global status

    # set icon for distro
    if _platform == 'Windows':
        icon = path_icon + '.ico'
    else:
        icon = path_icon + '.png'

    # set the theme
    sg.theme('DarkTeal6')

    # init default settings
    projectFolders = sg.DropDown(["No Projects"], key='dbProject', pad=[5, 5])
    loadingAnimation = sg.Image(path_loading_gif, visible=False, key='loadingAnim', background_color='white')
    textInput = sg.Multiline('Im Minental versammelt sich eine Armee des Bösen unter der Führung von Drachen! Wir müssen sie aufhalten, so lange wir noch können.',
     size=(60, 6), pad=[5, 5], border_width=1, font=('Arial', 12), text_color='#000000', background_color='#FFFFFF', key='textInput')
    use_cuda = sg.Checkbox('Use CUDA?', default=cuda_available, key='use_cuda', visible=cuda_available)
    cuda_color =  'green' if cuda_available else 'red'
    cuda_text = '(CUDA Support Enabled)' if cuda_available else '(CUDA Support Disabled)'

    # get project folders
    project_folders = glob(project_path)
    if project_folders:
        projectFolders = sg.DropDown(project_folders, project_folders[0], key='dbProject', enable_events=True, pad=[5, 5], size=[90, 5])
        speakers_path = Path(project_folders[0] + "/speakers.json")
        if speakers_path.is_file():
            with open(speakers_path, 'r') as json_file:
                speaker_data = json.load(json_file)
            speaker_lst = [speaker for speaker, _ in speaker_data.items()]
            speakerDropDown = sg.DropDown(speaker_lst, speaker_lst[0], key='dbSpeaker', pad=[5, 5])
        else:
            speakerDropDown = sg.DropDown(["Default"], "Default", visible=False, key='dbSpeaker', pad=[5, 5])

    # All the stuff inside your window.
    layout = [
        [sg.Text('Project Settings:', font=('Arial', 12, 'bold'))],
        [projectFolders],
        [sg.Text(cuda_text, text_color=cuda_color, font=('Arial', 10,'bold')), use_cuda],
        
        [sg.Text('_' * 90)],
        
        [sg.Text('Vocoder Settings:', font=('Arial', 12, 'bold'))],
        [sg.Radio('GriffinLim', 0, True, font=('Arial', 11), key='radioGL'),
         sg.Radio('WaveRNN', 0, font=('Arial', 11), key='radioWR'),
         sg.Radio('PwGAN / MelGAN', 0, font=('Arial', 11), key='radioGAN')],
        
        [sg.Text('Speaker:', pad=[5, 5], visible=False), speakerDropDown],
        [sg.Checkbox('Use GST?', True, visible=False, enable_events=True, pad=[5, 5], key='cbUseGST')],
        
        [sg.Text('_' * 90)],
        
        [sg.Text('Emotion Settings: ', font=('Arial', 12, 'bold'))],
        [sg.Text('Emotion: ', font=('Arial', 11)), sg.DropDown(['Normal', 'Angry', 'Dominant', 'Calm', ], 'Normal', key='dbEmotion')],
        [sg.Text('Speed of Speech: ', pad=[5, 5], size=(20, 0), font=('Arial', 11), key='token0'), 
            sg.Slider(range=(-50,50), default_value=0, size=(30,12), orientation='horizontal', font=('Arial', 10, 'bold'), key='speedSlider')],
        
        [sg.Text('Dominance of Speech: ', pad=[5, 5], size=(20, 0), font=('Arial', 11), key='token1and2'), 
            sg.Slider(range=(-50,50), default_value=0, size=(30, 12), orientation='horizontal', font=('Arial', 10, 'bold'), key='emotionSlider')],
        
        [sg.Text('Tone of Speech: ',  pad=[5, 5], size=(20, 0), font=('Arial', 11), key='token5'), 
            sg.Slider(range=(-50,50), default_value=0, size=(30, 12), orientation='horizontal', font=('Arial', 10, 'bold'), key='toneSlider')],

        # [sg.Text('1', pad=[5, 5], size=[1, 1], key='token1'), 
        #     sg.Slider(range=(-4,4), default_value=0, size=(30,15), orientation='horizontal', font=('Helvetica', 8), pad=[5, 5], key='slider1')],

        # [sg.Text('2', pad=[5, 5], size=[1, 1], key='token2'), 
        #     sg.Slider(range=(-4,4), default_value=0, size=(30,15), orientation='horizontal', font=('Helvetica', 8), pad=[5, 5], key='slider2')],
        
        # [sg.Text('3', pad=[5, 5], size=[1, 1], key='token3'), 
        #     sg.Slider(range=(-4,4), default_value=0, size=(30,15), orientation='horizontal', font=('Helvetica', 8), pad=[5, 5], key='slider3')],
                
        # [sg.Text('4', pad=[5, 5], size=[1, 1], key='token4'), 
        #     sg.Slider(range=(-4,4), default_value=0, size=(30,15), orientation='horizontal', font=('Helvetica', 8), pad=[5, 5], key='slider4')],
        
        # [sg.Text('5', pad=[5, 5], size=[1, 1], key='token5'), 
        #     sg.Slider(range=(-4,4), default_value=0, size=(30,15), orientation='horizontal', font=('Helvetica', 8), pad=[5, 5], key='slider5')],
        
        # [sg.Text('6', pad=[5, 5], size=[1, 1], key='token6'), 
        #     sg.Slider(range=(-4,4), default_value=0, size=(30,15), orientation='horizontal', font=('Helvetica', 8), pad=[5, 5], key='slider6')],
        
        # [sg.Text('7', pad=[5, 5], size=[1, 1], key='token7'), 
        #     sg.Slider(range=(-4,4), default_value=0, size=(30,15), orientation='horizontal', font=('Helvetica', 8), pad=[5, 5], key='slider7')],
                
        # [sg.Text('8', pad=[5, 5], size=[1, 1], key='token8'), 
        #     sg.Slider(range=(-4,4), default_value=0, size=(30,15), orientation='horizontal', font=('Helvetica', 8), pad=[5, 5], key='slider8')],
        
        # [sg.Text('9', pad=[5, 5], size=[1, 1], key='token9'), 
        #     sg.Slider(range=(-4,4), default_value=0, size=(30,15), orientation='horizontal', font=('Helvetica', 8), pad=[5, 5], key='slider9')],
          
    #    [sg.Text('Prosody of Speech:', pad=[5, 5]), sg.DropDown(['Normal', 'Happy', 'Angry'], 'Normal', key='dbProsody', pad=[5, 5])],
        [sg.Text('_' * 90)],
        
        [sg.Checkbox('Use file for speech generation', False, font=('Arial', 12, 'bold'), enable_events=True, pad=[5, 5], key='cbLoadFile')],
        [sg.Text('', size=[75, None], font=('Arial', 10), key='lblTextFile')],
        [sg.Text('Enter\nText:', font=('Arial', 12, 'bold'), pad=[5, 5]), textInput, sg.Button('Output\nFolder', size=[5, 3], enable_events=True, key='btnOpenOutput')],
        [sg.Button('Generate', key='btnGenerate'), sg.Button('Exit'), loadingAnimation]
    ]

    # Create the Window
    window = sg.Window('GothicTTS (Generate Speech From Text)', icon=icon, layout=layout, finalize=True)

    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read(timeout=100)

        if event in (None, 'Exit'):  # if user closes window or clicks cancel
            break

        if event in 'dbProject':
            if Path(values['dbProject'] + "/speakers.json").is_file():
                with open(Path(values['dbProject'] + "/speakers.json"), 'r') as json_file:
                    speaker_data = json.load(json_file)
                speaker_lst = [speaker for speaker, _ in speaker_data.items()]
                max_length_name = len(max(speaker_lst, key=len))
                window['dbSpeaker'].update(values=speaker_lst)
                window['dbSpeaker'].set_size(size=(max_length_name, None))
            else:
                window['dbSpeaker'].update(values=["Default"])
                window['dbSpeaker'].set_size(size=(len("Default"), None))
                
        if event in 'dbEmotion':
            print('was here')

        if event in 'cbLoadFile':
            if values['cbLoadFile']:
                path_to_textfile = sg.PopupGetFile('Please select a file or enter the file name', default_path=root_path, initial_folder=root_path,
                                       icon='g.ico', no_window=True, keep_on_top=True, file_types=(('Text file', '.txt'),))
                if path_to_textfile:
                    window['lblTextFile'].update(path_to_textfile)
                    sentence_file = path_to_textfile
                    text_memory = window['textInput'].get()
                    window['textInput'].update(disabled=True, background_color='#a7a5a5', value=f'Using textfile: {path_to_textfile}')
                else:
                    window['cbLoadFile'].update(value=False)
            else:
                window['lblTextFile'].update(value='')
                sentence_file = ''
                window['textInput'].update(disabled=False, background_color='#FFFFFF')
                window['textInput'].update(text_memory)

        if event in 'btnOpenOutput':
            if Path(values['dbProject'] + '/output').is_dir():
                speaker_path = Path(values['dbProject'] + '/output/' + values['dbSpeaker'])
                open_output_folder(speaker_path=speaker_path)
        
            
        if event in 'btnGenerate' and not thread:
            text = values['textInput'].replace('\n', '')
            if text or (sentence_file and values['cbLoadFile']):
                if values['dbSpeaker'] in 'Default':
                    speakers_file = ''
                    speaker_name = "Default"
                else:
                    speakers_file = Path(values['dbProject'] + '/speakers.json')
                    speaker_name = values['dbSpeaker']

                if values['radioGL']:
                    vocoder = 'GriffinLim'
                elif values['radioWR']:
                    vocoder = 'WaveRNN'
                elif values['radioGAN']:
                    vocoder = 'GAN'
                else:
                    vocoder = 'GriffinLim'
                
                # preload style token dict with zeros
                for index, _ in enumerate(range(10)):
                    gst_dict[str(index)] = float(0.0)
                
                
                gst_dict['0'] = round(float(values['speedSlider'] / 100), 3)
                emotion_temp = round(float(values['emotionSlider'] / 100), 3)
                gst_dict['1'] = emotion_temp
                if values['emotionSlider'] > 0:
                    gst_dict['2'] = round(emotion_temp - 0.10 , 3)
                elif values['emotionSlider'] < 0:
                    gst_dict['2'] = round(emotion_temp + 0.10 , 3)
                else:
                    gst_dict['2'] = emotion_temp
                gst_dict['5'] = round(float(values['toneSlider'] / 100), 3)
                
                
                # run speech generation in a new thread
                thread = threading.Thread(target=synthesize_tts,
                                          args=(text,
                                                values['use_cuda'],
                                                True, # use gst
                                                gst_dict,
                                                values['dbProject'],
                                                speakers_file,
                                                speaker_name,
                                                vocoder,
                                                sentence_file), daemon=True)
                thread.start()
                loadingAnimation.Update(filename=path_loading_gif, visible=True)
            else:
                sg.Popup('Type something into the textbox or select a file to generate speech.', title='Missing input!',
                         line_width=65, icon='g.ico')
                textInput.SetFocus()

        if status:
            print('Finished')
            loadingAnimation.Update(filename=path_loading_gif, visible=False)
            if Path(values['dbProject'] + '/output').is_dir():
                speaker_path = Path(values['dbProject'] + '/output/' + values['dbSpeaker'])
                open_output_folder(speaker_path=speaker_path)
            status = False

        if thread:  # If thread is running
            loadingAnimation.UpdateAnimation(source=path_loading_gif, time_between_frames=100)
            thread.join(timeout=0)
            if not thread.is_alive():       # the thread finished
                loadingAnimation.Update(filename=path_loading_gif, visible=False)
                thread = None               # reset variables for next run

    window.close()


if __name__ == '__main__':
    main_gui()
