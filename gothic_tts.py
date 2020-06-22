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
import logging
from datetime import date

# Global Variables
status = False
_platform = platform.system()
version = '0.0.2'


def synthesize_tts(text, 
                   use_cuda, 
                   use_gst, 
                   style_input, 
                   project, 
                   speaker_config, 
                   speaker_list, 
                   vocoder_type, 
                   sentence_file):
    
    global status
    for speaker in speaker_list:
        synthesize.main(text=text,
                        use_cuda=use_cuda,
                        use_gst=use_gst,
                        style_input=style_input,
                        project=project,
                        speaker_config=speaker_config,
                        speaker_name=speaker,
                        vocoder=vocoder_type,
                        sentence_file=sentence_file)

    status = True


def open_output_folder(speaker_path):
    """[Try to determin the operating system and use the corresponding function to open a folder]

    Args:
        speaker_path ([string]): [Path to the output folder of the selected speaker]
    """
    try:
        if _platform == 'Darwin':       # macOS
            subprocess.call(('open', str(speaker_path)))  # not tested
        elif _platform == 'Windows':    # Windows
            os.startfile(str(speaker_path))
        else:   # linux variants                      
            subprocess.Popen(['xdg-open', str(speaker_path)])
    except FileNotFoundError as er:
        print(er)
        

def get_emotion_weights(selected_emotion):
    """[This function gets a selected emotion as string from the dropdown gui element.]

    Args:
        selected_emotion ([string]): [Selected emotion from the gui element, used to filter for the needed weights]

    Returns:
        [dictionary]: [Return a selected dictionary with slider-keys and the corresponding values]
    """
    # Emotion weights can be defined here
    EMOTIONS = [
        ["normal",      {'speedSlider':0, 'emotionSlider':  0,  'toneSlider':   0}],
        ["angry",       {'speedSlider':0, 'emotionSlider':  30, 'toneSlider':   -10}],
        ["dominant",    {'speedSlider':0, 'emotionSlider':  30, 'toneSlider':   10}],
        ["calm",        {'speedSlider':0, 'emotionSlider': -20, 'toneSlider':   10}],
        ]
    for emotion in EMOTIONS:
        if selected_emotion in emotion[0]:
            # 
            return emotion[1]


def main_gui():
    
    global status

    # Variables
    current_date = date.today()
    current_date = current_date.strftime("%B %d %Y")
    thread = None
    sentence_file = ''
    text_memory = ''
    speaker_name = None
    cuda_available = is_available()
    gst_dict = {}
    # preload style token dict with zeros
    for index, _ in enumerate(range(10)):
        gst_dict[str(index)] = float(0.0)

    # set icon for distro
    if _platform == 'Windows':
        icon = PATH_GUI_ICON + '.ico'
    else:
        icon = PATH_GUI_ICON + '.png'
        
    # set the theme
    #sg.theme('DarkTeal6')
    sg.theme('CustomTheme') # Custom theme defined at the bottom of the script. This theme can be modified or replaced with default themes, see above.
    
    # init default settings
    radio_keys = {'radioGL': 'GriffinLim', 'radioWR': 'WaveRNN', 'radioMG': 'MelGAN'}
    radio_image_keys = {'radioGL': 'imgradioGL', 'radioWR': 'imgradioWR', 'radioMG': 'imgradioMG'}
    selected_color = ('white', '#273c75')
    active_radio_button = 'radioGL'
    projectFolders = sg.DropDown(["No Projects"], key='dbProject', pad=[5, 5])
    loadingAnimation = sg.Image(PATH_LOADING_GIF, visible=False, key='loadingAnim', background_color='white')
    textInput = sg.Multiline('Im Minental versammelt sich eine Armee des Bösen unter der Führung von Drachen! Wir müssen sie aufhalten, so lange wir noch können.',
     size=(60, 6), pad=[5, 5], border_width=1, font=('Arial', 12), text_color=TEXT_COLOR, background_color=TEXTINPUT_BACKGROUND, key='textInput')
    use_cuda = sg.Checkbox('Use CUDA?', default=cuda_available, key='use_cuda', visible=cuda_available)
    cuda_color =  'green' if cuda_available else 'red'
    cuda_text = '(CUDA Support Enabled)' if cuda_available else '(CUDA Support Disabled)'

    # get project folders
    project_folders = glob(PATH_PROJECT)
    if project_folders:
        projectFolders = sg.DropDown(project_folders, project_folders[0], enable_events=True, pad=[5, 5], size=[90, 5], key='dbProject')
        speakers_path = Path(project_folders[0] + "/speakers.json")
        if speakers_path.is_file():
            with open(speakers_path, 'r') as json_file:
                speaker_data = json.load(json_file)
            speaker_lst = [speaker for speaker, _ in speaker_data.items()]
            
        else:
            speaker_lst = ['Default']
    max_length_name = len(max(speaker_lst, key=len)) + 2
    # All the stuff inside your window.
    layout = [
        [sg.Text('Project Settings:', font=('Arial', 12, 'bold'))],
        [projectFolders],
        [sg.Text(cuda_text, text_color=cuda_color, font=('Arial', 10,'bold')), use_cuda],
        [sg.Text('Speaker:', pad=[5, 5], justification='left', font=('Arial', 11), key='lblSpeaker'), 
         sg.DropDown(speaker_lst, speaker_lst[0], size=(max_length_name, None), font=('Arial', 11), pad=[5, 5], key='dbSpeaker'),
         sg.Checkbox('Generate for all', default=False, font=('Arial', 11), key='create_all')],
        
        [sg.Text('_' * 90)],
        
        [sg.Text('Vocoder Settings:', font=('Arial', 12, 'bold'))],
        [sg.Image(filename=MEDIA_PATH+'/kcheck.png', pad=(0, 5), key='imgradioGL'),  sg.Button('GriffinLim',pad=((0, 15), 5), key='radioGL'),
         sg.Image(filename=MEDIA_PATH+'/kleer.png', pad=(0, 5), key='imgradioWR'), sg.Button('WaveRNN', pad=((0, 15), 5), key='radioWR'),
         sg.Image(filename=MEDIA_PATH+'/kleer.png', pad=(0, 5), key='imgradioMG'), sg.Button('MelGAN', pad=((0, 15), 5),  key='radioMG')
         ],

        # [sg.Radio('GriffinLim', 0, True, font=('Arial', 11), key='radioGL'),
        #  sg.Radio('WaveRNN', 0, font=('Arial', 11), key='radioWR'),
        #  sg.Radio('PwGAN / MelGAN', 0, font=('Arial', 11), key='radioGAN')],
        
        [sg.Text('_' * 90)],
        
        [sg.Text('Emotion Settings: ', font=('Arial', 12, 'bold'))],
        [sg.Text('Emotion: ', font=('Arial', 11)), sg.DropDown(['Normal', 'Angry', 'Dominant', 'Calm'], 'Normal', font=('Arial', 11), enable_events=True, key='dbEmotion')],
        [sg.Text('Speed of Speech: ', pad=[5, 5], size=(20, 0), font=('Arial', 11), key='token0'), 
            sg.Slider(range=(-50,50), default_value=0, size=(30,12), orientation='horizontal', font=('Arial', 10, 'bold'), key='speedSlider')],  
        [sg.Text('Dominance of Speech: ', pad=[5, 5], size=(20, 0), font=('Arial', 11), key='token1and2'), 
            sg.Slider(range=(-50,50), default_value=0, size=(30, 12), orientation='horizontal', font=('Arial', 10, 'bold'), key='emotionSlider')],
        [sg.Text('Tone of Speech: ',  pad=[5, 5], size=(20, 0), font=('Arial', 11), key='token5'), 
            sg.Slider(range=(-50,50), default_value=0, size=(30, 12), orientation='horizontal', font=('Arial', 10, 'bold'), key='toneSlider')],

        [sg.Text('_' * 90)],
        
        [sg.Checkbox('Use file for speech generation', False, font=('Arial', 12, 'bold'), enable_events=True, pad=[5, 5], key='cbLoadFile')],
        [sg.Text('', size=[75, None], font=('Arial', 10), key='lblTextFile')],
        [sg.Text('Enter\nText:', font=('Arial', 12, 'bold'), pad=[5, 5]), textInput, sg.Button('Output\nFolder', size=[5, 3], enable_events=True, key='btnOpenOutput')],
        [sg.Button('Generate', key='btnGenerate'), loadingAnimation, sg.Button('Exit', key='btnExit')],
    ]
    
    # Create the Window
    window = sg.Window('GothicTTS (Generate Speech From Text)', icon=icon, layout=layout, finalize=True)
    window.FindElement('btnGenerate').Widget.config(activebackground='#273c75', activeforeground='white')
    window.FindElement('btnExit').Widget.config(activebackground='#273c75', activeforeground='white')
    window.FindElement('btnOpenOutput').Widget.config(activebackground='#273c75', activeforeground='white')
    window[active_radio_button].update(button_color=selected_color)
    for key, _ in radio_keys.items():
        window.FindElement(key).Widget.config(activebackground='#273c75', activeforeground='white')
    
    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read(timeout=100)
        
        if event in ('btnExit', 'Exit'):  # if user closes window or clicks exit
            break

        # if another porject is select, load the corresponding configuration files
        if event in 'dbProject':
            if Path(values['dbProject'] + "/speakers.json").is_file():
                with open(Path(values['dbProject'] + "/speakers.json"), 'r') as json_file:
                    speaker_data = json.load(json_file)
                speaker_lst = [speaker for speaker, _ in speaker_data.items()]
                max_length_name = len(max(speaker_lst, key=len)) + 2
                window['dbSpeaker'].update(values=speaker_lst)
                window['dbSpeaker'].set_size(size=(max_length_name, None))
            else:   
                window['dbSpeaker'].update(values=["Default"])
                window['dbSpeaker'].set_size(size=(len("Default"), None))
                
        # select emotion weights from dropdown gui element
        if event in 'dbEmotion':
            emotions = get_emotion_weights(values['dbEmotion'].lower())
            for k, v in emotions.items():
                window[k].update(v)

        if event in radio_keys:
            for k in radio_keys:
                window[k].update(button_color=sg.theme_button_color())
            window[event].update(button_color=selected_color)
            for key, value in radio_image_keys.items():
                if key == event:
                    window[value].update(filename=MEDIA_PATH+'/kcheck.png')
                else:
                    window[value].update(filename=MEDIA_PATH+'/kleer.png')
            active_radio_button = event
            
        # open a prompt if the user wants to generate speech from an input file, textbox will be disabled
        if event in 'cbLoadFile':
            if values['cbLoadFile']:
                path_to_textfile = sg.PopupGetFile('Please select a file or enter the file name', default_path=ROOT_PATH, initial_folder=ROOT_PATH,
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
                window['textInput'].update(disabled=False, background_color=TEXTINPUT_BACKGROUND)
                window['textInput'].update(text_memory)

        # open the output folder of the currently selected speaker
        if event in 'btnOpenOutput':
            if Path(values['dbProject'],'output', values['dbSpeaker'], current_date).is_dir():
                speaker_path = Path(values['dbProject'], 'output', values['dbSpeaker'], current_date)
                open_output_folder(speaker_path=speaker_path)
            else:
                default_path = Path(values['dbProject'], 'output')
                open_output_folder(speaker_path=default_path)
        
        # start synthezising speech from the input
        if event in 'btnGenerate' and not thread:
            text = values['textInput'].replace('\n', '')
            if text or (sentence_file and values['cbLoadFile']):
                if values['dbSpeaker'] in 'Default':
                    speakers_file = ''
                    speaker_name = "Default"
                else:
                    speakers_file = Path(values['dbProject'] + '/speakers.json')
                    speaker_name = values['dbSpeaker']
                speaker = speaker_lst if values['create_all'] else [speaker_name]
                
                vocoder = [val for key, val in radio_keys.items() if active_radio_button == key][0]
                
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
                                                speaker,
                                                str(vocoder),
                                                sentence_file), daemon=True)
                thread.start()

                loadingAnimation.Update(filename=PATH_LOADING_GIF, visible=True)
            else:
                sg.Popup('Type something into the textbox or select a file to generate speech.', title='Missing input!',
                         line_width=65, icon='g.ico')
                textInput.SetFocus()

 
        if status:  # If gen process finish -> stop loading.gif and open output folder
            print('Finished')
            loadingAnimation.Update(filename=PATH_LOADING_GIF, visible=False)
            if Path(values['dbProject'] + '/output').is_dir():
                speaker_path = Path(values['dbProject'], 'output', speaker_name, current_date)
                open_output_folder(speaker_path=speaker_path)
            status = False
            
            
        if thread:  # If thread is running display loading gif
            loadingAnimation.UpdateAnimation(source=PATH_LOADING_GIF, time_between_frames=100)
            thread.join(timeout=0)
            if not thread.is_alive():       # the thread finished
                loadingAnimation.Update(filename=PATH_LOADING_GIF, visible=False)
                thread = None               # reset variables for next run

    window.close()


if __name__ == '__main__':
    
    # Paths
    ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    if ROOT_PATH:
        os.chdir(ROOT_PATH)
    MEDIA_PATH = str(Path(ROOT_PATH, "media"))
    PATH_LOADING_GIF = str(Path(MEDIA_PATH, "loading.gif"))
    PATH_GUI_ICON = str(Path(MEDIA_PATH, "g"))
    PATH_PROJECT = str(Path(ROOT_PATH, "TTS_lib/trained_models/*/"))
    
    # Theme settings
    BACKGROUND_COLOR = '#dcdde1'
    TEXT_COLOR = '#24292E'
    BUTTON_COLOR = '#40739e'
    PROGESS_COLOR = '#273c75'
    TEXTINPUT_BACKGROUND = '#f5f6fa'
    sg.LOOK_AND_FEEL_TABLE['CustomTheme'] = {
                                        'BACKGROUND': BACKGROUND_COLOR,         
                                        'TEXT': TEXT_COLOR,                     
                                        'INPUT': TEXTINPUT_BACKGROUND,          
                                        'TEXT_INPUT': TEXT_COLOR,               
                                        'SCROLL': PROGESS_COLOR,                
                                        'BUTTON': ('white', BUTTON_COLOR),      
                                        'PROGRESS': (PROGESS_COLOR, '#D0D0D0'), 
                                        'BORDER': 1, 'SLIDER_DEPTH': 2, 'PROGRESS_DEPTH': 0,
                                        }
    main_gui()
