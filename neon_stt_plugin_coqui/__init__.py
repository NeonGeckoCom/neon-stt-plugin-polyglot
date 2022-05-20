# NEON AI (TM) SOFTWARE, Software Development Kit & Application Framework
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2022 Neongecko.com Inc.
# Contributors: Daniel McKnight, Guy Daniels, Elon Gasper, Richard Leeds,
# Regina Bloomstine, Casimiro Ferreira, Andrii Pernatii, Kirill Hrymailo
# BSD-3 License
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS;  OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
from neon_speech.stt import STT
from typing import Optional
import os
from neon_utils.logger import LOG
import os.path
import shlex
import subprocess
import wave
import yaml
from pipes import quote
import deepspeech
from speech_recognition import AudioData

import requests


class CoquiSTT(STT):
    def __init__(self, config: dict = None):
        if isinstance(config, str):
            LOG.warning(f"Expected dict config but got: {config}")
            config = {"lang": config}
        config = config or dict()
        super().__init__(config)

        self.lang = config.get('lang') or 'en'

        # Model creation
        model, scorer = self.download_coqui_model()
        try:
            LOG.info(f"Loading model file: {model}")
            model = deepspeech.Model(model)
        except RuntimeError as e:
            LOG.exception(e)
            LOG.warning("Retrying model download")
            os.remove(model)
            model, scorer = self.download_coqui_model()
            model = deepspeech.Model(model)

        # Adding scorer
        if scorer:
            try:
                model.enableExternalScorer(scorer)
            except RuntimeError as e:
                LOG.exception(e)
                LOG.error(f"Not loading external scorer: {scorer}")
        # setting beam width A larger beam width value generates better results at the cost of decoding time
        beam_width = 500
        model.setBeamWidth(beam_width)
        self.model = model

    def get_model(self, model_url: str, scorer_url: Optional[str]):
        '''
        Downloading model and scorer for the specific language
        from CoQui models web-page: https://coqui.ai/models.
        Creating model and a scorer files in ~/.local/share/neon/

        Parameters:
                    model_url (str): url to model downloading in .pbmm format
                    scorer_url (str): url to scorer downloading in .scorer format

        Returns:
                    model, scorer (tuple): tuple that contains pathes to model and scorer
        '''
        try:
            if not os.path.isdir(os.path.expanduser("~/.local/share/neon/")):
                os.makedirs(os.path.expanduser("~/.local/share/neon/"))
            if model_url:
                model_path = os.path.expanduser(f"~/.local/share/neon/coqui-{self.lang}-models.pbmm")
                if not os.path.isfile(model_path):
                    LOG.info(f"Downloading {model_url}")
                    model = requests.get(model_url, allow_redirects=True)
                    with open(model_path, "wb") as out:
                        out.write(model.content)
                    LOG.info(f"Model Downloaded to {model_path}")
            else:
                raise ValueError("Null model_url passed")

            if scorer_url:
                scorer_path = os.path.expanduser(f"~/.local/share/neon/coqui-{self.lang}-models.scorer")
                if not os.path.isfile(scorer_path):
                    LOG.info(f"Downloading {scorer_url}")
                    scorer = requests.get(scorer_url, allow_redirects=True)
                    with open(scorer_path, "wb") as out:
                        out.write(scorer.content)
                    LOG.info(f"Scorer Downloaded to {scorer_path}")
            else:
                scorer_path = None
                  
            return model_path, scorer_path
        except Exception as e:
            LOG.info(f"Error getting deepspeech models! {e}")

    def download_coqui_model(self):
        '''
        Parsing yaml file with model and scorer urls
        Calls get_model() function for model and scorer downloading
        from CoQui models web-page: https://coqui.ai/models.

        Returns:
                    model, scorer (tuple): tuple that contains pathes to model and scorer
        '''
        credentials_path = os.path.dirname(os.path.abspath(__file__))+'/coqui_models.yml'
        with open(credentials_path, 'r') as json_file:
            models_dict = yaml.load(json_file, Loader=yaml.FullLoader)
        if self.lang not in models_dict.keys():
            raise RuntimeError(f"{self.lang} is not supported")
        if models_dict[self.lang]['scorer_url'] != "":
            model, scorer = \
                self.get_model(models_dict[self.lang]['model_url'],
                               models_dict[self.lang]['scorer_url'])
            return model, scorer
        else:
            model, scorer = self.get_model(models_dict[self.lang]['model_url'],  None)
            return model, scorer

    def convert_samplerate(self, audio, desired_sample_rate):
    
        """
        Audio rate convertation if it doesn't satisfy model sample rate.
        Returns buffer output of audio in numpy array.

        Parameters:
                    audio (str): path to audio file
                    desired_sample_rate: sample rate desired by coqui model

        Returns:
                    (numpy array): buffer output of audio
        """
        
        sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(
            quote(audio), desired_sample_rate)
        try:
            output = subprocess.check_output(shlex.split(sox_cmd), stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
        except OSError as e:
            raise OSError(e.errno,
                          'SoX not found, use {}hz files or install it: {}'.format(desired_sample_rate, e.strerror))
        return np.frombuffer(output, np.int16)

    def get_audio_data(self, audio_path):

        """
        Constructs an AudioData instance with the same parameters
        as the source and the specified frame_data.

        Converts audio samplerate if the original 
        samplerate doesn't satisfy model sample rate.

        Parameters:
                    audio_path (str): path to audio file

        Returns:
                    audio_length (int): length of the input audio in sec
                    audio_data (AudioData): audio data of the input audio file
        """
        
        fin = wave.open(audio_path, 'rb')

        desired_sample_rate = self.model.sampleRate()
        desired_sample_width = fin.getsampwidth()

        # samplerate conversion
        fs_orig = fin.getframerate()
        if fs_orig != desired_sample_rate:
            LOG.info(f'Warning: original sample rate ({fs_orig}) is different '
                     f'than {desired_sample_rate}hz. Resampling might produce '
                     f'erratic speech recognition.')
            audio = self.convert_samplerate(audio_path, desired_sample_rate)
        else:
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
        audio_data = AudioData(audio, desired_sample_rate,
                               desired_sample_width)

        # getting audio length
        audio_length = fin.getnframes() * (1 / fs_orig)
        fin.close()

        return audio_length, audio_data
    
    def execute(self, audio: AudioData, language: str = None):
        '''
        Executes speach recognition

        Parameters:
                    audio (AudioData): AudioData of the input audio
                    language (str): language code associated with audio
        Returns:
                    text (str): recognized text
        '''
        # TODO: Handle models per-language
        if audio.sample_rate != self.model.sampleRate():
            LOG.warning(f"Audio SR ({audio.sample_rate}) "
                        f"different from model: {self.model.sampleRate()}")
        transcription = str(self.model.stt(np.frombuffer(audio.get_raw_data(),
                                                         dtype=np.int16)))
        return transcription
