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

import deepspeech
import numpy as np
from neon_speech.stt import STT
import os
from neon_utils.logger import LOG
from os import listdir
from os.path import join
import shlex
import subprocess
import wave
from timeit import default_timer as timer

try:
    from shhlex import quote
except ImportError:
    from pipes import quote


class PolyglotSTT(STT):

    def __init__(self, lang, audio):
        """
        Model and scorer initialization for the specific language.
        """
        self.audio = audio
        self.lang = lang or 'en'
        # Model creation
        ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
        model_file_path = ROOT_DIR+'/Polyglot_models/'+self.lang
        model_path = join(model_file_path, [f for f in listdir(model_file_path) if f.endswith('.pbmm')][0])
        scorer_file_path = join(model_file_path, [f for f in listdir(model_file_path) if f.endswith('.scorer')][0])
        model = deepspeech.Model(model_path)
        #  Adding scorer and other parameters
        model.enableExternalScorer(scorer_file_path)
        # setting beam width A larger beam width value generates better results at the cost of decoding time
        beam_width = 500
        model.setBeamWidth(beam_width)
        self.model = model

    def convert_samplerate(self, audio, desired_sample_rate):
        """
        Audio rate convertation if it doesn't satisfy model sample rate.
        Returns buffer output of audio in numpy array.
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
        return desired_sample_rate, np.frombuffer(output, np.int16)


    def execute(self, language=None):
        desired_sample_rate = self.model.sampleRate()
        # reading audio file
        fin = wave.open(self.audio, 'rb')
        fs_orig = fin.getframerate()
        if fs_orig != desired_sample_rate:
            LOG.info('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(
                    fs_orig, desired_sample_rate))
            fs_new, audio = self.convert_samplerate(self.audio, desired_sample_rate)
        else:
            audio = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)
        audio_length = fin.getnframes() * (1 / fs_orig)
        fin.close()
        LOG.info('Running inference.')
        inference_start = timer()
        # sphinx-doc: python_ref_inference_start
        LOG.info("Transcription: "+str(self.model.stt(audio)))
        # sphinx-doc: python_ref_inference_stop
        inference_end = timer() - inference_start
        LOG.info('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length))
        return str(self.model.stt(audio))