# NEON AI (TM) SOFTWARE, Software Development Kit & Application Development System
# All trademark and other rights reserved by their respective owners
# Copyright 2008-2021 Neongecko.com Inc.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions
#    and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
#    and the following disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from neon_stt_plugin_coqui import CoquiSTT
from ovos_utils.log import LOG
import neon_utils.parse_utils
import unittest
from jiwer import cer

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_PATH_EN = os.path.join(ROOT_DIR, "test_audio/en")
TEST_PATH_FR = os.path.join(ROOT_DIR, "test_audio/fr")
TEST_PATH_ES = os.path.join(ROOT_DIR, "test_audio/es")
TEST_PATH_PL = os.path.join(ROOT_DIR, "test_audio/pl")

COQUI_CREDENTIALS = '/home/mariia/neon-stt-plugin-polyglot/tests/coqui_models.jsonl'

class TestGetSTT(unittest.TestCase):


    def test_en_stt(self):
        LOG.info("ENGLISH STT MODEL")
        ground_truth = []
        hypothesis = []
        for file in os.listdir(TEST_PATH_EN):
            transcription = ' '.join(file.split('_')[:-1]).lower()
            ground_truth.append(transcription)
            path = ROOT_DIR+'/test_audio/en/'+file
            stt = CoquiSTT('en')
            text = stt.execute(path)
            hypothesis.append(text)
        error = cer(ground_truth, hypothesis)
        LOG.info('Input: {}\nOutput:{}\nWER: {}'.format(ground_truth, hypothesis, error))
        self.assertTrue(error < 0.3)

    def test_fr_stt(self):
        LOG.info("FRENCH STT MODEL")
        ground_truth = []
        hypothesis = []
        for file in os.listdir(TEST_PATH_FR):
            transcription = ' '.join(file.split('_')[:-1]).lower()
            ground_truth.append(transcription)
            path = ROOT_DIR + '/test_audio/fr/' + file
            stt = CoquiSTT('fr')
            text = stt.execute(path)
            translit = neon_utils.parse_utils.transliteration(transcription, text, 'fr')
            hypothesis.append(translit)
        error = cer(ground_truth, hypothesis)
        # LOG.info('Input: {}\nOutput:{}\nWER: {}'.format(ground_truth, hypothesis, error))
        # self.assertTrue(error < 0.3)

    def test_es_stt(self):
        LOG.info("SPANISH STT MODEL")
        ground_truth = []
        hypothesis = []
        for file in os.listdir(TEST_PATH_ES):
            transcription = ' '.join(file.split('_')[:-1]).lower()
            ground_truth.append(transcription)
            path = ROOT_DIR + '/test_audio/es/' + file
            stt = CoquiSTT('es')
            text = stt.execute(path)
            translit = neon_utils.parse_utils.transliteration(transcription, text, 'es')
            hypothesis.append(translit)
        print(ground_truth)
        print(hypothesis)
        error = cer(ground_truth, hypothesis)
        # LOG.info('Input: {}\nOutput:{}\nWER: {}'.format(ground_truth, hypothesis, error))
        # self.assertTrue(error < 0.3)

    def test_pl_stt(self):
        LOG.info("POLISH STT MODEL")
        ground_truth = []
        hypothesis = []
        for file in os.listdir(TEST_PATH_PL):
            transcription = ' '.join(file.split('_')[:-1]).lower()
            ground_truth.append(transcription)
            path = ROOT_DIR + '/test_audio/pl/' + file
            stt = CoquiSTT('pl')
            text = stt.execute(path)
            translit = neon_utils.parse_utils.transliteration(transcription, text, 'pl')
            hypothesis.append(translit)
        error = cer(ground_truth, hypothesis)
        # LOG.info('Input: {}\nOutput:{}\nWER: {}'.format(ground_truth, hypothesis, error))
        # self.assertTrue(error < 0.3)

    def test_uk_stt(self):
        LOG.info("UKRAINIAN STT MODEL")
        ground_truth = []
        hypothesis = []
        for file in os.listdir(TEST_PATH_PL):
            transcription = ' '.join(file.split('_')[:-1]).lower()
            ground_truth.append(transcription)
            path = ROOT_DIR + '/test_audio/pl/' + file
            stt = CoquiSTT('uk')
            text = stt.execute(path)
            hypothesis.append(text)
        # error = cer(ground_truth, hypothesis)
        # LOG.info('Input: {}\nOutput:{}\nWER: {}'.format(ground_truth, hypothesis, error))
        # self.assertTrue(error < 0.3)

if __name__ == '__main__':
    unittest.main()
