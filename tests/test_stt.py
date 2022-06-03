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
from timeit import default_timer as timer
from datetime import date

ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
TEST_PATH_EN = os.path.join(ROOT_DIR, "test_audio/en")
TEST_PATH_FR = os.path.join(ROOT_DIR, "test_audio/fr")
TEST_PATH_ES = os.path.join(ROOT_DIR, "test_audio/es")
TEST_PATH_PL = os.path.join(ROOT_DIR, "test_audio/pl")
TEST_PATH_UK = os.path.join(ROOT_DIR, "test_audio/uk")

today = date.today()
reports_folder = ROOT_DIR+'/test_reports_'+str(today)+'/'
if os.path.isdir(reports_folder):
    LOG.info('Reports dir exists')
else:
    os.mkdir(reports_folder)


class TestGetSTT(unittest.TestCase):

    def evaluation_script(self, folder, lang, report_name):
        today = date.today()
        ground_truth = []
        hypothesis = []
        df_list = []
        for file in os.listdir(folder):
            transcription = ' '.join(file[:-4].split('_')).lower()
            ground_truth.append(transcription)
            path = folder+'/'+file
            stt = CoquiSTT(lang)
            LOG.info('Running inference.')
            inference_start = timer()
            audio_length, audio_data = stt.get_audio_data(path)
            text = stt.execute(audio_data)
            inference_end = timer() - inference_start
            LOG.info('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length))
            # model's output
            translit = neon_utils.parse_utils.transliteration(transcription, text, lang)
            LOG.info("Transcription transliterated: "+translit)
            hypothesis.append(translit)
            df_list.append([transcription, text, audio_length, inference_end])
        error = cer(ground_truth, hypothesis)
        self.assertTrue(error < 0.3)
        LOG.info('Input: {}\nOutput:{}\nWER: {}'.format(ground_truth, hypothesis, error))
        #creating dataframe
        df_list.append(['CER', error, '', ''])
        dataframe = pd.DataFrame(df_list, columns=['ground_truth', 'hypothesis', 'audio_length', 'inference_end'])
        df_path = ROOT_DIR+'/test_reports_'+str(today)+'/'+report_name+'.csv'
        dataframe.to_csv(df_path)


    def test_en_stt(self):
        LOG.info("ENGLISH STT MODEL")
        male_folder = TEST_PATH_EN+'/male'
        self.evaluation_script(male_folder, 'en', 'en_male_report')

    def test_fr_stt(self):
        LOG.info("FRENCH STT MODEL")
        ground_truth = []
        hypothesis = []
        for file in os.listdir(TEST_PATH_FR):
            # desired transcription
            transcription = ' '.join(file.split('_')[:-1]).lower()
            ground_truth.append(transcription)
            path = ROOT_DIR + '/test_audio/fr/' + file
            stt = CoquiSTT('fr')
            LOG.info('Running inference.')
            inference_start = timer()
            audio_length, audio_data = stt.get_audio_data(path)
            text = stt.execute(audio_data)
            LOG.info("Transcription: "+text)
            inference_end = timer() - inference_start
            LOG.info('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length))
            # transliteration of desired transcription
            translit = neon_utils.parse_utils.transliteration(transcription, text, 'fr')
            print('translit: ', translit)
            # model's output
            hypothesis.append(translit[0])
        error = cer(ground_truth, hypothesis)
        LOG.info('Input: {}\nOutput:{}\nWER: {}'.format(ground_truth, hypothesis, error))
        self.assertTrue(error < 0.3)

    def test_es_stt(self):
        LOG.info("SPANISH STT MODEL")
        ground_truth = []
        hypothesis = []
        for file in os.listdir(TEST_PATH_ES):
            transcription = ' '.join(file.split('_')[:-1]).lower()
            ground_truth.append(transcription)
            path = ROOT_DIR + '/test_audio/es/' + file
            stt = CoquiSTT('es')
            LOG.info('Running inference.')
            inference_start = timer()
            audio_length, audio_data = stt.get_audio_data(path)
            text = stt.execute(audio_data)
            LOG.info("Transcription: "+text)
            inference_end = timer() - inference_start
            LOG.info('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length))
            translit = neon_utils.parse_utils.transliteration(transcription, text, 'es')
            hypothesis.append(translit[0])
        error = cer(ground_truth, hypothesis)
        LOG.info('Input: {}\nOutput:{}\nWER: {}'.format(ground_truth, hypothesis, error))
        self.assertTrue(error < 0.3)

    def test_pl_stt(self):
        LOG.info("POLISH STT MODEL")
        ground_truth = []
        hypothesis = []
        for file in os.listdir(TEST_PATH_PL):
            transcription = ' '.join(file.split('_')[:-1]).lower()
            ground_truth.append(transcription)
            path = ROOT_DIR + '/test_audio/pl/' + file
            stt = CoquiSTT('pl')
            LOG.info('Running inference.')
            inference_start = timer()
            audio_length, audio_data = stt.get_audio_data(path)
            text = stt.execute(audio_data)
            LOG.info("Transcription: "+text)
            inference_end = timer() - inference_start
            LOG.info('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length))
            translit = neon_utils.parse_utils.transliteration(transcription, text, 'pl')
            hypothesis.append(translit[0])
        error = cer(ground_truth, hypothesis)
        LOG.info('Input: {}\nOutput:{}\nWER: {}'.format(ground_truth, hypothesis, error))
        self.assertTrue(error < 0.3)

    def test_uk_stt(self):
        LOG.info("UKRAINIAN STT MODEL")
        female_folder = TEST_PATH_UK+'/female'
        self.evaluation_script(female_folder, 'uk', 'uk_female_report')

    def test_pl_stt(self):
        LOG.info("POLISH STT MODEL")
        male_folder = TEST_PATH_PL+'/male'
        female_folder = TEST_PATH_PL+'/female'
        self.evaluation_script(male_folder, 'pl', 'pl_report_male')
        self.evaluation_script(female_folder, 'pl', 'pl_report_female')


if __name__ == '__main__':
    unittest.main()
