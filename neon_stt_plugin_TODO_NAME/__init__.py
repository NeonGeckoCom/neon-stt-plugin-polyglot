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

from inspect import signature
from queue import Queue

try:
    from neon_speech.stt import StreamingSTT, StreamThread
except ImportError:
    from ovos_plugin_manager.templates.stt import StreamingSTT, StreamThread
from neon_utils.logger import LOG


class TemplateStreamingSTT(StreamingSTT):  # TODO: Replace 'Template' with STT name
    def __init__(self, results_event, config=None):
        if len(signature(super(TemplateStreamingSTT, self).__init__).parameters) == 0:
            LOG.warning(f"Deprecated Signature Found; config will be ignored and results_event will not be handled!")
            super(TemplateStreamingSTT, self).__init__()
        else:
            super(TemplateStreamingSTT, self).__init__(results_event=results_event, config=config)

        if not hasattr(self, "results_event"):
            self.results_event = None
        # override language with module specific language selection
        self.language = self.config.get('lang') or self.lang
        self.queue = None
        self.client = None  # TODO: Initialize STT engine here DM

    def create_streaming_thread(self):
        self.queue = Queue()
        return TemplateStreamThread(
            self.queue,
            self.language,
            self.client,
            self.results_event
        )


class TemplateStreamThread(StreamThread):  # TODO: Replace 'Template' with STT name
    def __init__(self, queue, lang, client, results_event):
        super().__init__(queue, lang)
        self.client = client
        self.results_event = results_event
        self.transcriptions = []

    def handle_audio_stream(self, audio, language):
        # TODO: Handle audio stream and populate `self.transcriptions` until there is no more speech in input
        self.transcriptions = []

        if not self.transcriptions:
            LOG.info("Transcription is empty")
            self.text = None
            self.transcriptions = []
        else:
            LOG.debug("Audio had data")
            self.text = self.transcriptions[0]

        if self.results_event:
            self.results_event.set()
        return self.transcriptions
