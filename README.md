# NeonAI Template STT Plugin Polyglot
[Mycroft](https://mycroft-ai.gitbook.io/docs/mycroft-technologies/mycroft-core/plugins) compatible
STT Plugin for Polyglot DeepSpeech stt models streaming Speech-to-Text. 
# Tkinter installation if python version < 3.1
sudo apt-get install python3-tk-dbg

# Neon-sftp installation:
git clone --single-branch --branch initial_structure https://github.com/NeonGeckoCom/neon-sftp.git

# Configuration:
# TODO: Specify any optional or required configuration values
```yaml
stt:
    module: neon-stt-plugin-polyglot  
    neon_stt_plugin_polyglot : {}  # TODO: Any module config

```