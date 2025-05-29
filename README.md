[中文](README-CN.md)|[English](README.md)

# Dia's ComfyUI Node

Text-to-Speech, Voice Cloning, Generating Ultra-Realistic Two-Person Conversations in One Go.

Supported oral tags include `(laughs), (clears throat), (sighs), (gasps), (coughs), (singing), (sings), (mumbles), (beep), (groans), (sniffs), (claps), (screams), (inhales), (exhales), (applause), (burps), (humming), (sneezes), (chuckle), (whistles)`.

## 📣 Updates

[2025-05-29]⚒️: Speaker audio can be loaded and saved separately.

[2025-05-19]⚒️: Released v1.1.0. Text-to-speech generation of arbitrary length is possible (requires splitting with blank lines). Speakers can be saved and loaded directly afterward.

[2025-04-24]⚒️: Released v1.0.0.

## Usage

- Prompt and text format must be as follows, use blank lines for particularly long ones:
```
[S1] Hi, how are you.
[S2] Fine, thank you, and you?
[S1] I'm fine, too.
[S2] What are you planning to do?

[S1] Hi, how are you.
[S2] Fine, thank you, and you?
```

- Generating conversation by cloning voice:
![](https://github.com/billwuhao/ComfyUI_Dia/blob/main/images/2025-04-24_08-56-13.png)

- Loading saved speaker:
![](https://github.com/billwuhao/ComfyUI_Dia/blob/main/images/2025-05-18_22-22-40.png)

- Automatically generating conversation with Gemini:
![](https://github.com/billwuhao/ComfyUI_Dia/blob/main/images/2025-05-19_00-07-01.png)

https://github.com/user-attachments/assets/6b27114d-aa9e-4f70-99c1-683994621402

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_Dia.git
cd ComfyUI_Dia
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## Model Download

- [Dia-1.6B](https://huggingface.co/nari-labs/Dia-1.6B/tree/main): Download the entire directory and place it under the `ComfyUI/models/TTS` directory.
- [weights.pth](https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth): Download and rename to `weights_44khz_8kbps_0.0.1.pth`, then place it under the `ComfyUI/models/TTS/DAC.speech.v1.0` directory.

## Acknowledgements

[dia](https://github.com/nari-labs/dia)