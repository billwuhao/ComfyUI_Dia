[中文](README-CN.md)|[English](README.md)

# Dia 的 ComfyUI 节点

![](https://github.com/billwuhao/ComfyUI_Dia/blob/main/images/2025-04-24_08-56-13.png)

文本转语音, 声音克隆, 一次生成超真实对话. 克隆声音时, clone-prompt-text 需要放在 gen-text 前面, 只会生成 gen-text 的音频.

支持的口头标签有 `(laughs), (clears throat), (sighs), (gasps), (coughs), (singing), (sings), (mumbles), (beep), (groans), (sniffs), (claps), (screams), (inhales), (exhales), (applause), (burps), (humming), (sneezes), (chuckle), (whistles)`.

## 📣 更新

[2025-04-24]⚒️: 发布 v1.0.0。

## 安装

```
cd ComfyUI/custom_nodes
git clone https://github.com/billwuhao/ComfyUI_Dia.git
cd ComfyUI_Dia
pip install -r requirements.txt

# python_embeded
./python_embeded/python.exe -m pip install -r requirements.txt
```

## 模型下载

- [Dia-1.6B](https://huggingface.co/nari-labs/Dia-1.6B/tree/main): 整个目录下载放到 `ComfyUI/models/TTS` 目录下.
- [weights.pth](https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.1/weights.pth): 下载后重命名为 `weights_44khz_8kbps_0.0.1.pth` 放到 `ComfyUI/models/TTS/DAC.speech.v1.0` 目录下.

## 鸣谢

[dia](https://github.com/nari-labs/dia)