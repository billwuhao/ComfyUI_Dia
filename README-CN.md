[中文](README-CN.md)|[English](README.md)

# Dia 的 ComfyUI 节点

文本转语音, 声音克隆, 一次生成超真实双人对话.

支持的口头标签有 `(laughs), (clears throat), (sighs), (gasps), (coughs), (singing), (sings), (mumbles), (beep), (groans), (sniffs), (claps), (screams), (inhales), (exhales), (applause), (burps), (humming), (sneezes), (chuckle), (whistles)`.

## 📣 更新

[2025-05-29]⚒️: 说话这音频分开加载和保存.

[2025-05-19]⚒️: 发布 v1.1.0。 可任意长度文本转语音生成(需要空行分割). 可保存说话者, 之后直接加载.

[2025-04-24]⚒️: 发布 v1.0.0。

## 用法

- 提示和文本格式必须如下, 特别长用空行分割:
```
[S1] Hi, how are you.
[S2] Fine, thank you, and you?
[S1] I'm fine, too.
[S2] What are you planning to do?

[S1] Hi, how are you.
[S2] Fine, thank you, and you?
```

- 克隆声音生成对话:
![](https://github.com/billwuhao/ComfyUI_Dia/blob/main/images/2025-04-24_08-56-13.png)

- 加载已保存说话者:
![](https://github.com/billwuhao/ComfyUI_Dia/blob/main/images/2025-05-18_22-22-40.png)

- 用 Gemini 自动生成对话:
![](https://github.com/billwuhao/ComfyUI_Dia/blob/main/images/2025-05-19_00-07-01.png)

https://github.com/user-attachments/assets/6b27114d-aa9e-4f70-99c1-683994621402

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