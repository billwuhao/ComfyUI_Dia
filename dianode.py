import numpy as np
import torch
import folder_paths
import os
import sys
import tempfile
import torchaudio
from typing import Optional, List, Union
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from dia.model import Dia


models_dir = folder_paths.models_dir
model_path = os.path.join(models_dir, "TTS")
config_path = os.path.join(model_path, "Dia-1.6B", "config.json")
checkpoint_path = os.path.join(model_path, "Dia-1.6B")
dac_model_path = os.path.join(model_path, "DAC.speech.v1.0", "weights_44khz_8kbps_0.0.1.pth")
cache_dir = folder_paths.get_temp_directory()
speakers_path = os.path.join(model_path, "speakers", "dialogue_speakers")


def cache_audio_tensor(
    cache_dir,
    audio_tensor: torch.Tensor,
    sample_rate: int,
    filename_prefix: str = "cached_audio_",
    audio_format: Optional[str] = ".wav"
) -> str:
    try:
        with tempfile.NamedTemporaryFile(
            prefix=filename_prefix,
            suffix=audio_format,
            dir=cache_dir,
            delete=False 
        ) as tmp_file:
            temp_filepath = tmp_file.name
        
        torchaudio.save(temp_filepath, audio_tensor, sample_rate)

        return temp_filepath
    except Exception as e:
        raise Exception(f"Error caching audio tensor: {e}")


def concatenate(audio_a, audio_b):
    """
    Concatenates two audio waveforms after resampling and smart channel adjustment.

    Args:
        audio_a (dict): First audio input {"waveform": tensor, "sample_rate": int}.
        audio_b (dict): Second audio input {"waveform": tensor, "sample_rate": int}.

    Returns:
        tuple: A tuple containing the output audio dictionary.
                ({"waveform": concatenated_tensor, "sample_rate": max_sr},)
    """
    waveform_a = audio_a["waveform"]
    sr_a = audio_a["sample_rate"]
    waveform_b = audio_b["waveform"]
    sr_b = audio_b["sample_rate"]

    final_sr = max(sr_a, sr_b)

    resampled_waveform_a = waveform_a
    if sr_a != final_sr:
        print(f"Concatenate Audio Node: Resampling audio A from {sr_a} Hz to {final_sr} Hz")
        resample_a = torchaudio.transforms.Resample(orig_freq=sr_a, new_freq=final_sr).to(waveform_a.device)
        resampled_waveform_a = resample_a(waveform_a)

    resampled_waveform_b = waveform_b
    if sr_b != final_sr:
        print(f"Concatenate Audio Node: Resampling audio B from {sr_b} Hz to {final_sr} Hz")
        resample_b = torchaudio.transforms.Resample(orig_freq=sr_b, new_freq=final_sr).to(waveform_b.device)
        resampled_waveform_b = resample_b(waveform_b)

    channels_a = resampled_waveform_a.shape[1]
    channels_b = resampled_waveform_b.shape[1]

    if channels_a == 1 and channels_b == 1:
        target_channels = 1 
        print("Concatenate Audio Node: Both inputs are mono, output will be mono (1 channel).")
    else:
        target_channels = 2 
        print(f"Concatenate Audio Node: At least one input is not mono ({channels_a} vs {channels_b}), output will be stereo (2 channels).")

    def adjust_channels(wf, current_channels, target_channels, name):
        if current_channels == target_channels:
            return wf
        elif target_channels == 1 and current_channels > 1:

                print(f"Concatenate Audio Node Warning: Attempting to downmix {name} from {current_channels} to {target_channels} (mono). Simple average downmix applied.")

                return wf.mean(dim=1, keepdim=True)
        elif target_channels == 2:
            if current_channels == 1:

                print(f"Concatenate Audio Node: Converting {name} from {current_channels} to {target_channels} channels (mono to stereo).")
                return wf.repeat(1, target_channels, 1) 
            elif current_channels > 2:

                print(f"Concatenate Audio Node Warning: Converting {name} from {current_channels} to {target_channels} channels (multi-channel to stereo). Applying simple average downmix.")

                mono_wf = wf.mean(dim=1, keepdim=True)
                return mono_wf.repeat(1, target_channels, 1)

        else:

                raise RuntimeError(f"Concatenate Audio Node: Unsupported channel adjustment requested for {name}: from {current_channels} to {target_channels}.")

    adjusted_waveform_a = adjust_channels(resampled_waveform_a, channels_a, target_channels, "Audio A")
    adjusted_waveform_b = adjust_channels(resampled_waveform_b, channels_b, target_channels, "Audio B")

    concatenated_waveform = torch.cat((adjusted_waveform_a, adjusted_waveform_b), dim=2)

    output_audio = {
        "waveform": concatenated_waveform,
        "sample_rate": final_sr 
    }

    return output_audio

MODEL_CACHE = None
class DiaTTSRun:
    def __init__(self):
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.device = torch.device(device)
        print(f"Using device: {device}")

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "prompt": ("STRING",  {
                    "multiline": True, 
                    "default": ""}),
                "audio_s1": ("AUDIO",),
                "audio_s2": ("AUDIO",),
                "max_new_tokens": ("INT", {"default": 3000, "min": 860, "max": 3072, "step": 2}),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 5.0, "step": 0.1}),
                "temperature": ("FLOAT", {"default": 1.3, "min": 1.0, "max": 1.5, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.80, "max": 1.0, "step": 0.01}),
                "cfg_filter_top_k": ("INT", {"default": 30, "min": 15, "max": 50, "step": 1}),
                # "speed_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                # "use_torch_compile": ("BOOLEAN", {"default": True}),
                "unload_model": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "save_speakers": ("BOOLEAN", {"default": True}),
                "speakers_id": ("STRING", {"default": "A_and_B"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run_inference"
    CATEGORY = "🎤MW/MW-Dia"

    def run_inference(
        self,
        text: str, 
        prompt: str,
        audio_s1: dict,
        audio_s2: dict,
        max_new_tokens: int,
        cfg_scale: float,
        temperature: float,
        top_p: float,
        cfg_filter_top_k: int,
        # speed_factor: float,
        # use_torch_compile: bool,
        unload_model: bool,
        save_speakers: bool,
        speakers_id: str,
    ):
        global MODEL_CACHE
        if MODEL_CACHE is None:
            dia_model_path = os.path.join(checkpoint_path, "dia-v0_1.pth")
            MODEL_CACHE = Dia.from_local(config_path, dia_model_path, compute_dtype="float16", device=self.device, dac_model_path=dac_model_path)

        if text.strip() == "" or prompt.strip() == "":
            raise ValueError("Text or prompt input is empty.")

        texts = [i.strip() for i in re.split(r'\n\s*\n', text.strip()) if i.strip()]

        speakers_audio_input = concatenate(audio_s1, audio_s2)
        audio_data = speakers_audio_input["waveform"].squeeze(0)
        sr = speakers_audio_input["sample_rate"]

        if save_speakers:
            if speakers_id.strip() == "":
                raise ValueError("Speakers ID is empty.")

            if not os.path.exists(speakers_path):
                os.makedirs(speakers_path)

            audio_s1_path = os.path.join(speakers_path, f"{speakers_id}_1.wav")
            torchaudio.save(audio_s1_path, audio_s1["waveform"].squeeze(0), audio_s1["sample_rate"])

            audio_s2_path = os.path.join(speakers_path, f"{speakers_id}_2.wav")
            torchaudio.save(audio_s2_path, audio_s2["waveform"].squeeze(0), audio_s2["sample_rate"])

            text_path = os.path.join(speakers_path, f"{speakers_id}.txt")
            
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(prompt)

        audio_path = cache_audio_tensor(
                                        cache_dir,
                                        audio_data,
                                        sr,
                                    )

        audio_prompt = MODEL_CACHE.load_audio(audio_path)

        text_gens = [prompt.strip() + f"\n{i}" for i in texts]
        audio_prompts = [audio_prompt for i in range(len(texts))]

        # Use torch.inference_mode() context manager for the generation call
        with torch.inference_mode():
            output_audio_np = MODEL_CACHE.generate(
                text=text_gens,
                max_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
                use_torch_compile=False, 
                audio_prompt=audio_prompts,
                verbose=True,
            )

        if output_audio_np[0] is not None:
            # Get sample rate from the loaded DAC model
            output_sr = 44100

            output_audio = np.concatenate(output_audio_np).astype(np.float32)
            output_audio = np.clip(output_audio, -1.0, 1.0)

            # Unload model if requested
            if unload_model:
                MODEL_CACHE = None
                torch.cuda.empty_cache()

            return ({"waveform": torch.from_numpy(output_audio).unsqueeze(0).unsqueeze(0), "sample_rate": output_sr},)

        else:
            raise  RuntimeError("Audio generation failed.")


def get_all_files(
    root_dir: str,
    return_type: str = "list",
    extensions: Optional[List[str]] = None,
    exclude_dirs: Optional[List[str]] = None,
    relative_path: bool = False
) -> Union[List[str], dict]:
    """
    递归获取目录下所有文件路径
    
    :param root_dir: 要遍历的根目录
    :param return_type: 返回类型 - "list"(列表) 或 "dict"(按目录分组)
    :param extensions: 可选的文件扩展名过滤列表 (如 ['.py', '.txt'])
    :param exclude_dirs: 要排除的目录名列表 (如 ['__pycache__', '.git'])
    :param relative_path: 是否返回相对路径 (相对于root_dir)
    :return: 文件路径列表或字典
    """
    file_paths = []
    file_dict = {}
    
    # 规范化目录路径
    root_dir = os.path.normpath(root_dir)
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 处理排除目录
        if exclude_dirs:
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        
        current_files = []
        for filename in filenames:
            # 扩展名过滤
            if extensions:
                if not any(filename.lower().endswith(ext.lower()) for ext in extensions):
                    continue
            
            # 构建完整路径
            full_path = os.path.join(dirpath, filename)
            
            # 处理相对路径
            if relative_path:
                full_path = os.path.relpath(full_path, root_dir)
            
            current_files.append(full_path)
        
        if return_type == "dict":
            # 使用相对路径或绝对路径作为键
            dict_key = os.path.relpath(dirpath, root_dir) if relative_path else dirpath
            if current_files:
                file_dict[dict_key] = current_files
        else:
            file_paths.extend(current_files)
    
    return file_dict if return_type == "dict" else file_paths

def get_speakers():
    speakers_dir = os.path.join(checkpoint_path, "speakers")
    if not os.path.exists(speakers_dir):
        os.makedirs(speakers_dir, exist_ok=True)
        return []
    speakers = get_all_files(speakers_dir, extensions=[".txt"], relative_path=True)
    return speakers


class DiaSpeakersPreview:
    def __init__(self):
        self.speakers_dir = speakers_path
    @classmethod
    def INPUT_TYPES(s):
        speakers = get_speakers()
        return {
            "required": {"speaker":(speakers,),},}

    RETURN_TYPES = ("STRING", "AUDIO", "AUDIO",)
    RETURN_NAMES = ("prompt", "audio_s1", "audio_s2",)
    FUNCTION = "preview"
    CATEGORY = "🎤MW/MW-Dia"

    def preview(self, speaker):
        text_path = os.path.join(self.speakers_dir, speaker)
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        audio_s1_path = text_path.replace(".txt", "_1.wav")
        waveform, sample_rate = torchaudio.load(audio_s1_path)
        waveform = waveform.unsqueeze(0)
        output_audio_s1 = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }

        audio_s2_path = text_path.replace(".txt", "_2.wav")
        waveform, sample_rate = torchaudio.load(audio_s2_path)
        waveform = waveform.unsqueeze(0)
        output_audio_s2 = {
            "waveform": waveform,
            "sample_rate": sample_rate
        }

        return (text, output_audio_s1, output_audio_s2)


NODE_CLASS_MAPPINGS = {
    "DiaTTSRun": DiaTTSRun,
    "DiaSpeakersPreview": DiaSpeakersPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiaSpeakersPreview": "DiaTTS Speakers Preview",
    "DiaTTSRun": "DiaTTS Run",
}