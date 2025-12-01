import os
import sys
import torch
import numpy as np
import cv2
import face_alignment
import folder_paths
from PIL import Image
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2FeatureExtractor
import torchvision.transforms as transforms
import torchaudio

# ==============================================================================
# SETUP PATHS & IMPORTS
# ==============================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Check if source folders exist
generator_path = os.path.join(current_dir, "generator")
renderer_path = os.path.join(current_dir, "renderer")

if not os.path.exists(generator_path) or not os.path.exists(renderer_path):
    raise ImportError(
        f"\n\nCRITICAL ERROR: Missing source code folders!\n"
        f"You must copy the 'generator' and 'renderer' folders from the IMTalker repository\n"
        f"into this folder: {current_dir}\n"
    )

try:
    from generator.FM import FMGenerator
    from renderer.models import IMTRenderer
except ImportError as e:
    raise ImportError(
        f"\n\nIMPORT ERROR: {e}\n"
        f"Please ensure 'generator' and 'renderer' folders contain __init__.py files\n"
        f"and all requirements.txt dependencies are installed.\n"
    )

# ==============================================================================
# CONFIGURATION
# ==============================================================================
IMTALKER_MODELS_DIR = os.path.join(folder_paths.models_dir, "imtalker")
os.makedirs(IMTALKER_MODELS_DIR, exist_ok=True)

class IMTalkerConfig:
    def __init__(self, device, wav2vec_path):
        self.device = device
        self.wav2vec_model_path = wav2vec_path 
        self.input_size = 256
        self.input_nc = 3
        self.fps = 25.0
        self.rank = device 
        self.sampling_rate = 16000
        self.audio_marcing = 2
        self.wav2vec_sec = 2.0
        self.attention_window = 5
        self.only_last_features = True
        self.audio_dropout_prob = 0.1
        self.style_dim = 512
        self.dim_a = 512
        self.dim_h = 512
        self.dim_e = 7
        self.dim_motion = 32
        self.dim_c = 32
        self.dim_w = 32
        self.fmt_depth = 8
        self.num_heads = 8
        self.mlp_ratio = 4.0
        self.no_learned_pe = False
        self.num_prev_frames = 10
        self.max_grad_norm = 1.0
        self.ode_atol = 1e-5
        self.ode_rtol = 1e-5
        self.nfe = 10
        self.torchdiffeq_ode_method = 'euler'
        self.a_cfg_scale = 3.0
        self.swin_res_threshold = 128
        self.window_size = 8
        
        # ATTRIBUTES
        self.fix_noise_seed = True 
        self.seed = 42

class IMTalkerLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "auto_download": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMTALKER_MODEL",)
    RETURN_NAMES = ("model_bundle",)
    FUNCTION = "load_models"
    CATEGORY = "IMTalker"

    def load_models(self, auto_download):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Define Paths
        wav2vec_path = os.path.join(IMTALKER_MODELS_DIR, "wav2vec2-base-960h")
        renderer_ckpt = os.path.join(IMTALKER_MODELS_DIR, "renderer.ckpt")
        generator_ckpt = os.path.join(IMTALKER_MODELS_DIR, "generator.ckpt")

        # 2. Handle Downloads
        files_to_download = [
            "renderer.ckpt",
            "generator.ckpt",
            "wav2vec2-base-960h/config.json",
            "wav2vec2-base-960h/pytorch_model.bin",
            "wav2vec2-base-960h/preprocessor_config.json",
            "wav2vec2-base-960h/feature_extractor_config.json",
        ]
        
        REPO_ID = "cbsjtu01/IMTalker"
        
        if auto_download:
            for remote_filename in files_to_download:
                local_file = os.path.join(IMTALKER_MODELS_DIR, remote_filename)
                if not os.path.exists(local_file):
                    print(f"IMTalker: Downloading {remote_filename}...")
                    try:
                        hf_hub_download(
                            repo_id=REPO_ID,
                            filename=remote_filename,
                            local_dir=IMTALKER_MODELS_DIR,
                            local_dir_use_symlinks=False
                        )
                    except Exception as e:
                        print(f"IMTalker Download Error for {remote_filename}: {e}")

        # 3. Verification
        if not os.path.exists(renderer_ckpt) or not os.path.exists(generator_ckpt):
            raise FileNotFoundError(f"Models not found in {IMTALKER_MODELS_DIR}. Please enable auto_download.")

        if not os.path.exists(os.path.join(wav2vec_path, "config.json")):
             print("IMTalker Warning: Local wav2vec not found. Using 'facebook/wav2vec2-base-960h' from cache.")
             wav2vec_path = "facebook/wav2vec2-base-960h"

        # 4. Initialize Config
        opt = IMTalkerConfig(device, wav2vec_path)
        
        # 5. Load Models
        print(f"IMTalker: Loading Renderer on {device}...")
        renderer = IMTRenderer(opt).to(device)
        self._load_ckpt(renderer, renderer_ckpt, "gen.")
        renderer.eval()

        print(f"IMTalker: Loading Generator on {device}...")

        # --- PATCH: Fix for Transformers Attention Error ---
        patch_applied = False
        orig_from_pretrained = None
        local_wav2vec_class = None

        try:
            from generator.wav2vec2 import Wav2VecModel
            local_wav2vec_class = Wav2VecModel
            orig_from_pretrained = Wav2VecModel.from_pretrained

            @classmethod
            def patched_from_pretrained(cls, *args, **kwargs):
                kwargs["attn_implementation"] = "eager"
                return orig_from_pretrained(*args, **kwargs)

            Wav2VecModel.from_pretrained = patched_from_pretrained
            patch_applied = True
            
        except ImportError:
            pass

        generator = FMGenerator(opt).to(device)
        self._load_fm_ckpt(generator, generator_ckpt, device)
        generator.eval()

        if patch_applied and local_wav2vec_class and orig_from_pretrained:
            local_wav2vec_class.from_pretrained = orig_from_pretrained
        # -----------------------------------------------------------

        print("IMTalker: Loading Preprocessor...")
        wav2vec_processor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_path, local_files_only=(wav2vec_path != "facebook/wav2vec2-base-960h"))

        # 6. Initialize Face Alignment
        print("IMTalker: Loading Face Alignment...")
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu', flip_input=False)

        model_bundle = {
            "renderer": renderer,
            "generator": generator,
            "wav2vec": wav2vec_processor,
            "fa": fa,
            "opt": opt
        }
        
        return (model_bundle,)

    def _load_ckpt(self, model, path, prefix="gen."):
        checkpoint = torch.load(path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        clean_state_dict = {k.replace(prefix, ""): v for k, v in state_dict.items() if k.startswith(prefix)}
        model.load_state_dict(clean_state_dict, strict=False)

    def _load_fm_ckpt(self, model, path, device):
        checkpoint = torch.load(path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        if 'model' in state_dict: state_dict = state_dict['model']
        prefix = 'model.'
        clean_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in clean_dict:
                    param.copy_(clean_dict[name].to(device))

class IMTalkerAudioDriven:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_bundle": ("IMTALKER_MODEL",),
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "crop_face": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "IMTalker"

    def generate(self, model_bundle, image, audio, seed, steps, cfg_scale, crop_face):
        renderer = model_bundle["renderer"]
        generator = model_bundle["generator"]
        wav2vec_proc = model_bundle["wav2vec"]
        fa = model_bundle["fa"]
        opt = model_bundle["opt"]
        device = opt.device

        # Image Processing
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        
        if crop_face:
            img_pil = self.crop_face(img_np, fa, opt.input_size)
        else:
            img_pil = Image.fromarray(img_np).resize((opt.input_size, opt.input_size))

        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        s_tensor = transform(img_pil).unsqueeze(0).to(device)

        # Audio Processing
        waveform = audio["waveform"][0] # [Channels, Samples]
        if waveform.dim() > 1:
             waveform = torch.mean(waveform, dim=0) # Mix to mono
        
        source_sr = audio["sample_rate"]
        target_sr = 16000
        
        if source_sr != target_sr:
            resampler = torchaudio.transforms.Resample(source_sr, target_sr)
            waveform = resampler(waveform)
            
        speech_array = waveform.numpy()
        
        processed_audio = wav2vec_proc(speech_array, sampling_rate=target_sr, return_tensors='pt').input_values[0]
        a_tensor = processed_audio.unsqueeze(0).to(device)

        data = {'s': s_tensor, 'a': a_tensor, 'pose': None, 'cam': None, 'gaze': None, 'ref_x': None}
        
        with torch.no_grad():
            f_r, g_r = renderer.dense_feature_encoder(s_tensor)
            t_lat = renderer.latent_token_encoder(s_tensor)
            if isinstance(t_lat, tuple): t_lat = t_lat[0]
            data['ref_x'] = t_lat
            
            # Pass seed to Config
            opt.seed = seed
            opt.fix_noise_seed = True 
            
            torch.manual_seed(seed)
            # Generator Inference
            sample = generator.sample(data, a_cfg_scale=cfg_scale, nfe=steps, seed=seed)
            
            d_hat = []
            T_frames = sample.shape[1]
            ta_r = renderer.adapt(t_lat, g_r)
            m_r = renderer.latent_token_decoder(ta_r)
            
            for t in range(T_frames):
                ta_c = renderer.adapt(sample[:, t, ...], g_r)
                m_c = renderer.latent_token_decoder(ta_c)
                out_frame = renderer.decode(m_c, m_r, f_r)
                d_hat.append(out_frame)
            
            # vid_tensor shape is [T, 3, H, W] due to stack dim 1 (from list of [1,3,H,W]) and squeeze 0
            vid_tensor = torch.stack(d_hat, dim=1).squeeze(0)
            
        # PERMUTATION: [T, 3, H, W] -> [T, H, W, 3]
        vid_tensor = vid_tensor.permute(0, 2, 3, 1).cpu() 


        if vid_tensor.min() < -0.2:
            vid_tensor = (vid_tensor + 1) / 2

        vid_tensor = torch.clamp(vid_tensor, 0, 1)
        
        return (vid_tensor,)

    def crop_face(self, img_arr, fa, input_size):
        h, w = img_arr.shape[:2]
        try:
            bboxes = fa.face_detector.detect_from_image(img_arr)
        except Exception as e:
            print(f"Face detection failed: {e}")
            bboxes = None
            
        valid_bboxes = []
        if bboxes is not None:
            valid_bboxes = [(int(x1), int(y1), int(x2), int(y2), score) for (x1, y1, x2, y2, score) in bboxes if score > 0.5]
            
        if not valid_bboxes:
            print("Warning: No face detected. Using center crop.")
            cx, cy = w // 2, h // 2
            half = min(w, h) // 2
            x1_new, x2_new = cx - half, cx + half
            y1_new, y2_new = cy - half, cy + half
        else:
            x1, y1, x2, y2, _ = valid_bboxes[0]
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            w_face = x2 - x1
            h_face = y2 - y1
            half_side = int(max(w_face, h_face) * 0.8)
            x1_new = cx - half_side
            y1_new = cy - half_side
            x2_new = cx + half_side
            y2_new = cy + half_side
            
            pad_x1 = max(0, -x1_new)
            pad_y1 = max(0, -y1_new)
            pad_x2 = max(0, x2_new - w)
            pad_y2 = max(0, y2_new - h)
            
            if pad_x1 > 0 or pad_y1 > 0 or pad_x2 > 0 or pad_y2 > 0:
                img_arr = np.pad(img_arr, ((pad_y1, pad_y2), (pad_x1, pad_x2), (0, 0)), mode='constant')
                x1_new += pad_x1
                x2_new += pad_x1
                y1_new += pad_y1
                y2_new += pad_y1

        crop_img = img_arr[int(y1_new):int(y2_new), int(x1_new):int(x2_new)]
        crop_pil = Image.fromarray(crop_img)
        return crop_pil.resize((input_size, input_size))


class IMTalkerVideoDriven:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_bundle": ("IMTALKER_MODEL",),
                "source_image": ("IMAGE",), 
                "driving_images": ("IMAGE",),
                "crop_face": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "IMTalker"

    def generate(self, model_bundle, source_image, driving_images, crop_face):
        renderer = model_bundle["renderer"]
        fa = model_bundle["fa"]
        opt = model_bundle["opt"]
        device = opt.device
        
        transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
        
        src_np = (source_image[0].cpu().numpy() * 255).astype(np.uint8)
        
        def process_one_image(img_arr, crop):
            if crop:
                h, w = img_arr.shape[:2]
                bboxes = fa.face_detector.detect_from_image(img_arr)
                if bboxes is not None and len(bboxes) > 0:
                     box = max(bboxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
                     x1, y1, x2, y2, _ = box
                     cx, cy = (x1+x2)//2, (y1+y2)//2
                     size = int(max(x2-x1, y2-y1) * 1.6)
                     x1n = max(0, int(cx - size//2))
                     y1n = max(0, int(cy - size//2))
                     x2n = min(w, int(cx + size//2))
                     y2n = min(h, int(cy + size//2))
                     img_arr = img_arr[y1n:y2n, x1n:x2n]
            return Image.fromarray(img_arr).resize((opt.input_size, opt.input_size))

        src_pil = process_one_image(src_np, crop_face)
        s_tensor = transform(src_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            f_r, i_r = renderer.app_encode(s_tensor)
            t_r = renderer.mot_encode(s_tensor)
            ta_r = renderer.adapt(t_r, i_r)
            ma_r = renderer.mot_decode(ta_r)
            
            vid_results = []
            
            for i in range(driving_images.shape[0]):
                frame_np = (driving_images[i].cpu().numpy() * 255).astype(np.uint8)
                frame_pil = process_one_image(frame_np, crop_face)
                
                d_tensor = transform(frame_pil).unsqueeze(0).to(device)
                
                t_c = renderer.mot_encode(d_tensor)
                ta_c = renderer.adapt(t_c, i_r)
                ma_c = renderer.mot_decode(ta_c)
                out = renderer.decode(ma_c, ma_r, f_r)
                
                vid_results.append(out.cpu())

            vid_tensor = torch.cat(vid_results, dim=0)
            
        vid_tensor = vid_tensor.permute(0, 2, 3, 1)

        if vid_tensor.min() < -0.2:
            vid_tensor = (vid_tensor + 1) / 2

        vid_tensor = torch.clamp(vid_tensor, 0, 1)
        
        return (vid_tensor,)

NODE_CLASS_MAPPINGS = {
    "IMTalkerLoader": IMTalkerLoader,
    "IMTalkerAudioDriven": IMTalkerAudioDriven,
    "IMTalkerVideoDriven": IMTalkerVideoDriven
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IMTalkerLoader": "IMTalker Model Loader",
    "IMTalkerAudioDriven": "IMTalker Audio Driven",
    "IMTalkerVideoDriven": "IMTalker Video Driven"
}
