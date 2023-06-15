"""@xvdp
rewrite of würstchen-stage-C.ipynb
"""
from typing import Optional, Tuple, List, Union
import time
import os
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import transformers
from transformers.utils import is_torch_bf16_gpu_available as _bf16_gpu
from transformers.utils import is_torch_bf16_cpu_available as _bf16_cpu
from transformers import AutoTokenizer, CLIPTextModel

from vqgan import VQModel
from modules import Paella, EfficientNetEncoder, Prior
from diffuzz import Diffuzz

transformers.utils.logging.set_verbosity_error()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# pylint: disable=no-member
class Wurst:
    """
    wrap wuerstchen inference in class
    kwargs
        device      (str) [cuda:0 if available]
        cache_dir   (str) ['~/.cache'] to prevent redownload with docker and share volumes
        weights_dir (str) ['models'] defaults to local dir  
    """
    def __init__(self, **kwargs):
        self.set_cache_paths(kwargs.get('cache_dir', None))
        self.set_wurst_model_path(kwargs.get('weights_dir', 'models'))

        device = kwargs.get('device', 'cuda:0')
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        device = self.device.type

        self.float16 = torch.float16
        if (device == 'cuda' and _bf16_gpu()) or (device == 'cpu' and _bf16_cpu()):
            self.float16  = torch.bfloat16

        self.diffuzz = None

        self.generator = None
        self.model = None
        self.clip_model = None
        self.vqmodel = None
        self.clip_tokenizer = None
        self.effnet = None
        self.help

    @property
    def help(self):
        print('usage:\n\tdownload models from  https://huggingface.co/dome272/wuerstchen and place under "models/"')
        print('    W = Wurst([cache_dir="~/.cache", weights_dir="./models", device="cuda:0"])')
        print('    W.load_models()\n    W.infer(caption=<>, negative_caption=<>, batch_size=<>)')
        print('to cache torch and huggingface models other than "~/.cache" init with Wurst(cache_dir=)')

    def load_models(self) -> None:
        """
        loads first to cpu to avoid cuda peaks
        out -> self.
                    diffuzz:        Difuzz
                    model:          Prior               # 4215MB
                    effnet:         EfficientNetEncoder
                    generator:      Paella
                    clip_model:     CLIP-ViT
                    clip_tokenizer: AutoTokenizer
                    vqmodel:        VQModel
        """
        print("loading Difuzz")
        self.diffuzz = Diffuzz(device=self.device)

        _stage_c = osp.join(self.weights,"model_stage_c_ema.pt")
        print(f"loading Prior from {_stage_c}")
        self.model = Prior(c_in=16, c=1536, c_cond=1024, c_r=64, depth=32, nhead=24) # 993,243,680
        self.model.load_state_dict(torch.load(_stage_c)['ema_state_dict'])
        self.model.eval().requires_grad_(False).to(device=self.device)

        _stage_b = torch.load(osp.join(self.weights, "model_stage_b.pt"))
        print(f"loading EfficientNetEncoder from {_stage_b}")
        self.effnet = EfficientNetEncoder(effnet="efficientnet_v2_l")   # 117,254,784
        self.effnet.load_state_dict(_stage_b['effnet_state_dict'])
        self.effnet.eval().requires_grad_(False).to(device=self.device)

        print(f"loading Paella from {_stage_b}")
        self.generator = Paella(byt5_embd=1024)                         # 730,814,336
        self.generator.load_state_dict(_stage_b['state_dict'])
        self.generator.eval().requires_grad_(False).to(device=self.device)

        _clipvit = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        print(f"loading CLIPTextModel from {_clipvit}")
        self.clip_model = CLIPTextModel.from_pretrained(_clipvit)   # 352,984,064
        self.clip_model.eval().requires_grad_(False).to(device=self.device)
        self.clip_tokenizer = AutoTokenizer.from_pretrained(_clipvit)

        _vq = osp.join(self.weights,"vqgan_f4_v1_500k.pt")
        print(f"loading VQModel from {_vq}")
        self.vqmodel = VQModel()                            # 18,406,894
        self.vqmodel.load_state_dict(torch.load(_vq)["state_dict"])
        self.vqmodel.eval().requires_grad_(False).to(device=self.device)


    def unload(self) -> None:
        """ delete cuda cache"""
        self.generator = None
        self.model = None
        self.clip_model = None
        self.vqmodel = None
        self.clip_tokenizer = None
        self.effnet = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


    @staticmethod
    def set_cache_paths(cache_dir: Optional[str] = None) -> None:
        """ from docker point to shared volume to prevent repeated downloads
        """
        if cache_dir is not None:
            cache_dir = osp.abspath(osp.expanduser(cache_dir))
            os.environ['XDG_CACHE_HOME'] = cache_dir
            os.environ['TORCH_HOME'] = osp.join(cache_dir, 'torch')
            os.environ['HUGGINGFACE_HOME'] = osp.join(cache_dir, 'huggingface')


    def set_wurst_model_path(self, path: str = 'models') -> None:
        """  this ought to be replaced by an automatic download to huggingface_home
        """
        self.weights = osp.abspath(osp.expanduser(path))
        _msg = "git clone from https://huggingface.co/dome272/wuerstchen with lfs"
        assert osp.isdir(self.weights), f"weights '{self.weights}' not found, create and {_msg}"
        _models = ['vqgan_f4_v1_500k.pt', 'model_stage_b.pt', 'model_stage_c_ema.pt']
        assert all([osp.isfile(osp.join(self.weights, m)) for m in _models]), \
            f"missing models {_msg}"


    def infer(self,
              caption: str,
              negative_caption: str = "low resolution, low detail, bad quality, blurry",
              batch_size: int = 4,
              prior_cfg: int = 6,
              prior_timesteps: int = 60,
              prior_sampler: str = "ddpm",
              seed: Optional[str] = None,
              show: bool = True) -> Tensor:
        """ runs inference on caption
        """

        clip_text_embed, clip_text_embed_uncond = self.embed_clip(caption, negative_caption,
                                                                  batch_size)

        effnet_features_shape = (batch_size, 16, 12, 12)
        effnet_embeddings_uncond = torch.zeros(effnet_features_shape).to(self.device)
        generator_latent_shape = (batch_size, 128, 128)
        if seed is not None:
            torch.manual_seed(seed)

        with torch.cuda.amp.autocast(dtype=self.float16), torch.no_grad():
            s = time.time()
            sampled = self.diffuzz.sample(self.model, {'c': clip_text_embed},
                                          unconditional_inputs={"c": clip_text_embed_uncond},
                                          shape=effnet_features_shape, timesteps=prior_timesteps,
                                          cfg=prior_cfg, sampler=prior_sampler, t_start=1.0)[-1]
            print(f"Prior Sampling:    {time.time() - s:.3f} s")
            temperature, cfg, steps =(1.0, 0.6), (2.0, 2.0), 8
            s = time.time()
            orig, inter = self.sample(self.generator,
                                      model_inputs={'effnet': sampled,'byt5': clip_text_embed},
                                      latent_shape=generator_latent_shape,
                                      unconditional_inputs = {'effnet': effnet_embeddings_uncond,
                                                              'byt5': clip_text_embed_uncond},
                                      temperature=temperature, cfg=cfg, steps=steps
            )
            print(f"Generator Sampling: {time.time() - s:.3f} s")

        s = time.time()
        sampled = torch.clamp(self.decode(orig), 0, 1).cpu()
        print(f"Decoder Generation: {time.time() - s:.3f} s")
        # inter = [decode(i) for i in inter]
        print(f"Temperature: {temperature}, CFG: {cfg}, Steps: {steps}")

        if show:
            showimages(sampled, title=caption)
        return sampled


    def sample(self,
               model: nn.Module,
               model_inputs: dict,
               latent_shape: tuple,
               unconditional_inputs: Optional[dict] = None,
               init_x: Optional[Tensor] = None,
               steps: int = 12,
               renoise_steps: Optional[int] = None,
               temperature: Tuple[float] = (0.7, 0.3),
               cfg: Tuple[float] = (8.0, 8.0),
               mode: str = 'multinomial', # 'quant', 'multinomial', 'argmax'
               t_start: int = 1.0,
               t_end: int = 0.0,
               sampling_conditional_steps: Optional[int] = None,
               sampling_quant_steps: Optional[int] = None
               ) -> Tuple[Tensor, List[Tensor]]:
        """
        Args
            model       (nn.Module) generator model
            mode        (str ['multinomial']), # 'quant', 'multinomial', 'argmax'
        """
        assert mode in ('quant', 'multinomial', 'argmax'), \
            f"mode='{mode}' invalid: use 'quant' | 'multinomial' | 'argmax'"

        with torch.no_grad():
            device = unconditional_inputs["byt5"].device

            sampling_conditional_steps = sampling_conditional_steps or steps
            sampling_quant_steps = sampling_quant_steps or steps
            renoise_steps = renoise_steps or steps-1
            if unconditional_inputs is None:
                unconditional_inputs = {k: torch.zeros_like(v) for k, v in model_inputs.items()}
            intermediate_images = []


            init_noise = torch.randint(0, model.num_labels, size=latent_shape, device=device)
            sampled = init_x or init_noise.clone()

            t_list = torch.linspace(t_start, t_end, steps+1)
            temperatures = torch.linspace(temperature[0], temperature[1], steps)
            cfgs = torch.linspace(cfg[0], cfg[1], steps)
            if cfg is not None:
                model_inputs = {k:torch.cat([v, v_u]) for
                                (k, v), (k_u, v_u) in zip(model_inputs.items(),
                                                          unconditional_inputs.items())}
            for i, tv in enumerate(t_list[:steps]):
                if i >= sampling_quant_steps:
                    mode = "quant"
                t = torch.ones(latent_shape[0], device=device) * tv

                if cfg is not None and i < sampling_conditional_steps:
                    logits, uncond_logits = model(torch.cat([sampled]*2), torch.cat([t]*2),
                                                  **model_inputs).chunk(2)
                    logits = logits * cfgs[i] + uncond_logits * (1-cfgs[i])
                else:
                    logits = model(sampled, t, **model_inputs)

                scores = logits.div(temperatures[i]).softmax(dim=1)

                if mode == 'argmax':
                    sampled = logits.argmax(dim=1)
                elif mode == 'multinomial':
                    sampled = scores.permute(0,2,3,1).reshape(-1, logits.size(1))
                    sampled = torch.multinomial(sampled, 1)[:, 0].view(logits.size(0),
                                                                       *logits.shape[2:])
                elif mode == 'quant':
                    sampled = scores.permute(0,2,3,1) @ self.vqmodel.vquantizer.codebook.weight.data
                    sampled = self.vqmodel.vquantizer.forward(sampled, dim=-1)[-1]

                intermediate_images.append(sampled)

                if i < renoise_steps:
                    t_next = torch.ones(latent_shape[0], device=device) * t_list[i+1]
                    sampled = model.add_noise(sampled, t_next, random_x=init_noise)[0]
                    intermediate_images.append(sampled)
        return sampled, intermediate_images

    def encode(self, x):
        return self.vqmodel.encode(x)[2]

    def decode(self, img_seq):
        return self.vqmodel.decode_indices(img_seq)

    def embed_clip(self,
                   caption: str,
                   negative_caption: str = "",
                   batch_size: int = 4) -> Tuple[Tensor, Tensor]:
        """tokenize"""
        _len = self.clip_tokenizer.model_max_length
        clip_tokens = self.clip_tokenizer([caption] * batch_size, truncation=True,
                                          padding="max_length", max_length=_len,
                                          return_tensors="pt").to(self.device)
        clip_text_embeddings = self.clip_model(**clip_tokens).last_hidden_state

        clip_tokens_uncond = self.clip_tokenizer([negative_caption] * batch_size, truncation=True,
                                                 padding="max_length", max_length=_len,
                                                 return_tensors="pt").to(self.device)
        clip_text_embeddings_uncond = self.clip_model(**clip_tokens_uncond).last_hidden_state
        return clip_text_embeddings, clip_text_embeddings_uncond


def showimages(imgs: list, rows: bool = False, **kwargs) -> None:
    plt.figure(figsize=(kwargs.get("width", 32), kwargs.get("height", 32)))
    plt.axis("off")
    if 'title' in kwargs:
        plt.title(kwargs['title'])
    _toimg = lambda x: x.cpu().permute(1, 2, 0).contiguous().numpy()
    if rows:
        plt.imshow(_toimg(torch.cat([torch.cat([i for i in row], dim=-1) for row in imgs], dim=-2)))
    else:
        plt.imshow(_toimg(torch.cat([torch.cat([i for i in imgs], dim=-1)], dim=-2)))
    plt.show()



class ImageDataset(Dataset):
    """simple image dataset"""
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.image_list = os.listdir(folder_path)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.folder_path, image_name)
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, image_name

def get_dataset(path: str,
                resize: Union[Tuple[int,int], int, None] = None) -> Dataset:
    """simple image dataset instance"""
    transform = [transforms.ToTensor()]
    if resize is not None:
        transform = [transforms.Resize(resize)] + transform
    transform = transforms.Compose(transform)
    
    return ImageDataset(path, transform=transform)

# Create a data loader to iterate through the dataset
# batch_size = 4  # Define your desired batch size
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)