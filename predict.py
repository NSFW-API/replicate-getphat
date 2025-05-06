import json
import os
import subprocess
import time
from typing import List

from cog import BasePredictor, Input, Path

from comfyui import ComfyUI  # pip-installed via cog.yaml

# ---- constants -------------------------------------------------------------
WORKFLOW_JSON = "getphat_flux_workflow.json"
OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
ALL_DIRS = [OUTPUT_DIR, INPUT_DIR]


# ----------------------------------------------------------------------------

class Predictor(BasePredictor):
    def setup(self):
        # Create necessary directories
        for dir_path in ALL_DIRS:
            os.makedirs(dir_path, exist_ok=True)

        # Download the model checkpoint
        models_dir = "ComfyUI/models/checkpoints"
        os.makedirs(models_dir, exist_ok=True)

        ckpt_path = os.path.join(models_dir, "getphat_v6.safetensors")
        if not os.path.exists(ckpt_path):
            print(f"Downloading checkpoint to {ckpt_path}...")
            try:
                diff_dir = "ComfyUI/models/diffusion_models"  # where FluxMod expects it
                helper_ckpt_dir = "checkpoints"  # where handle_weights looks
                ckpt_file = "getphat_v6.safetensors"

                os.makedirs(diff_dir, exist_ok=True)
                os.makedirs(helper_ckpt_dir, exist_ok=True)

                ckpt_path = os.path.join(diff_dir, ckpt_file)

                # 1. Download if missing
                if not os.path.exists(ckpt_path):
                    subprocess.check_call([
                        "wget", "--no-verbose",
                        "https://huggingface.co/NSFW-API/GetPhat/resolve/main/getphat_v6.safetensors",
                        "-O", ckpt_path
                    ])

                # 2. Symlink (or copy) into ./checkpoints so handle_weights() won't redownload
                link_path = os.path.join(helper_ckpt_dir, ckpt_file)
                if not os.path.exists(link_path):
                    try:
                        os.symlink(os.path.abspath(ckpt_path), link_path)
                    except OSError:  # container FS may not allow symlinks
                        import shutil
                        shutil.copy2(ckpt_path, link_path)

                print(f"Downloaded checkpoint to {ckpt_path}")
            except Exception as e:
                print(f"Error downloading checkpoint: {e}")
                raise

        # Ensure VAE exists
        vae_dir = "ComfyUI/models/vae"
        os.makedirs(vae_dir, exist_ok=True)
        vae_path = os.path.join(vae_dir, "ae.safetensors")
        if not os.path.exists(vae_path):
            print(f"Downloading VAE to {vae_path}...")
            try:
                VAE_URL = "https://huggingface.co/ffxvs/vae-flux/resolve/main/ae.safetensors"
                subprocess.check_call(
                    ["wget", "--no-verbose", VAE_URL, "-O", vae_path]
                )
                print(f"Downloaded VAE to {vae_path}")
            except Exception as e:
                print(f"Error downloading VAE: {e}")
                raise

        # Ensure T5-XXL CLIP model exists
        clip_dir = "ComfyUI/models/clip"
        os.makedirs(clip_dir, exist_ok=True)
        t5_clip_path = os.path.join(clip_dir, "t5xxl_fp16.safetensors")
        if not os.path.exists(t5_clip_path):
            print(f"Downloading T5-XXL CLIP to {t5_clip_path}...")
            try:
                T5_CLIP_URL = (
                    "https://huggingface.co/comfyanonymous/flux_text_encoders/"
                    "resolve/main/t5xxl_fp16.safetensors"
                )
                subprocess.check_call(
                    ["wget", "--no-verbose", T5_CLIP_URL, "-O", t5_clip_path]
                )
                print(f"Downloaded T5-XXL CLIP to {t5_clip_path}")
            except Exception as e:
                print(f"Error downloading T5-XXL CLIP: {e}")
                raise

        # Download CLIP-L model
        clipl_path = os.path.join(clip_dir, "clip_l.safetensors")
        if not os.path.exists(clipl_path):
            print(f"Downloading CLIP-L to {clipl_path}...")
            try:
                CLIPL_URL = (
                    "https://huggingface.co/comfyanonymous/flux_text_encoders/"
                    "resolve/main/clip_l.safetensors"
                )
                subprocess.check_call(
                    ["wget", "--no-verbose", CLIPL_URL, "-O", clipl_path]
                )
                print(f"Downloaded CLIP-L to {clipl_path}")
            except Exception as e:
                print(f"Error downloading CLIP-L: {e}")
                raise

        # Start ComfyUI server
        print("Starting ComfyUI server...")
        self.comfy = ComfyUI("127.0.0.1:8188")
        self.comfy.start_server(OUTPUT_DIR, INPUT_DIR)

        # Wait for server to be ready
        max_retries = 30
        retry_count = 0
        while retry_count < max_retries:
            try:
                self.comfy.connect()
                print("ComfyUI server is ready")
                break
            except Exception as e:
                print(f"Waiting for ComfyUI server to be ready... {e}")
                time.sleep(1)
                retry_count += 1

        if retry_count == max_retries:
            raise RuntimeError("Failed to connect to ComfyUI server")

    # ---- helpers -----------------------------------------------------------
    def _nearest_multiple(self, x: int, k: int = 8) -> int:
        return ((x + k - 1) // k) * k

    # -----------------------------------------------------------------------

    def predict(
            self,
            prompt: str = Input(description="Main text prompt."),
            negative_prompt: str = Input(default="", description="Optional negative."),
            width: int = Input(default=1024, ge=64, le=1536),
            height: int = Input(default=1024, ge=64, le=1536),
            steps: int = Input(default=25, ge=1, le=150),
            cfg: float = Input(default=5.0, ge=1.0, le=20.0),
            sampler_name: str = Input(default="euler",
                                      choices=["euler", "euler_ancestral", "heun", "dpmpp_2s_ancestral", "uni_pc"]),
            scheduler: str = Input(default="normal", choices=["beta", "normal"]),
            seed: int = Input(default=0, description="0 = random"),
    ) -> List[Path]:

        # 1. housekeeping
        self.comfy.cleanup(ALL_DIRS)
        if seed == 0: seed = int.from_bytes(os.urandom(2), "big")

        # 2. Load workflow.json
        with open(WORKFLOW_JSON) as f:
            wf = json.load(f)

        # 3. Update workflow with user inputs
        if "nodes" in wf:  # Style B
            by_id = {str(n["id"]): n for n in wf["nodes"]}
        else:  # Style A
            by_id = wf

        def node(idx: int):
            """Return the node dict for a given numeric id"""
            return by_id[str(idx)]

        # ----- prompt nodes -------------------------------------------------

        node(4)["inputs"]["text"] = prompt
        node(5)["inputs"]["text"] = negative_prompt

        # ----- latent size --------------------------------------------------
        latent_inputs = node(6)["inputs"]
        latent_inputs["width"] = self._nearest_multiple(width)
        latent_inputs["height"] = self._nearest_multiple(height)
        latent_inputs["batch_size"] = 1

        # ----- sampler settings --------------------------------------------
        sampler_inputs = node(7)["inputs"]
        sampler_inputs["seed"] = seed
        sampler_inputs["steps"] = steps
        sampler_inputs["cfg"] = cfg
        sampler_inputs["sampler_name"] = sampler_name
        sampler_inputs["scheduler"] = scheduler
        sampler_inputs["denoise"] = 1.0

        # 4. Run the workflow
        print("Loading workflow...")
        wf_loaded = self.comfy.load_workflow(wf)
        print("Running workflow...")
        self.comfy.run_workflow(wf_loaded)

        # 5. Get the output images
        print("Getting output files...")
        all_files = self.comfy.get_files(OUTPUT_DIR)
        image_files = [
            p
            for p in all_files
            if p.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]
        return image_files
