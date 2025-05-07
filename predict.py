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

        # Define model information - paths and URLs
        models_to_download = {
            "ComfyUI/models/unet/getphat_v6.safetensors":
                "https://huggingface.co/NSFW-API/GetPhat/resolve/main/getphat_v6.safetensors",
            "ComfyUI/models/vae/ae.safetensors":
                "https://huggingface.co/ffxvs/vae-flux/resolve/main/ae.safetensors",
            "ComfyUI/models/text_encoders/t5xxl_fp16.safetensors":
                "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors",
            "ComfyUI/models/text_encoders/clip_l.safetensors":
                "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
        }

        # Download all required models
        for model_path, model_url in models_to_download.items():
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            if not os.path.exists(model_path):
                print(f"Downloading model to {model_path}...")
                try:
                    subprocess.check_call(
                        ["wget", "--no-verbose", model_url, "-O", model_path]
                    )
                    print(f"Downloaded model to {model_path}")
                except Exception as e:
                    print(f"Error downloading model: {e}")
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
            steps: int = Input(default=35, ge=1, le=150),
            cfg: float = Input(default=4.0, ge=1.0, le=20.0),
            sampler_name: str = Input(default="dpmpp_2m",
                                      choices=["dpmpp_2m", "euler", "euler_ancestral", "heun", "dpmpp_2s_ancestral"]),
            scheduler: str = Input(default="sgm_uniform", choices=["sgm_uniform", "normal", "karras", "exponential"]),
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
        # Update prompts in CLIP Text Encode nodes
        clip_encode_inputs = node(4)["inputs"]
        clip_encode_inputs["clip_l"] = prompt
        clip_encode_inputs["t5xxl"] = prompt

        neg_clip_encode_inputs = node(5)["inputs"]
        neg_clip_encode_inputs["clip_l"] = negative_prompt
        neg_clip_encode_inputs["t5xxl"] = negative_prompt

        # ----- latent size --------------------------------------------------
        latent_inputs = node(6)["inputs"]
        latent_inputs["width"] = self._nearest_multiple(width)
        latent_inputs["height"] = self._nearest_multiple(height)
        latent_inputs["batch_size"] = 1

        # ----- sampler settings --------------------------------------------
        # RandomNoise
        noise_inputs = node(12)["inputs"]
        noise_inputs["seed"] = seed

        # KSamplerSelect
        sampler_select_inputs = node(13)["inputs"]
        sampler_select_inputs["sampler_name"] = sampler_name

        # BasicScheduler
        scheduler_inputs = node(14)["inputs"]
        scheduler_inputs["steps"] = steps
        scheduler_inputs["schedule"] = scheduler

        # FluxGuidance
        try:
            node(15)["widgets"]["guidance"] = cfg
        except KeyError:
            # If there's no widgets field, create it
            node(15)["widgets"] = {"guidance": cfg}

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
