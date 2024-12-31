import os
import subprocess
import sys
from datetime import datetime
import torch.multiprocessing as mp

def set_environment():
    """Set environment variables"""
    os.environ["WANDB_MODE"] = "offline"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
    os.environ["FINETRAINERS_LOG_LEVEL"] = "DEBUG"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Set multiprocessing method to spawn for Windows
    mp.set_start_method('spawn', force=True)

def build_command():
    """Build the training command"""
    # Configuration
    data_root = "./video-dataset-disney"
    caption_column = "prompt.txt"
    video_column = "videos.txt"
    output_dir = "./output/ltx-video/ltxv_disney"

    # Command components
    base_cmd = [
        "accelerate", 
        "launch",
        "--config_file", "accelerate_config_windows.yaml",
        "train.py"
    ]
    #        "--enable_model_cpu_offload",
    #        "--pretrained_model_name_or_path", "Lightricks/LTX-Video",
    #        "--pretrained_model_name_or_path", "E:\ComfyUI_windows_portable\ComfyUI\models\checkpoints",t5xxl_fp16.safetensors, t5xxl_fp8_e4m3fn.safetensors
    args = [
        "--model_name", "ltx_video",
        "--pretrained_model_name_or_path", "a-r-r-o-w/LTX-Video-0.9.1-diffusers",
        "--data_root", data_root,
        "--video_column", video_column,
        "--caption_column", caption_column,
        "--id_token", "BW_STYLE",
        "--video_resolution_buckets", "49x512x768",
        "--caption_dropout_p", "0.05",
        "--dataloader_num_workers", "1",  # Changed from 0 to 1 for Windows
        "--flow_resolution_shifting",
        "--training_type", "lora",
        "--seed", "42",
        "--mixed_precision", "bf16",
        "--batch_size", "1",
        "--train_steps", "10",
        "--rank", "128",
        "--lora_alpha", "128",
        "--target_modules", "to_q", "to_k", "to_v", "to_out.0", "ff.net.0.proj", "ff.net.2",
        "--gradient_accumulation_steps", "1",
        "--gradient_checkpointing",
        "--checkpointing_steps", "500",
        "--checkpointing_limit", "2",
        "--enable_slicing",
        "--enable_tiling",
        "--optimizer", "adamw",
        "--lr", "3e-5",
        "--lr_scheduler", "constant_with_warmup",
        "--lr_warmup_steps", "100",
        "--lr_num_cycles", "1",
        "--beta1", "0.9",
        "--beta2", "0.95",
        "--weight_decay", "1e-4",
        "--epsilon", "1e-8",
        "--max_grad_norm", "1.0",
        "--validation_prompts", """afkx A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions.@@@49x512x768:::A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage@@@49x512x768""",
        "--num_validation_videos", "1",
        "--validation_steps", "999999999999999999999",
        "--tracker_name", "finetrainers-ltxv",
        "--output_dir", output_dir,
        "--nccl_timeout", "1800",
        "--report_to", "wandb",
        "--precompute_conditions",
        "--gradient_checkpointing"
    ]
    
    return base_cmd + args

def main():
    """Main execution function"""
    print("Starting training script...")
    
    # Set environment variables
    try:
        set_environment()
    except RuntimeError as e:
        print(f"Warning: {e}")
        print("Continuing with default multiprocessing settings...")
    
    # Build command
    cmd = build_command()
    print("Running command:", " ".join(cmd))
    
    try:
        # Create and run the process with real-time output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Read and print output in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                sys.stdout.flush()

        # Get the return code
        return_code = process.poll()
        
        if return_code == 0:
            print("\n-------------------- Finished executing script --------------------")
        else:
            print(f"\nProcess failed with exit code {return_code}")
            sys.exit(return_code)
            
    except Exception as e:
        print(f"Error executing command: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()