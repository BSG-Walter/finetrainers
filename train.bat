@echo off

REM Set environment variables
SET WANDB_MODE=offline
SET NCCL_P2P_DISABLE=1
SET TORCH_NCCL_ENABLE_MONITORING=0
SET FINETRAINERS_LOG_LEVEL=DEBUG

SET GPU_IDS=0

SET DATA_ROOT=./video-dataset-disney
SET CAPTION_COLUMN=prompt.txt
SET VIDEO_COLUMN=videos.txt
SET OUTPUT_DIR=./output/ltx-video/ltxv_disney

SET ID_TOKEN=BW_STYLE

REM Model arguments
SET model_cmd=--model_name ltx_video ^
  --pretrained_model_name_or_path Lightricks/LTX-Video

REM Dataset arguments
SET dataset_cmd=--data_root %DATA_ROOT% ^
  --video_column %VIDEO_COLUMN% ^
  --caption_column %CAPTION_COLUMN% ^
  --id_token %ID_TOKEN% ^
  --video_resolution_buckets 49x512x768 ^
  --caption_dropout_p 0.05

REM Dataloader arguments
SET dataloader_cmd=--dataloader_num_workers 0

REM Diffusion arguments
SET diffusion_cmd=--flow_resolution_shifting

REM Training arguments
SET training_cmd=--training_type lora ^
  --seed 42 ^
  --mixed_precision bf16 ^
  --batch_size 1 ^
  --train_steps 1200 ^
  --rank 64 ^
  --lora_alpha 64 ^
  --target_modules to_q to_k to_v to_out.0 ff.net.0.proj ff.net.2^
  --gradient_accumulation_steps 1 ^
  --gradient_checkpointing ^
  --checkpointing_steps 500 ^
  --checkpointing_limit 2 ^
  --enable_slicing ^
  --enable_tiling ^
  --precompute_conditions ^
  --gradient_checkpointing 

REM Optimizer arguments
SET optimizer_cmd=--optimizer adamw ^
  --lr 3e-5 ^
  --lr_scheduler constant_with_warmup ^
  --lr_warmup_steps 100 ^
  --lr_num_cycles 1 ^
  --beta1 0.9 ^
  --beta2 0.95 ^
  --weight_decay 1e-4 ^
  --epsilon 1e-8 ^
  --max_grad_norm 1.0

REM Validation arguments
SET validation_cmd=--validation_prompts ^
  "%ID_TOKEN% A black and white animated scene unfolds with an anthropomorphic goat surrounded by musical notes and symbols, suggesting a playful environment. Mickey Mouse appears, leaning forward in curiosity as the goat remains still. The goat then engages with Mickey, who bends down to converse or react. The dynamics shift as Mickey grabs the goat, potentially in surprise or playfulness, amidst a minimalistic background. The scene captures the evolving relationship between the two characters in a whimsical, animated setting, emphasizing their interactions and emotions.@@@49x512x768:::%ID_TOKEN% A woman with long brown hair and light skin smiles at another woman with long blonde hair. The woman with brown hair wears a black jacket and has a small, barely noticeable mole on her right cheek. The camera angle is a close-up, focused on the woman with brown hair's face. The lighting is warm and natural, likely from the setting sun, casting a soft glow on the scene. The scene appears to be real-life footage@@@49x512x768" ^
  --num_validation_videos 1 ^
  --validation_steps 100

REM Miscellaneous arguments
SET miscellaneous_cmd=--tracker_name finetrainers-ltxv ^
  --output_dir %OUTPUT_DIR% ^
  --nccl_timeout 1800 ^
  --report_to wandb

REM Construct the command
SET cmd=accelerate launch --config_file accelerate_configs/uncompiled_1.yaml --gpu_ids %GPU_IDS% train.py ^
  %model_cmd% ^
  %dataset_cmd% ^
  %dataloader_cmd% ^
  %diffusion_cmd% ^
  %training_cmd% ^
  %optimizer_cmd% ^
  %validation_cmd% ^
  %miscellaneous_cmd%

echo Running command: %cmd%
%cmd%
echo.-------------------- Finished executing script --------------------
echo.
pause