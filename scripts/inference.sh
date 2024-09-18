python inference.py --seed 2024\
 -cp ./ckpt/sam_med3d_turbo.pth \
 -tdp ./data/test_data \
 --output_dir ./results  --task_name test_amos_move \
 --sliding_window --save_image_and_gt