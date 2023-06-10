CUDA_VISIBLE_DEVICES=2  \
python main/test_ffvideo.py \
--dataset_root /mnt/lvdisk1/miaodata/DeeperForensics/ \
--test_dir test_0.txt \
--model_type xcp \
--test_frames 110 \
--model_path  /mnt/lvdisk1/miaodata/ff++_code/output/Deeper/deeper_xcp-30frames/models/best_model_acc.pth

CUDA_VISIBLE_DEVICES=2  \
python main/test_ffvideo.py \
--dataset_root /mnt/lvdisk1/miaodata/DeeperForensics/ \
--test_dir test_random.txt \
--model_type xcp \
--test_frames 110 \
--model_path  /mnt/lvdisk1/miaodata/ff++_code/output/Deeper/deeper_xcp-30frames/models/best_model_acc.pth

CUDA_VISIBLE_DEVICES=2  \
python main/test_ffvideo.py \
--dataset_root /mnt/lvdisk1/miaodata/DeeperForensics/ \
--test_dir test_level5.txt \
--model_type xcp \
--test_frames 110 \
--model_path  /mnt/lvdisk1/miaodata/ff++_code/output/Deeper/deeper_xcp-30frames/models/best_model_acc.pth

CUDA_VISIBLE_DEVICES=1 \
python main/test_ffvideo.py \
--dataset_root /mnt/lvdisk1/miaodata/DeeperForensics/ \
--test_dir test_mix3.txt \
--model_type xcp \
--test_frames 110 \
--model_path  /mnt/lvdisk1/miaodata/ff++_code/output/Deeper/deeper_xcp-30frames/models/best_model_acc.pth


# CUDA_VISIBLE_DEVICES=1  \
# python main/test_ffvideo.py \
# --dataset_root /mnt/lvdisk1/miaodata/DeeperForensics/ \
# --test_dir test_mix3.txt \
# --model_type xcp \
# --test_frames 110 \
# --model_path  /mnt/lvdisk1/miaodata/ff++_code/output/Deeper/deeper_xcp-30frames/models/best_model_eer.pth


# CUDA_VISIBLE_DEVICES=1  \
# python main/test_ffvideo.py \
# --dataset_root /mnt/lvdisk1/miaodata/DeeperForensics/ \
# --test_dir test_mix3.txt \
# --model_type xcp \
# --test_frames 110 \
# --model_path  /mnt/lvdisk1/miaodata/ff++_code/output/Deeper/deeper_xcp-30frames/models/best_model_logloss.pth