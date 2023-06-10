CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main/test_ffvideo.py \
--dataset_root /mnt/lvdisk1/miaodata/DFDC/preview/ \
--test_dir test_preview.txt \
--model_type eff \
--test_frames 20 \
--model_path  /mnt/lvdisk1/miaodata/ff++_code/output/preview/preview_landmark_mask_2_5-20frames-075/models/best_model_acc.pth


CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main/test_ffvideo.py \
--dataset_root /mnt/lvdisk1/miaodata/DFDC/preview/ \
--test_dir test_preview.txt \
--model_type eff \
--test_frames 20 \
--model_path  /mnt/lvdisk1/miaodata/ff++_code/output/preview/preview_landmark_mask_2_5-20frames-075/models/best_model_auc.pth


CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main/test_ffvideo.py \
--dataset_root /mnt/lvdisk1/miaodata/DFDC/preview/ \
--test_dir test_preview.txt \
--model_type eff \
--test_frames 20 \
--model_path  /mnt/lvdisk1/miaodata/ff++_code/output/preview/preview_landmark_mask_2_5-20frames-075/models/best_model_eer.pth


CUDA_VISIBLE_DEVICES=4,5,6,7 \
python main/test_ffvideo.py \
--dataset_root /mnt/lvdisk1/miaodata/DFDC/preview/ \
--test_dir test_preview.txt \
--model_type eff \
--test_frames 20 \
--model_path  /mnt/lvdisk1/miaodata/ff++_code/output/preview/preview_landmark_mask_2_5-20frames-075/models/best_model_logloss.pth