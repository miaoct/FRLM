CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 \
python main/test_fb01.py \
--dataset_root /mnt/lvdisk1/miaodata/DFDC/preview/ \
--test_list /mnt/lvdisk1/miaodata/DFDC/preview/List_of_testing_videos.txt \
--bbox_path /mnt/lvdisk1/miaodata/DFDC/preview/testing_videos_bboxs/ \
--image_size 380 \
--batchsize 24 \
--model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C23/C23_landmark_mask_2_5-20frames-1/models/best_model_acc.pth

CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 \
python main/test_fb01.py \
--dataset_root /mnt/lvdisk1/miaodata/DFDC/preview/ \
--test_list /mnt/lvdisk1/miaodata/DFDC/preview/List_of_testing_videos.txt \
--bbox_path /mnt/lvdisk1/miaodata/DFDC/preview/testing_videos_bboxs/ \
--image_size 380 \
--batchsize 24 \
--model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C23/C23_landmark_mask_2_5-20frames-1/models/best_model_auc.pth

CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 \
python main/test_fb01.py \
--dataset_root /mnt/lvdisk1/miaodata/DFDC/preview/ \
--test_list /mnt/lvdisk1/miaodata/DFDC/preview/List_of_testing_videos.txt \
--bbox_path /mnt/lvdisk1/miaodata/DFDC/preview/testing_videos_bboxs/ \
--image_size 380 \
--batchsize 24 \
--model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C23/C23_landmark_mask_2_5-20frames-1/models/best_model_eer.pth

CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 \
python main/test_fb01.py \
--dataset_root /mnt/lvdisk1/miaodata/DFDC/preview/ \
--test_list /mnt/lvdisk1/miaodata/DFDC/preview/List_of_testing_videos.txt \
--bbox_path /mnt/lvdisk1/miaodata/DFDC/preview/testing_videos_bboxs/ \
--image_size 380 \
--batchsize 24 \
--model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C23/C23_landmark_mask_2_5-20frames-1/models/best_model_logloss.pth