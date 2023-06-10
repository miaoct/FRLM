CUDA_VISIBLE_DEVICES=4,5 \
python main/test_Celeb-DF.py \
--dataset_root /mnt/lvdisk1/miaodata/Celeb-DF_V1/ \
--test_list /mnt/lvdisk1/miaodata/Celeb-DF_V1/List_of_testing_videos.txt \
--bbox_path /mnt/lvdisk1/miaodata/Celeb-DF_V1/testing_videos_bboxs/ \
--image_size 299 \
--batchsize 128 \
--model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C23_new/C23_XCP_landmark_mask_5-11-30frames-dr0.5-1/models/best_model_acc.pth

CUDA_VISIBLE_DEVICES=4,5 \
python main/test_Celeb-DF.py \
--dataset_root /mnt/lvdisk1/miaodata/Celeb-DF_V1/ \
--test_list /mnt/lvdisk1/miaodata/Celeb-DF_V1/List_of_testing_videos.txt \
--bbox_path /mnt/lvdisk1/miaodata/Celeb-DF_V1/testing_videos_bboxs/ \
--image_size 299 \
--batchsize 128 \
--model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C23_new/C23_XCP_landmark_mask_5-11-30frames-dr0.9-er0.5-ed8-nr0.5-nd8-mr0.5-md6/models/best_model_acc.pth

CUDA_VISIBLE_DEVICES=4,5 \
python main/test_Celeb-DF.py \
--dataset_root /mnt/lvdisk1/miaodata/Celeb-DF_V1/ \
--test_list /mnt/lvdisk1/miaodata/Celeb-DF_V1/List_of_testing_videos.txt \
--bbox_path /mnt/lvdisk1/miaodata/Celeb-DF_V1/testing_videos_bboxs/ \
--image_size 299 \
--batchsize 128 \
--model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C23_new/C23_XCP_landmark_mask_5-11-30frames-dr0.9-er0.5-ed4-nr0.5-nd4-mr0.5-md3/models/best_model_acc.pth

CUDA_VISIBLE_DEVICES=4,5 \
python main/test_Celeb-DF.py \
--dataset_root /mnt/lvdisk1/miaodata/Celeb-DF_V1/ \
--test_list /mnt/lvdisk1/miaodata/Celeb-DF_V1/List_of_testing_videos.txt \
--bbox_path /mnt/lvdisk1/miaodata/Celeb-DF_V1/testing_videos_bboxs/ \
--image_size 299 \
--batchsize 128 \
--model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C23_new/C23_XCP_landmark_mask_5-11-30frames-dr0.9-er0.7-ed4-nr0.8-nd4-mr0.9-md3/models/best_model_acc.pth

# CUDA_VISIBLE_DEVICES=0,1 \
# python main/test_Celeb-DF.py \
# --dataset_root /mnt/lvdisk1/miaodata/Celeb-DF_V1/ \
# --test_list /mnt/lvdisk1/miaodata/Celeb-DF_V1/List_of_testing_videos.txt \
# --bbox_path /mnt/lvdisk1/miaodata/Celeb-DF_V1/testing_videos_bboxs/ \
# --image_size 299 \
# --batchsize 128 \
# --model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C23_new/C23_XCP_landmark_mask_5-11-30frames-dr0.35/models/best_model_acc.pth

# CUDA_VISIBLE_DEVICES=0,1 \
# python main/test_Celeb-DF.py \
# --dataset_root /mnt/lvdisk1/miaodata/Celeb-DF_V1/ \
# --test_list /mnt/lvdisk1/miaodata/Celeb-DF_V1/List_of_testing_videos.txt \
# --bbox_path /mnt/lvdisk1/miaodata/Celeb-DF_V1/testing_videos_bboxs/ \
# --image_size 299 \
# --batchsize 128 \
# --model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C23_new/C23_XCP_landmark_mask_5-11-30frames-dr0.3/models/best_model_acc.pth

# CUDA_VISIBLE_DEVICES=2 \
# python main/test_Celeb-DF.py \
# --dataset_root /mnt/lvdisk1/miaodata/Celeb-DF_V1/ \
# --test_list /mnt/lvdisk1/miaodata/Celeb-DF_V1/List_of_testing_videos.txt \
# --bbox_path /mnt/lvdisk1/miaodata/Celeb-DF_V1/testing_videos_bboxs/ \
# --image_size 299 \
# --batchsize 128 \
# --model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C23_new/C23_XCP_landmark_mask_5-11-30frames-dr0.65/models/best_model_acc.pth

# CUDA_VISIBLE_DEVICES=2 \
# python main/test_Celeb-DF.py \
# --dataset_root /mnt/lvdisk1/miaodata/Celeb-DF_V1/ \
# --test_list /mnt/lvdisk1/miaodata/Celeb-DF_V1/List_of_testing_videos.txt \
# --bbox_path /mnt/lvdisk1/miaodata/Celeb-DF_V1/testing_videos_bboxs/ \
# --image_size 299 \
# --batchsize 128 \
# --model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C23_new/C23_XCP_landmark_mask_5-11-30frames-dr0.6/models/best_model_acc.pth

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python main/test_Celeb-DF.py \
# --dataset_root /mnt/lvdisk1/miaodata/Celeb-DF_V1/ \
# --test_list /mnt/lvdisk1/miaodata/Celeb-DF_V1/List_of_testing_videos.txt \
# --bbox_path /mnt/lvdisk1/miaodata/Celeb-DF_V1/testing_videos_bboxs/ \
# --image_size 299 \
# --batchsize 256 \
# --model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C23/C23_XCP_-30frames_facesv3/models/best_model_acc.pth

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python main/test_Celeb-DF.py \
# --dataset_root /mnt/lvdisk1/miaodata/Celeb-DF_V1/ \
# --test_list /mnt/lvdisk1/miaodata/Celeb-DF_V1/List_of_testing_videos.txt \
# --bbox_path /mnt/lvdisk1/miaodata/Celeb-DF_V1/testing_videos_bboxs/ \
# --image_size 380 \
# --batchsize 24 \
# --model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C23/C23_landmark_mask_2_5-30frames-08/models/best_model_eer.pth

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python main/test_Celeb-DF.py \
# --dataset_root /mnt/lvdisk1/miaodata/Celeb-DF_V1/ \
# --test_list /mnt/lvdisk1/miaodata/Celeb-DF_V1/List_of_testing_videos.txt \
# --bbox_path /mnt/lvdisk1/miaodata/Celeb-DF_V1/testing_videos_bboxs/ \
# --image_size 380 \
# --batchsize 24 \
# --model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C23/C23_landmark_mask_2_5-30frames-08/models/best_model_logloss.pth