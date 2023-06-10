CUDA_VISIBLE_DEVICES=3 \
python main/test_Celeb-DF.py \
--image_size 299 \
--batchsize 128 \
--model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C23/C23_XCP_landmark_mask_5-11-30frames-v3-lr/models/best_model_acc.pth

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python main/test_Celeb-DF.py \
# --image_size 299 \
# --batchsize 256 \
# --model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C23/C23_XCP_-30frames_facesv3/models/best_model_acc.pth

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python main/test_Celeb-DF.py \
# --image_size 380 \
# --batchsize 32 \
# --model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C40/C23_mask_2_5-20frames-test/models/best_model_eer.pth

# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# python main/test_Celeb-DF.py \
# --image_size 380 \
# --batchsize 32 \
# --model_path  /mnt/lvdisk1/miaodata/ff++_code/output/C40/C23_mask_2_5-20frames-test/models/best_model_logloss.pth