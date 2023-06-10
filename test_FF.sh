CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main/test_FF.py \
--cfg configs/deeper_xcp.yaml \
--model_path /mnt/lvdisk1/miaodata/ff++_code/output/Deeper/deeper_xcp-30frames/models/best_model_acc.pth
