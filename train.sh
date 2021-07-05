# recommended paddle.__version__ == 2.0.0
#/home/python3/python -m paddle.distributed.launch --log_dir=./debug/ --gpus='0'  tools/train.py -c configs/rec/rec_mv3_tps_bilstm_att.yml
/home/python3/python  tools/train.py  -c configs/rec/ch_ppocr_v2.0/rec_chinese_lite_train_v2.0_lmdb.yml
#/home/python3/python  tools/train.py  -c configs/rec/rec_mv3_tps_bilstm_att.yml