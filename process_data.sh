#python data_util/process_data.py --id=$1 --step=0 &
#python data_util/process_data.py --id=$1 --step=1
#python data_util/process_data.py --id=$1 --step=2

#python data_util/process_data.py --id=$1 --step=3
#python data_util/process_data.py --id=$1 --step=4
#python data_util/process_data.py --id=$1 --step=5
#wait
python data_util/process_data.py --id=$1 --step=6
python data_util/process_data.py --id=$1 --step=7


# TODO: step6 需要使用3dmm pytorch-3d，暂时与服务器的gcc版本不兼容，需要升级之后才能执行