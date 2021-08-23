
# 跳到工程目录
cd /mnt/cpfs/users/gpuwork/zheng.zhu/talking-head-code/AD-NeRF

# 安装依赖包
pip install -r requirement.txt

# 运行命令
python ./NeRFs/HeadNeRF/run_nerf.py --config dataset/Obama/HeadNeRF_config.txt

# Torso train
python ./NeRFs/TorsoNeRF/run_nerf.py --config dataset/Obama/TorsoNeRF_config.txt

# Test
python NeRFs/TorsoNeRF/run_nerf.py --config dataset/Obama/TorsoNeRFTest_config.txt --aud_file=slow.npy --test_size=-1