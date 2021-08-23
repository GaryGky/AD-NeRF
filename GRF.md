GRF遇到的问题
- intrinsic是相机的内参，论文中写明这是一个必传参数，需要想想怎么计算。
intrinsic = np.array([[focal, 0., W / 2],[0, focal, H / 2],[0, 0, 1.]]) # 使用focal计算出来

- GRF使用了一个在NeRF之前还用了一个MLP没找到啊，把2D映射到3D加上p点的位置信息，通过MLP对每个点输出一个向量，然后使用Attention将这个点与特征进行对应。

和学长沟通的问题：
1. 大致讲下baseline的设计，以及同步一下最新的进度
2. CNN的问题，能不能将loss前置
3. 博士和硕士的申请哪个比较容易