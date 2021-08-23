import logging
import os
import time

import face_alignment
import imageio
import torch.optim
from natsort import natsorted
from tqdm import tqdm, trange

from load_audface import load_audface_data
from models.attsets import AttentionSets
from models.audio_net import AudioNet, AudioAttNet
from models.face_nerf import FaceNeRF
from models.face_unet import FaceUNetCNN
from models.nerf_attention_model import NeRFAttentionModel
from run_nerf_helpers import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

parser = config_parser()
args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}")
np.random.seed(0)

logger = logging.getLogger('adnerf')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)

logger.info(f"device: {device}")


def run_network(inputs, viewdirs, aud_para, fn, embed_fn, embeddirs_fn,
                intrisic, cnn_features, attention_poses, netchunk=1024 * 64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]]).to(device)
    embeded = embed_fn(inputs_flat)
    aud = aud_para.unsqueeze(0).expand(inputs_flat.shape[0], -1)
    # embed直接concat了音频特征
    embeded = torch.cat((embeded, aud), -1)
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embeded_dirs = embeddirs_fn(input_dirs_flat)
        embeded = torch.cat([embeded, embeded_dirs], -1)

    outputs_flat = torch.cat(
        [fn([embeded[i:i + netchunk], gather_indices(inputs_flat[i:i + netchunk]), attention_poses, intrisic,
             cnn_features]) for i in
         range(0, embeded.shape[0], netchunk)], 0)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, bc_rgb, aud_para, chunk=1024 * 32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i + chunk], bc_rgb[i:i + chunk], aud_para, **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


"""
    c2w: pose, 传入了c2w之后，就会在整张图像上render
"""


def render_dynamic_face(H, W, focal, cx, cy, chunk=1024 * 32, rays=None, bc_rgb=None, aud_para=None,
                        c2w=None, ndc=True, near=0., far=1.,
                        use_viewdirs=False, c2w_staticcam=None,
                        **kwargs):
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, focal, c2w, cx, cy)
        bc_rgb = bc_rgb.reshape(-1, 3)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, focal, c2w_staticcam, cx, cy)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

    sh = rays_d.shape  # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1, 3]).float()
    rays_d = torch.reshape(rays_d, [-1, 3]).float()

    near, far = near * \
                torch.ones_like(rays_d[..., :1]), far * \
                torch.ones_like(rays_d[..., :1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, bc_rgb, aud_para, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map', 'last_weight']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, aud_paras, bc_img, hwfcxy,
                chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal, cx, cy = hwfcxy

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []
    last_weights = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        logger.info(f'time: {i, time.time() - t}')
        t = time.time()
        rgb, disp, acc, last_weight, _ = render_dynamic_face(
            H, W, focal, cx, cy, chunk=chunk, c2w=c2w[:3, :4], aud_para=aud_paras[i], bc_rgb=bc_img,
            **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        last_weights.append(last_weight.cpu().numpy())
        if i == 0:
            logger.info(f'rgb_shape: {rgb.shape}, disp_shape: {disp.shape}')

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    last_weights = np.stack(last_weights, 0)

    return rgbs, disps, last_weights


def create_model(args):
    output_ch = 4
    skips = [4]
    start = 0
    basedir = args.basedir
    expname = args.expname

    embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
    embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
    """定义网络结构"""
    # 1. 创建人脸CNN
    attention_embed_func, attention_embed_out_dim = get_embedder(5, 0)  # embed_out_dim = 33
    face_unet = FaceUNetCNN(attention_embed_out_dim)

    # 2. 创建AudioNet
    aud_net = AudioNet(args.dim_aud, args.win_size).to(device)
    aud_att_net = AudioAttNet().to(device)

    # 3. 创建Attention模型 TODO：这里先使用AttSets，以后可以改为SlotAttention
    attention_block = AttentionSets(input_ch=128 + attention_embed_out_dim, attention_output_length=512)

    # 4. 创建NeRF
    face_nerf_coarse = FaceNeRF(D=args.netdepth, W=args.netwidth,
                                input_ch=input_ch, dim_aud=args.dim_aud,
                                output_ch=output_ch, skips=skips,
                                input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)

    # 5. 创建Nerf_Attention模块
    nerf_attention = NeRFAttentionModel(face_nerf_coarse, attention_block, attention_embed_out_dim)
    grad_vars = list(nerf_attention.parameters())

    # 6. 创建Nerf_fine_Attention模块
    _, attention_embed_out_dim_2 = get_embedder(2, 0, 9)
    face_nerf_fine = FaceNeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                              input_ch=input_ch, dim_aud=args.dim_aud,
                              output_ch=output_ch, skips=skips,
                              input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    nerf_fine_attention = NeRFAttentionModel(face_nerf_fine, attention_block, attention_embed_out_dim_2)

    """统一管理 models"""
    models = {'nerf_attention_model': nerf_attention}
    models['face_unet_model'] = face_unet
    models['aud_model'] = aud_net
    models['aud_att_model'] = aud_att_net
    models['nerf_fine_attention_model'] = nerf_fine_attention

    grad_vars += list(nerf_fine_attention.parameters())

    """定义优化器"""
    # UNet优化器用于优化CNN提取的特征
    optimizer_unet = torch.optim.Adam(params=face_unet.parameters(), lr=args.lrate, betas=(0.9, 0.999))
    # nerf的优化器包括coarse和fine两部分
    optimizer_nerf = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    # aud优化器用于优化音频特征的提取
    optimizer_aud = torch.optim.Adam(params=list(aud_net.parameters()), lr=args.lrate, betas=(0.9, 0.999))
    optimizer_aud_att = torch.optim.Adam(params=list(aud_att_net.parameters()), lr=args.lrate, betas=(0.9, 0.999))
    """统一管理优化器"""
    optimizers = {'optimizer_unet': optimizer_unet}
    optimizers['optimizer_nerf'] = optimizer_nerf
    optimizers['optimizer_aud'] = optimizer_aud
    optimizers['optimizer_aud_att'] = optimizer_aud_att

    # Load checkpoints
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in natsorted(
            os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    logger.info(f'Found ckpts:{ckpts}')

    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        logger.info(f'Reloading from: {ckpt_path}')
        ckpt = torch.load(ckpt_path)
        start = ckpt['global_step']

        optimizer_nerf.load_state_dict(ckpt['optimizer_nerf_state_dict'])
        optimizer_aud.load_state_dict(ckpt['optimizer_aud_state_dict'])
        optimizer_aud_att.load_state_dict(ckpt['optimize_audatt_state_dict'])
        optimizer_unet.load_state_dict(ckpt['optimizer_unet_state_dict'])

        aud_net.load_state_dict(ckpt['audnet_state_dict'], strict=False)
        face_nerf_coarse.load_state_dict(ckpt['face_nerf_coarse_state_dict'])
        face_nerf_fine.load_state_dict(ckpt['face_nerf_fine_state_dict'])
        aud_att_net.load_state_dict(ckpt['audattnet_state_dict'], strict=False)
        face_unet.load_state_dict(ckpt['face_unet_state_dict'])

    def network_query_fn(inputs, viewdirs, aud_para, network_fn, intrinsic, cnn_features):
        return run_network(inputs, viewdirs, aud_para, network_fn,
                           intrisic=intrinsic,
                           cnn_features=cnn_features,
                           embed_fn=embed_fn,
                           embeddirs_fn=embeddirs_fn,
                           netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'N_samples': args.N_samples,
        'use_viewdirs': args.use_viewdirs,
        'white_bkgd': args.white_bkgd,
        'raw_noise_std': args.raw_noise_std,
    }

    render_kwargs_test = {k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, models, optimizers


def raw2outputs(raw, z_vals, rays_d, bc_rgb, raw_noise_std=0, white_bkgd=False, pytest=False):
    def raw2alpha(raw, dists, act_fn=F.relu):
        return 1. - torch.exp(-(act_fn(raw) + 1e-6) * dists)

    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape).to(device)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    rgb = torch.cat((rgb[:, :-1, :], bc_rgb.unsqueeze(1)), dim=1)
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
            noise = torch.Tensor(noise).to(device)

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * \
              torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1. - alpha + 1e-10], -1), -1).to(
                  device)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map).to(device), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(ray_batch,
                bc_rgb,
                aud_para,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples).to(near.device)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * (t_vals)
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape).to(z_vals.device)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand).to(lower.device)
        t_rand[..., -1] = 1.0
        z_vals = lower + (upper - lower) * t_rand
        z_vals.to(device)
        viewdirs.to(device)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * \
          z_vals[..., :, None]  # [N_rays, N_samples, 3]
    raw = network_query_fn(pts, viewdirs, aud_para, network_fn)

    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
        raw, z_vals, rays_d, bc_rgb, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * \
              z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
        raw = network_query_fn(pts, viewdirs, aud_para, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, rays_d, bc_rgb, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]
        ret['last_weight'] = weights[..., -1]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()):
            logger.info(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def train():
    # Load data
    if args.dataset_type == 'audface':
        logger.info(f"load landmarks from: {args.lmsdir}")
        logger.info(f"load audio from： {args.aud_file}")

        if args.with_test == 1:
            poses, auds, bc_img, hwfcxy = load_audface_data(args.datadir, args.testskip, args.test_file, args.aud_file,
                                                            head_nerf=True, lms_file=args.lmsdir)
            images = np.zeros(1)
        else:
            images, poses, auds, bc_img, hwfcxy, face_rects, \
            mouth_rects, i_split, driven_landmarks = \
                load_audface_data(args.datadir, args.testskip, aud_file=args.aud_file,
                                  head_nerf=True, lms_file=args.lmsdir)
        logger.info(f'Loaded audface: images_shape: {images.shape} || hwfcxy: {hwfcxy} || datadir: {args.datadir}')
        if args.with_test == 0:
            i_train, i_val = i_split

        near = args.near
        far = args.far
    else:
        raise Exception(f'Unknown dataset type{args.dataset_type} existing')
        return

    # Cast intrinsics to right types
    H, W, focal, cx, cy, intrinsic = hwfcxy
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    hwfcxy = [H, W, focal, cx, cy]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    # 第二行是加载已保存模型的结果
    render_kwargs_train, render_kwargs_test, start, models, optimizers = create_model(args)
    global_step = start

    aud_net = models['aud_model']
    aud_att_net = models['aud_att_model']
    optimizer_aud = optimizers['optimizer_aud']
    optimizer_aud_att = optimizers['optimizer_aud_att']
    optimizer_nerf = optimizers['optimizer_nerf']

    bds_dict = {
        'near': near,
        'far': far,
    }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)

    # Move training data to GPU
    bc_img = torch.Tensor(bc_img).to(device).float() / 255.0
    poses = torch.Tensor(poses).to(device).float()
    auds = torch.Tensor(auds).to(device).float()

    if args.render_only:
        logger.info('RENDER ONLY')
        with torch.no_grad():
            # Default is smoother render_poses path
            images = None
            testsavedir = os.path.join(basedir, expname, 'renderonly_{}_{:06d}'.format(
                'test' if args.render_test else 'path', start))
            os.makedirs(testsavedir, exist_ok=True)
            logger.info(f'test poses shape: {poses.shape}')
            auds_val = aud_net(auds)
            rgbs, disp, last_weight = render_path(poses, auds_val, bc_img, hwfcxy, args.chunk, render_kwargs_test,
                                                  gt_imgs=images, savedir=testsavedir, render_factor=args.render_factor)
            np.save(os.path.join(testsavedir, 'last_weight.npy'), last_weight)
            logger.info(f'Done rendering: {testsavedir}')
            imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)
            return

    # Prepare raybatch tensor if batching random rays
    N_rand = args.N_rand
    logger.info(f'N_rand: {N_rand} , use_batching: {args.use_batching}, sample_rate {args.sample_rate}')
    use_batching = args.use_batching

    N_iters = args.N_iters + 1
    logger.info('Begin')
    logger.info(f'TRAIN views are {i_train}')
    logger.info(f'VAL views are {i_val}')

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            pass
        else:
            # Random from one image
            img_i = np.random.choice(i_train)
            # 目标图像
            raw_img = imageio.imread(images[img_i])
            target = torch.as_tensor(raw_img).to(device).float() / 255.0
            pose = poses[img_i, :3, :4]
            rect = face_rects[img_i]
            aud = auds[img_i]
            landmark = driven_landmarks[i]  # (68 * 2)
            if global_step >= args.nosmo_iters:
                # args.smo_size=8
                smo_half_win = int(args.smo_size / 2)
                left_i = img_i - smo_half_win
                right_i = img_i + smo_half_win
                pad_left, pad_right = 0, 0
                if left_i < 0:
                    pad_left = -left_i
                    left_i = 0
                if right_i > i_train.shape[0]:
                    pad_right = right_i - i_train.shape[0]
                    right_i = i_train.shape[0]
                auds_win = auds[left_i:right_i]
                if pad_left > 0:
                    auds_win = torch.cat(
                        (torch.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
                if pad_right > 0:
                    auds_win = torch.cat(
                        (auds_win, torch.zeros_like(auds_win)[:pad_right]), dim=0)
                auds_win = aud_net(auds_win)
                aud = auds_win[smo_half_win]
                aud_smo = aud_att_net(auds_win)
            else:
                aud = aud_net(aud.unsqueeze(0))

            # 计算射线穿过的像素点位置
            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, focal, torch.Tensor(pose).to(device), cx, cy)  # (H, W, 3), (H, W, 3)

                if i < args.precrop_iters:
                    pass
                else:
                    # 这里给出了选点的范围 - 整张图片
                    coords = torch.stack(
                        torch.meshgrid(torch.linspace(0, H - 1, H),
                                       torch.linspace(0, W - 1, W)), -1)  # (H, W, 2)

                coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
                if args.sample_rate > 0:
                    # TODO: 计算人脸的关键点坐标
                    raw_lms = np.loadtxt(os.path.join(args.datadir, 'ori_imgs', f'{img_i}.lms'))
                    if raw_lms is None:
                        logger.error(f"Can not detect face in {i} iter, img_index: {img_i}")
                    lms_tensor = torch.tensor(raw_lms).long()

                    # 这里是计算人脸的位置
                    rect_inds = (coords[:, 0] >= rect[0]) & (coords[:, 0] <= rect[0] + rect[2]) \
                                & (coords[:, 1] >= rect[1]) & (coords[:, 1] <= rect[1] + rect[3])

                    coords_rect = coords[rect_inds]  # 包含人脸的像素
                    coords_norect = coords[~rect_inds]  # 不包含人脸的像素

                    """ 计算除了关键点之外的采样数量 """
                    sample_num = N_rand - landmark.shape[0]
                    rect_num = int(sample_num * args.sample_rate)
                    norect_num = sample_num - rect_num

                    select_inds_rect = np.random.choice(
                        coords_rect.shape[0], size=[rect_num], replace=False)  # (N_rand,)
                    select_coords_rect = coords_rect[select_inds_rect].long()  # (N_rand * sample_rate, 2)

                    select_inds_norect = np.random.choice(
                        coords_norect.shape[0], size=[norect_num], replace=False)  # (N_rand,)
                    select_coords_norect = coords_norect[select_inds_norect].long()  # (N_rand * (1-sample_rate), 2)

                    select_coords = torch.cat((lms_tensor, select_coords_rect, select_coords_norect), dim=0)
                else:
                    pass

                rays_o = rays_o[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                rays_d = rays_d[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
                batch_rays = torch.stack([rays_o, rays_d], 0)
                target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
                bc_rgb = bc_img[select_coords[:, 0],
                                select_coords[:, 1]]

        """  Core optimization loop  """
        if global_step >= args.nosmo_iters:
            rgb, disp, acc, _, extras = render_dynamic_face(H, W, focal, cx, cy, chunk=args.chunk, rays=batch_rays,
                                                            aud_para=aud_smo, bc_rgb=bc_rgb,
                                                            verbose=i < 10, retraw=True,
                                                            **render_kwargs_train)
        else:
            rgb, disp, acc, _, extras = render_dynamic_face(H, W, focal, cx, cy, chunk=args.chunk, rays=batch_rays,
                                                            aud_para=aud, bc_rgb=bc_rgb,
                                                            verbose=i < 10, retraw=True,
                                                            **render_kwargs_train)

        optimizer_nerf.zero_grad()
        optimizer_aud.zero_grad()
        optimizer_aud_att.zero_grad()
        img_loss = img2mse(rgb, target_s)
        # 计算LMD
        lms_loss = torch.mean((torch.tensor(rgb[:68]) - torch.tensor(landmark)) ** 2)

        loss = img_loss + lms_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0

        loss.backward()
        optimizer_nerf.step()
        optimizer_aud.step()
        if global_step >= args.nosmo_iters:
            optimizer_aud_att.step()

        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1500
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer_nerf.param_groups:
            param_group['lr'] = new_lrate

        for param_group in optimizer_aud.param_groups:
            param_group['lr'] = new_lrate

        for param_group in optimizer_aud_att.param_groups:
            param_group['lr'] = new_lrate * 5

        dt = time.time() - time0

        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}_head.tar'.format(i))
            # TODO: 这里需要把新加的模型参数保存进来
            torch.save({
                'global_step': global_step,

                'face_nerf_coarse_state_dict': models['nerf_attention_model'].state_dict(),
                'face_nerf_fine_state_dict': models['nerf_fine_attention_model'].state_dict(),
                'audnet_state_dict': aud_net.state_dict(),
                'audattnet_state_dict': aud_att_net.state_dict(),
                'face_unet_state_dict': models['face_unet_model'].state_dict(),

                'optimizer_nerf_state_dict': optimizer_nerf.state_dict(),
                'optimizer_aud_state_dict': optimizer_aud.state_dict(),
                'optimizer_audatt_state_dict': optimizer_aud_att.state_dict(),
                'optimizer_unet_state_dict': optimizers['optimizer_unet'].state_dict(),
            }, path)
            logger.info(f'Saved checkpoints at {path}')

        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(
                basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)
            logger.info(f'test poses shape: {poses[i_val].shape}')
            auds_val = aud_net(auds[i_val])
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_val]).to(device), auds_val, bc_img, hwfcxy, args.chunk,
                            render_kwargs_test, gt_imgs=None, savedir=testsavedir)
            logger.info('Saved test set')

        if i % args.i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()}  PSNR: {psnr.item()}")
        global_step += 1


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
