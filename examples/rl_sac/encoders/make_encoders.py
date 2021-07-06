from deep_geometry.model import *
import deep_geometry.tools as gtools
import torchvision.transforms as T
from deep_geometry.vi_key_point_bottleneck import *
from deep_geometry.debug_tools import *


def make_encoders(cfg):
    if cfg['name'] == 'm1_deep_geometry':
        return m1_build_deep_geometry_set_encoder(cfg['params'])
    elif cfg['name'] == 'm4_pixel_e2e':
        return m4_build_conv_encoder(cfg['params'])
    elif cfg['name'] == 'm6_EKVi':
        return m6_build_EKVi_encoder(cfg['params'])
    elif cfg['name'] == 'm7_EKG':
        return m7_build_EKG_encoder(cfg['params'])
    elif cfg['name'] == 'm8_keypoint_bottleneck':
        return m8_build_EK_encoder(cfg['params'])
    else:
        raise NotImplemented('encoder not defined!')


def m1_build_deep_geometry_set_encoder(cfg, vi_mode_on=True):
    # encoder:
    # name: deep_geometry_set
    # params:
    # image_input_channel_num: 3
    # conv_layer_params: {'filter num': [16, 16, 32, 32], 'kernel sizes': [7, 3, 3, 3], 'strides': [1, 1, 2, 1]}
    # gnn_params: {'layer_nums': [3, 4, 5], 'msg_dim': 128, 'h_dim': 128, 'output_dim': 128, 'share_basis_graphs': True}
    # key_point_num: 20
    # key_pointer_gauss_std: 0.1,
    # key_pointer_flatten: 'conv2d'
    # debug_save_dir: '../../raw'
    # debug_frequency: 1000
    # initialize deep geometry image encoder
    image_input_channel_num = cfg['image_input_channel_num']
    conv_layer_params = cfg['conv_layer_params']
    gnn_params = cfg['gnn_params']
    print(gnn_params)
    key_point_num = cfg['key_point_num']
    key_pointer_gauss_std = cfg['key_pointer_gauss_std']
    key_pointer_flatten = cfg['key_pointer_flatten']
    debug_save_dir = cfg['debug_save_dir']
    debug_frequency = cfg['debug_frequency']
    output_dim = cfg['output_dim']
    conv_encoder = gtools.ImageEncoder(input_channel_num=image_input_channel_num, layer_params=conv_layer_params)
    encoder_out_channel_num = conv_layer_params['filter num'][-1]

    vi_key_pointer = ViKeyPointBottleneck(_key_point_num=key_point_num, _gauss_std=key_pointer_gauss_std,
                                          _output_dim=gnn_params['h_dim'], _flatten=key_pointer_flatten)
    debug_tool = DeepGeometrySetVisualizer(_save_dir=debug_save_dir, _name_prefix='ours', _verbose=True)
    deep_geometry_set = DeepGeometricSet(_conv_encoder=conv_encoder,
                                         _encoder_out_channels=encoder_out_channel_num,
                                         _vi_key_pointer=vi_key_pointer,
                                         key_point_num=20, _gnn_params=gnn_params,
                                         _debug_tool=None, _debug_frequency=None,
                                         _vi_mode_on=vi_mode_on
                                         )
    deep_geometry_set.apply(gtools.init_weights)
    return deep_geometry_set


def m4_build_conv_encoder(cfg):  # Pixel E2E
    image_input_channel_num = cfg['image_input_channel_num']
    conv_layer_params = cfg['conv_layer_params']
    output_dim = cfg['output_dim']
    conv_encoder = gtools.ImageEncoder(input_channel_num=image_input_channel_num, layer_params=conv_layer_params)
    conv_encoder = nn.Sequential(
        conv_encoder,
        gtools.View(-1, )
        # nn.Linear(169, output_dim),  # for a fair comparison 896 is exactly the dim of geometry set output dim
        # nn.ReLU()
    )
    return conv_encoder


def m6_build_EKVi_encoder(cfg, vi_mode_on=True):  # M6: ours - G: E + K + Vi
    image_input_channel_num = cfg['image_input_channel_num']
    conv_layer_params = cfg['conv_layer_params']
    # gnn_params = cfg['gnn_params']
    key_point_num = cfg['key_point_num']
    key_pointer_gauss_std = cfg['key_pointer_gauss_std']
    key_pointer_flatten = cfg['key_pointer_flatten']
    debug_save_dir = cfg['debug_save_dir']
    debug_frequency = cfg['debug_frequency']
    output_dim = cfg['output_dim']
    conv_encoder = gtools.ImageEncoder(input_channel_num=image_input_channel_num, layer_params=conv_layer_params)
    encoder_out_channel_num = conv_layer_params['filter num'][-1]
    vi_key_pointer = ViKeyPointBottleneck(_key_point_num=key_point_num, _gauss_std=key_pointer_gauss_std,
                                          _output_dim=gnn_params['h_dim'], _flatten=key_pointer_flatten)
    debug_tool = DeepGeometrySetVisualizer(_save_dir=debug_save_dir, _name_prefix='ours', _verbose=True)
    m6_encoder = M6EKViEncoder(_conv_encoder=conv_encoder,
                               _encoder_out_channels=encoder_out_channel_num,
                               _vi_key_pointer=vi_key_pointer,
                               key_point_num=20,
                               _debug_tool=debug_tool, _debug_frequency=debug_frequency, _vi_mode=vi_mode_on
                               )
    m6_encoder.apply(gtools.init_weights)

    return m6_encoder


# M7EKGEncoder:  # that's DeepGeometricSet with vi_mode = False
# M8: keypoint bottle neck: E + K, # that's M6EKViEncoder with vi_mode = False


def m7_build_EKG_encoder(cfg):  # M7: ours - Vi: E + K + G
    return m1_build_deep_geometry_set_encoder(cfg, vi_mode_on=False)


def m8_build_EK_encoder(cfg):  # M8: keypoint bottleneck : E + K
    return m6_build_EKVi_encoder(cfg, vi_mode_on=False)
