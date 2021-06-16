import torch
import torch.nn as nn
import copy


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


#########################################################
# image encoders #

"Any image encoders could replace my implementation here"
#########################################################


class ImageEncoder(nn.Module):
    def __init__(self, input_channel_num=3, layer_params=None):
        # {'filter num': [16, 16, 32, 32], 'kernel sizes': [7, 3, 3, 3], 'strides': [1, 1, 2, 1]}
        super(ImageEncoder, self).__init__()
        layers = []
        layer_params['filter num'] = [input_channel_num] + list(layer_params['filter num'])
        for i in range(len(layer_params['filter num'])-1):
            layers.append(nn.Conv2d(layer_params['filter num'][i], layer_params['filter num'][i+1],
                                    kernel_size=layer_params['kernel sizes'][i], stride=layer_params['strides'][i]))
            # layers.append(nn.BatchNorm2d(filter_nums[i]))
            layers.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*layers)

    def forward(self, batch_image_tensors):
        return self.encoder(batch_image_tensors)


class KeyPointHeatmapEncoder(nn.Module):
    def __init__(self, layer_params, input_channel_num=1):
        # layer_params {'filter num': [1, 2], 'operator':['conv2d','max_pool'], 'kernel sizes': [3, 3], 'strides': [1, 2]}
        super(KeyPointHeatmapEncoder, self).__init__()
        layers = []
        layer_params['filter num'] = [input_channel_num] + layer_params['filter num']
        for i in range(len(layer_params['filter num']) - 1):
            if layer_params['operator'] == 'conv2d':
                layers.append(nn.Conv2d(layer_params['filter num'][i], layer_params['filter num'][i + 1],
                                        kernel_size=layer_params['kernel sizes'][i], stride=layer_params['strides'][i]))
                layers.append(nn.ReLU(inplace=True))
            else:
                layers.append(nn.MaxPool2d(kernel_size=layer_params['kernel sizes'][i], stride=layer_params['strides'][i]))
        layers.append(View(-1, ))
        self.encoder_conv = nn.Sequential(*layers)

    def forward(self, batch_heatmap_tensor):
        return self.encoder_conv(batch_heatmap_tensor)


#########################################################
# Graph Neural Network Library, a simple implementation #

"Any other graph neural network library could be used "
"TODO: use pytorchGeometrics lib"
"https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html"
#########################################################


class PairMessageGenerator(nn.Module):
    def __init__(self, dim_hv, dim_hw, msg_dim):
        """
        generate pair message between node Hv and Hw.
        since the cat operation, msgs from hv -> hw and hw -> hv are different
        """
        super(PairMessageGenerator, self).__init__()
        self.dim_hv, self.dim_hw, self.msg_dim = dim_hv, dim_hw, msg_dim
        self.in_dim = dim_hv + dim_hw  # row * feature_dim, 2048
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.in_dim),  # this layer norm is important to create diversity
            nn.Linear(self.in_dim, self.msg_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, Hv, Hw):
        """
        Hv: m v nodes : node feature
        Hw: m w nodes : node feature
        """
        inputs = torch.cat((Hv, Hw), 1)
        m_vw = self.mlp(inputs)
        return m_vw


class MessagePassing(nn.Module):
    def __init__(self, dim_h, msg_dim, msg_aggrgt='AVG'):
        """
        input:
        1 generate pair message between all connected nodes
        2 do message aggregate
        3 gru update, output all nodes' next h, only do one step
        """
        super(MessagePassing, self).__init__()
        self.dim_h = dim_h
        self.msg_aggrgt = msg_aggrgt
        self.msg_dim = msg_dim
        self.msg_generator = PairMessageGenerator(dim_h, dim_h, msg_dim)  # parameters shared
        self.update = nn.GRUCell(msg_dim, dim_h)  # parameters shared

    def forward(self, Ht_batch, A):
        """
        intput: Ht, hidden state of all nodes at time t, n instances
                Ht, 3D matrix, instances : nodes : node vectors (dim_h)
                A, Adjacent matrix, assuming all have the same adjacent matrix as all represent in one task kernel
        steps: 1 generate pair message stored in a hash table (for undirected graph here)
               2 aggregate msgs mt+1 for each node.
               3 feed ht and mt+1 in GRU update module
        output: Ht+1, hidden state of all nodes at time t+1, n instances
              : 3D matrix, instance entries : nodes: node vectors
        """
        # generate pair message
        pair_msgs = {}  ## hash msgs
        device = Ht_batch.device
        node_Ht_next_step = torch.zeros(Ht_batch.size()).to(device)  # n instances : nodes_num : node vectors (dim_h)
        # scan adjacent matrix
        for i in range(A.shape[0]):  # all the nodes
            pair_msgs[i] = []  # connected_msg_num * n * msg_dim
            Hv = Ht_batch[:, i, :]  # n*dim_h
            for j in range(A.shape[1]):  # scan the other nodes
                if A[i, j] == 1:  ## connected nodeds
                    Hw = Ht_batch[:, j, :]  # n*dim_h
                    # msg from Hv to Hw
                    msg_i_j = self.msg_generator(Hv, Hw)  # n * dim_msgs
                    pair_msgs[i].append(msg_i_j)
            # aggregate all connected msgs
            msg_next_step = self.aggregate_msgs(pair_msgs[i]).to(device)  # n*msg_dim
            # update function, get next hidden state
            Ht_next_step = self.update(msg_next_step, Hv)  # n * dim_h
            node_Ht_next_step[:, i, :] = Ht_next_step

        return node_Ht_next_step

    def aggregate_msgs(self, connected_msgs_list):
        """
        # for n graph instances
        # given edge index number
        # this connected_msgs_list contains all connnected edges msg, say m connected edges
        # connected_msgs_list: m items, each item is n*msg_dim matrix
        # return n*msg_dim matrix, representing next step msg of this edge index
        :param connected_msgs_list:
        :return:
        """
        msg_num = len(connected_msgs_list)
        agg_msg = connected_msgs_list[0]
        for i in range(1, msg_num):
            agg_msg += connected_msgs_list[i]

        if self.msg_aggrgt == 'AVG':
            return agg_msg / msg_num
        elif self.msg_aggrgt == 'SUM':
            return agg_msg


class GraphReadOut(nn.Module):
    """
    that's the readout function
    input: all nodes final hidden state
    output: readout vector representing the graph
    """
    def __init__(self, input_dim, node_num, output_dim):
        super(GraphReadOut, self).__init__()
        self.input_dim = input_dim  # 1024
        self.output_dim = output_dim
        self.node_num = node_num
        self.mlp = nn.Sequential(
            View((-1, self.node_num * self.input_dim)),
            # nn.LayerNorm(self.node_num*self.input_dim),  # layer norm is good
            nn.Linear(self.node_num * self.input_dim, self.output_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, nodes_final_hidden):
        """
        nodes_final_hidden: n graphs : node_num (as the sequence) : node_hidden_state_dim
        output: n scalar values representing weight of each graph
        """
        return self.mlp(nodes_final_hidden)


# ****************************************************************
# --Test--    : Fri, 16:02:00, Jun 7, 2019, MDT
# --Result--  : PASS / NG, Jun Jin, 16:02:00, Jun 7, 2019, MDT
# ****************************************************************
# Ht = torch.randn((120, 2, 128))
# readout = GraphReadOut(128, 2)
# outputs = readout(Ht)
# print (outputs.shape)
# error_vec = torch.ones(120)
# outputs.backward(error_vec)
# print(list(readout.parameters())[0].grad)
# # when include lstm, forward function is OK. but after backward, grad is None

# images = np.load('../raw/sample_img.npy')
#     transform = T.Compose([
#         T.ToPILImage(),
#         T.ToTensor(),
#         T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
#     images = transform(images).to(device).unsqueeze(0)
#     images = torch.randn((10,3,128,128)).to(device)
#     deep_geometry_set = deep_geometry_set.to(device)
#     basis_weighted_layer = deep_geometry_set(images)  # dim N * (7*_gnn_output_dim)

def init_weights(m):
    if type(m) == nn.Linear:
        # torch.nn.init.xavier_uniform(m.weight)
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')


def tie_weights_of_two_networks(src, tgt):
    i = 0
    params = list(tgt.parameters())
    assert len(list(src.parameters())) == len(params), "failed, length not match!"
    for f in src.parameters():  # set weights
        f.data = params[i].data
        i += 1

#########################################################
# baseline encoders #

"baseline encoders for ablation study"
#########################################################


class PixelE2E(nn.Module):
    def __init__(self, input_channel_num=3, layer_params=None, output_dim=896):
        # {'filter num': [16, 16, 32, 32], 'kernel sizes': [7, 3, 3, 3], 'strides': [1, 1, 2, 1]}
        super(PixelE2E, self).__init__()
        self.conv_encoder = nn.Sequential(
            ImageEncoder(input_channel_num=input_channel_num, layer_params=layer_params),
            View(-1, ),
            nn.Linear(169, output_dim),  # for a fair comparison 896 is exactly the dim of geometry set output dim
            nn.ReLU())

    def forward(self, batch_image_tensors):
        return self.conv_encoder(batch_image_tensors)


class M6EKViEncoder(nn.Module):
    """
    M6 baseline, Ours - G: E+k+Vi
    """
    def __init__(self, _conv_encoder=None, _encoder_out_channels=32, _vi_key_pointer=None, key_point_num=20, _debug_tool=None, _debug_frequency=None, _vi_mode=True):
        """
        deep geometric feature set as state representation
        :param _conv_encoder:
        :param _encoder_out_channels:
        :param _vi_key_pointer:
        :param key_point_num: default 20
        """
        super(M6EKViEncoder, self).__init__()
        self.encoder = _conv_encoder
        self.k_heatmaps_layer = nn.Sequential(
            nn.Conv2d(_encoder_out_channels, key_point_num, kernel_size=(1, 1), stride=(1, 1)),
            nn.BatchNorm2d(key_point_num),
            nn.ReLU())
        self.vi_key_pointer = _vi_key_pointer
        self.debug_tool = _debug_tool
        self.debug_frequency = _debug_frequency
        self.it_count = 0
        self.vi_mode = _vi_mode

    def forward(self, batch_image_tensors):
        self.it_count += 1  # this is only for debug purpose
        x = self.encoder(batch_image_tensors)
        x = self.k_heatmaps_layer(x)  # x: k=20 heat maps
        print('conv heatmap size')
        print(x.shape)
        if self.debug_tool is not None and self.it_count % self.debug_frequency == 0:  # if debug mode, output more info
            conv_heatmaps = copy.deepcopy(x.detach().to('cpu'))
            x, gauss_mu, gauss_maps, vi_key_point_heatmaps = self.vi_key_pointer(x, self.vi_mode)
            self.debug_tool.vis_debugger(batch_image_tensors.detach().to('cpu'),
                                         conv_heatmaps,
                                         gauss_maps.detach().to('cpu'),
                                         vi_key_point_heatmaps.detach().to('cpu'),
                                         gauss_mu[:, :, 0].detach().to('cpu'),
                                         gauss_mu[:, :, 1].detach().to('cpu'),
                                         show_graphs=True
                                         )
        else:
            x, _, _, _ = self.vi_key_pointer(x)  # x: k=20 key points with visual features
        return x


# class M7EKGEncoder(nn.Module):  # that's DeepGeometricSet with vi_mode = False
# class M8: keypoint bottle neck: E + K, # that's M6EKViEncoder with vi_mode = False


