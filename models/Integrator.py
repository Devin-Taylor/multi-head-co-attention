import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


class MultiheadCoattention(nn.Module):
    '''
    Implementation of multi-head attention adapted from https://github.com/shichence/AutoInt

    The class implements the multi-head co-attention algorithm.
    '''

    def __init__(self, embedding_dim, num_units, num_heads=2, dropout_keep_prob=1.):
        super(MultiheadCoattention, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_units = num_units
        self.key_dim = nn.Parameter(torch.tensor(data=[num_units//num_heads], requires_grad=False, dtype=torch.float32))

        self.Q = torch.nn.Linear(embedding_dim, num_units)
        self.K = torch.nn.Linear(embedding_dim, num_units)
        self.V = torch.nn.Linear(embedding_dim, num_units)
        self.res_k = torch.nn.Linear(embedding_dim, num_units)
        self.res_q = torch.nn.Linear(embedding_dim, num_units)

        self.softmax_q = nn.Softmax(dim=1)
        self.softmax_k = nn.Softmax(dim=2)

        self.dropout = torch.nn.Dropout(1-dropout_keep_prob)
        self.layer_norm = nn.LayerNorm(num_units)

    def forward(self, queries, keys, values, has_residual=True):
        Q = F.relu(self.Q(queries))
        K = F.relu(self.K(keys))
        V = F.relu(self.V(values))
        if has_residual:
            res_k = F.relu(self.res_k(queries))
            res_q = F.relu(self.res_q(values))

        # split heads
        chunk_size = int(self.num_units / self.num_heads)
        Q_ = torch.cat(torch.split(Q, chunk_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, chunk_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, chunk_size, dim=2), dim=0)
        # get scaled similarity
        weights = torch.bmm(Q_, K_.transpose(1, 2))
        weights = weights / torch.sqrt(self.key_dim)
        # save similarities for later inspection
        ret_weights = weights.clone()
        ret_weights = ret_weights.cpu().detach().numpy()
        # get weights
        weights_k = self.softmax_k(weights) # prob dist over keys
        weights_q = self.softmax_q(weights) # prob dist over queries
        weights_k = self.dropout(weights_k)
        weights_q = self.dropout(weights_q)
        # get outputs
        v_out = torch.bmm(weights_k, V_)
        q_out = torch.bmm(weights_q.transpose(1, 2), Q_)
        # reshuffle for heads
        restore_chunk_size = int(v_out.size(0) / self.num_heads)
        v_out = torch.cat(torch.split(v_out, restore_chunk_size, dim=0), dim=2)
        q_out = torch.cat(torch.split(q_out, restore_chunk_size, dim=0), dim=2)
        # add residual connection
        if has_residual:
            v_out += res_k
            q_out += res_q
        # combine latent spaces through addition and normalise
        outputs = v_out + q_out
        outputs = F.relu(outputs)
        outputs = self.layer_norm(outputs)

        return outputs, ret_weights

class MethSpectIntegrator(nn.Module):
    '''
    Used for all PPMI experiments.
    '''
    def __init__(self, meth_enc, spect_enc, **kwargs):
        super(MethSpectIntegrator, self).__init__()

        self.classification = kwargs.get("classification")
        self.ignore_attention = kwargs.get("ignore_attention")
        self.weights = None

        self.meth_encoder = meth_enc(kwargs.get("feature_size"), kwargs.get("embedding_size"))
        self.spect_encoder = spect_enc()

        if not self.ignore_attention:
            self.multihead_attention = MultiheadCoattention(embedding_dim=kwargs.get("embedding_size"),
                                                            num_units=kwargs.get("block_shape"),
                                                            num_heads=kwargs.get("num_heads"),
                                                            dropout_keep_prob=kwargs.get("dropout_keep_prob"))
            multiplier = 1
        else:
            self.integrator_1 = nn.Linear(kwargs.get("embedding_size"), kwargs.get("block_shape"))
            self.integrator_2 = nn.Linear(kwargs.get("embedding_size"), kwargs.get("block_shape"))
            multiplier = 2

        self.h1 = torch.nn.Linear(kwargs.get("feature_size") * multiplier*kwargs.get("block_shape"), kwargs.get("hidden_dim"))
        self.dropout = nn.Dropout(0.2)

        if self.classification:
            output_dim = 2
        else:
            output_dim = 1

        self.out = torch.nn.Linear(kwargs.get("hidden_dim"), output_dim)


    def get_weights(self):
        return self.weights

    def forward(self, x):
        # split out inputs into separate modes
        y = (x[1])
        x = (x[0])
        # pass through mode encoders
        x = F.relu(self.meth_encoder(x))
        y = F.relu(self.spect_encoder(y))
        # "manipulate" spect output
        y = y.reshape((y.shape[0], y.shape[1], y.shape[2]*y.shape[3]*y.shape[4]))
        y = y.permute(0, 2, 1)

        if not self.ignore_attention:
            z, self.weights = self.multihead_attention(
                queries=x,
                keys=y,
                values=y)
        else:
            h_1 = self.integrator_1(x)
            h_2 = self.integrator_2(y)
            z = torch.cat((h_1, h_2), dim=1)

        z = F.relu(self.h1(z.reshape(z.shape[0], z.shape[1]*z.shape[2])))
        z = self.dropout(z)
        out = self.out(z)
        if self.classification:
            out = F.log_softmax(out, dim=1)

        return out
