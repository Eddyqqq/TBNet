import copy
import utils
import torch

import torch.nn as nn
import torch.nn.functional as F
from losses import IteLoss, ItaLoss
import clip
import CrossAttn


def unpad_padded(x, xl, dim=0):
    dims = list(range(len(x.shape)))
    dims.insert(0, dims.pop(dim))
    x = x.permute(*dims)
    return [xi[:xli] for xi, xli in zip(x, xl)]

def compute_lgt(lgt, kernel_type, kernel_size):
    feat_len = copy.deepcopy(lgt)
    for i in range(len(kernel_type)):
        feat_len -= int(kernel_size[0]) - 1
        feat_len = torch.div(feat_len, 2)
    return feat_len.cpu()

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=512, num_layers=1, dp_ratio=0.3, bidirectional=True):
        super(BiLSTM, self).__init__()

        self.dp_ratio = dp_ratio
        self.num_layers = num_layers
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.hidden_size = int(hidden_size / 2) if self.bidirectional == True else hidden_size
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dp_ratio,
            bidirectional=self.bidirectional)

    def forward(self, x, lgt, hidden=None):
        packed_seq = nn.utils.rnn.pack_padded_sequence(x, lgt)

        out, hidden = self.lstm(packed_seq, hidden)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(out)

        return outputs


class Conv1d(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_type, kernel_size):
        super(Conv1d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size
        self.temporal_conv = nn.ModuleList()
        for layer_idx in range(len(self.kernel_type)):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            self.temporal_conv.append(
                nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(self.kernel_size[0]), stride=1, padding=0)
            )
            self.temporal_conv.append(nn.BatchNorm1d(self.hidden_size))
            self.temporal_conv.append(nn.ReLU(inplace=True))
            self.temporal_conv.append(nn.MaxPool1d(kernel_size=int(self.kernel_size[1]), ceil_mode=False))

    def forward(self, vis_fea):
        for module in self.temporal_conv:
            vis_fea = module(vis_fea)

        return vis_fea.permute(2, 0, 1)


# class CrossAttn():

class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, vit_type,
            hidden_size=1024, gloss_dict=None, loss_weights=None,
    ):
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.clip, _ = clip.load(vit_type, device="cuda:0",
                                 download_root="./clip")
        if self.training:
        # IAM and IEM
        self.cond1d_type = ["K5P2", "K5P2"]
        self.cond1d_size = ["5", "2"]
        self.conv1d = Conv1d(input_size=512, hidden_size=hidden_size, cond1d_type=self.conv1d_type,
                             cond1d_size=self.cond1d_size)
        self.decoder = utils.Decode(gloss_dict, num_classes)
        self.temporal_model = BiLSTM(input_size=hidden_size, hidden_size=hidden_size,
                                     num_layers=2, bidirectional=True)
        self.classifier = NormLinear(hidden_size, self.num_classes)
        self.register_backward_hook(self.backward_hook)

    def backward_hook(self, module, grad_input, grad_output):
        for g in grad_input:
            g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.clip.encode_image(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x, len_x, label=None, label_lgt=None, ann=None):
        if len(x.shape) == 5:
            # videos
            batch, temp, channel, height, width = x.shape
            inputs = x.reshape(batch * temp, channel, height, width)
            framewise = self.masked_bn(inputs, len_x)
            framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)
        else:
            # frame-wise features
            framewise = x

        x = self.conv1d(framewise)
        # x: T, B, C
        lgt = compute_lgt(len_x, self.cond1d_type, self.cond1d_size)
        tm_outputs = self.temporal_model(x, lgt)  # T, B, C
        outputs = self.classifier(tm_outputs)
        (gls_emd, vis_emd) = self.calculate_similarity(tm_outputs, lgt, ann) if self.training \
            else (None, None)
        pred = None if self.training \
            else self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        return {
            "fea_lgt": lgt,
            "sequence_logits": outputs,
            "recognized_sents": pred,
            "vis_emd": vis_emd,
            "gls_emd": gls_emd,
        }

    def losses_calculation(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'SeqCTC':
                loss += weight * self.loss['CTCLoss'](ret_dict["sequence_logits"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["feat_len"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'IteLoss':
                IteLoss = weight * self.loss['IteLoss'](ret_dict["gls_emd"], ret_dict["vis_emd"], label.cpu(),
                                                        label_lgt)
                loss += IteLoss
            elif k == 'ItaLoss':
                ItaLoss = weight * self.loss['ItaLoss'](ret_dict["gls_emd"], ret_dict["vis_emd"], label.cpu(),
                                                        label_lgt)
                loss += ItaLoss
        return {"total_loss": loss, "IteLoss": IteLoss, "ItaLoss": ItaLoss}

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['IteLoss'] = IteLoss()
        self.loss['ItaLoss'] = ItaLoss()
        return self.loss
