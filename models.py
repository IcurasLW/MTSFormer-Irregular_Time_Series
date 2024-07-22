
import numpy as np
import torch
import torch.nn as nn




DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEVICE = 'cpu'
torch.autograd.set_detect_anomaly(True)
torch.set_default_dtype(torch.float32)



class PositionalEncodingTF(nn.Module):
    def __init__(self, d_model, max_len=500, MAX=10000):
        super(PositionalEncodingTF, self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        self.MAX = MAX
        self._num_timescales = (d_model // 2) if d_model % 2 == 0 else ((d_model+1) // 2)

    def getPE(self, P_time):
        B = P_time.shape[1]
        timescales = self.max_len ** np.linspace(0, 1, self._num_timescales)
        times = torch.Tensor(P_time.cpu()).unsqueeze(2)
        scaled_time = times / torch.Tensor(timescales[None, None, :])
        pe = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], axis=-1)  # T x B x d_model
        pe = pe.type(torch.FloatTensor)
        if self.d_model % 2 != 0:
            pe = pe[:, :, :-1]
        return pe

    def forward(self, P_time):
        pe = self.getPE(P_time)
        pe = pe.to(DEVICE)
        return pe



class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_layer=3, kernel_size=3, stride=1, padding='same'):
        super().__init__()
        self.n_layer = n_layer
        head_conv = [nn.Conv1d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, dilation=1, stride=stride)]
        self.conv_list = nn.ModuleList(head_conv + [nn.Conv1d(out_channels, 
                                                              out_channels, 
                                                              padding=padding, 
                                                              kernel_size=kernel_size + i*(2), 
                                                              dilation=i+1, 
                                                              stride=stride) for i in range(1, n_layer)])

    def forward(self, x):
        for i in range(self.n_layer):
            if i == 0:
                out = self.conv_list[i](x)
            else:
                out = self.conv_list[i](out)
            out_tanh = torch.tanh(out)
            out_sigmoid = torch.sigmoid(out)
            out_gate = out_tanh * out_sigmoid
            out = out + out_gate
            
        return out



class SensorEncoder(nn.Module):
    def __init__(self, d_inp=36, d_hid=128, n_head=1):
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_hid, nhead=n_head, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=2)
        self.linear = nn.Linear(d_hid, d_hid)
        
    def forward(self, sensor_data):
        sensor_data = self.transformer_encoder(sensor_data)
        sensor_data = self.linear(sensor_data)
        return sensor_data



class MTSFormer_Layer(nn.Module):
    def __init__(self, d_inp=36, d_hid=128, n_head=1):
        super().__init__()
        self.d_inp = d_inp
        self.d_hid = d_hid
        self.local_conv1d = TemporalConv(in_channels=self.d_inp, out_channels=d_hid, kernel_size=7, stride=1, padding='same')
        timetoken_enc =  nn.TransformerEncoderLayer(d_hid, nhead=1, dropout=0, batch_first=True)
        self.timetoken_enc = nn.TransformerEncoder(timetoken_enc, num_layers=1)
        
        self.sensor_transformer = SensorEncoder(d_inp, d_hid, 1)
        self.selfatt = nn.MultiheadAttention(d_hid, num_heads=n_head, dropout=0.2)
        self.layer_norm_1 = nn.LayerNorm(normalized_shape=self.d_hid)
        self.feedward = nn.Sequential(nn.Linear(in_features=d_hid, out_features=d_hid*3),
                                      nn.ReLU(),
                                      nn.Linear(in_features=d_hid*3, out_features=d_hid))
        
        self.layer_norm_2 = nn.LayerNorm(normalized_shape=self.d_hid)
        
    def forward(self, conv_data, att_data, sensor_data):
        conv_data = conv_data.permute(0,2,1)
        conv_data = self.local_conv1d(conv_data)
        conv_data = conv_data.permute(0,2,1)
        
        att_data = self.timetoken_enc(att_data)
        sensor_data = self.sensor_transformer(sensor_data)
        
        all_data = torch.cat([conv_data, att_data, sensor_data], dim=1)
        conv_len = conv_data.shape[1]
        att_len = conv_data.shape[1]
        
        all_data_att, _ = self.selfatt(all_data, all_data, all_data)
        all_data = all_data + all_data_att
        all_data_1 = self.layer_norm_1(all_data)
        all_data_2 = self.feedward(all_data_1)
        all_data_2 = self.layer_norm_2(all_data_2 + all_data_1)
        
        conv_data = all_data[:, :conv_len, :]
        att_data = all_data[:, conv_len:conv_len+att_len, :]
        sensor_data = all_data[:, conv_len+att_len:, :]

        return conv_data, att_data, sensor_data


class MTSFormer(nn.Module):
    def __init__(self, d_inp=1, d_hid=128, max_len=215, n_layers=4, n_classes=2, n_head=2, dropout=0.2):
        super().__init__()
        self.max_len = max_len
        self.n_layers = n_layers
        self.pos_encoder = PositionalEncodingTF(d_inp, max_len, 10000)
        self.d_hid = d_hid
        self.Trans_head = MTSFormer_Layer(d_inp=d_inp, d_hid=d_hid, n_head=1)
        self.Trans_modi_layer = MTSFormer_Layer(d_inp=self.d_hid, d_hid=self.d_hid, n_head=n_head)
        self.Transformer_encoder = nn.ModuleList([self.Trans_head] + [self.Trans_modi_layer for _ in range(n_layers - 1)])
        
        self.mlp = nn.Sequential(nn.Linear(self.d_hid*2 + d_inp, self.d_hid*3),
                                 nn.BatchNorm1d(self.d_hid*3),
                                 nn.Dropout(dropout),
                                 nn.ReLU(),
                                 nn.Linear(self.d_hid*3, n_classes))
        
        self.att_enc = nn.Linear(d_inp, d_hid)
        self.sen_enc = nn.Linear(max_len, d_hid)
        self.tanh_encoder = nn.Sequential(nn.Linear(self.d_hid, self.d_hid),
                                          nn.Tanh()
                                          )
        self.sigmoid_encoder = nn.Sequential(nn.Linear(self.d_hid, self.d_hid),
                                            nn.Sigmoid()
                                            )
        
    def forward(self, data, time, seq_len):
        seq_mask = torch.arange(self.max_len)[None, :] >= (seq_len.cpu()[:, None])
        seq_mask = seq_mask.int().to(DEVICE).unsqueeze(-1)
        seq_len = seq_len.unsqueeze(-1)
        pe = self.pos_encoder(time)
        data = data.permute(1,0,2)
        sensor_data = data.clone().permute(0,2,1) + pe.permute(0,2,1)
        conv_data = data.clone()
        
        att_irre = (data == 0).float()
        sensor_irre = (sensor_data == 0).float()
        conv_irre = (data == 0).float()
        
        att_irre = self.att_enc(att_irre)
        sensor_irre = self.sen_enc(sensor_irre)
        conv_irre, att_irre, sensor_irre = self.Trans_head(conv_irre, att_irre, sensor_irre)
        
        conv_irre = self.sigmoid_encoder(conv_irre) * self.tanh_encoder(conv_irre)
        att_irre = self.sigmoid_encoder(att_irre) * self.tanh_encoder(att_irre)
        sensor_irre = self.sigmoid_encoder(sensor_irre) * self.tanh_encoder(sensor_irre)
        
        att_data = data.clone() + pe
        att_data = self.att_enc(att_data)
        sensor_data = self.sen_enc(sensor_data)
        for i in range(self.n_layers):
            conv_data, att_data, sensor_data = self.Transformer_encoder[i](conv_data, att_data, sensor_data)
        
        conv_data = conv_data + conv_irre
        att_data = att_data + att_irre
        sensor_data = sensor_data + sensor_irre
        
        conv_data = torch.sum(conv_data * (1-seq_mask), dim=1) / seq_len
        att_data = torch.sum(att_data * (1-seq_mask), dim=1) / seq_len
        sensor_data = torch.sum(sensor_data, dim=2) / seq_len
        
        out = torch.cat([conv_data, att_data, sensor_data], dim=1)
        out = self.mlp(out)
        return out
