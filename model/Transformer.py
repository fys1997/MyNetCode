import torch
import torch.nn as nn
import model.PositionEmbedding as PE


class Transformer(nn.Module):
    def __init__(self, dmodel, cnn_in_channels, cnn_out_channels, nhead, num_layers, dropout, device,Tout):
        super().__init__()
        self.transformerCNN1 = nn.Conv1d(in_channels=cnn_in_channels, out_channels=cnn_out_channels, kernel_size=1)
        self.transformerCNN2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_in_channels, kernel_size=1)
        # encoder
        self.encoderLayer = nn.TransformerEncoderLayer(d_model=dmodel, nhead=nhead, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoderLayer, num_layers=num_layers)
        # decoder
        self.decoderLayer = nn.TransformerDecoderLayer(d_model=dmodel, nhead=nhead, dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoderLayer, num_layers=num_layers)
        self.device = device
        self.positionEmbedding = PE.PositionalEncoding(d_model=dmodel, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.Tout=Tout
        self.projection=nn.Linear(8+Tout,8+Tout)

    def forward(self, X, Y, teacher_forcing_ratio):
        """

        :param X: [Tin*batch*N]
        :param Y: [Tout*batch*N] 时间坐标
        :param teacher_forcing_ratio: decide train or val
        :return:
        """
        X=X.permute(0,2,1).contiguous() # Tin*N*batch
        Y=Y.permute(0,2,1).contiguous() # Tout*N*batch
        X = self.transformerCNN1(X)  # Tin*dmodel*batch
        Y = self.transformerCNN1(Y)  # Tout*dmodel*batch
        X=X.permute(0,2,1).contiguous() # Tin*batch*dmodel
        Y=Y.permute(0,2,1).contiguous() # Tout*batch*dmodel
        xin = self.positionEmbedding(X)
        encoder_output = self.encoder(xin)

        # decoder 一步预测
        Y=torch.cat([X[-8:,...],Y],dim=0) # (8+Tout)*dmodel*batch
        target_len = Y.shape[0]
        Y=self.positionEmbedding(Y)
        # tgt_mask = nn.Transformer().generate_square_subsequent_mask(target_len).to(self.device)
        decoder_output=self.decoder(tgt=Y,memory=encoder_output) # (8+Tout)*dmodel*batch
        decoder_output=self.projection(decoder_output.permute(2,1,0).contiguous()) # batch*dmodel*(8+Tout)


        decoder_output = self.transformerCNN2(decoder_output)  # batch*N*(8+Tout)
        return decoder_output[...,-self.Tout:] # batch*N*Tout
