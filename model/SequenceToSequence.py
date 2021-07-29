import random

import torch
import torch.nn as nn
import model.GCN as gcn


class Encoder(nn.Module):
    def __init__(self,inputSize,hiddenSize,n_layers,dropout):
        """

        :param inputSize: 输入数据的维度
        :param hiddenSize: 输出时候的维度
        :param dropout:
        :param n_layers: 几层GRU
        """
        super().__init__()
        self.inputSize=inputSize
        self.hiddenSize=hiddenSize
        self.dropout=nn.Dropout(dropout)
        self.n_layers=n_layers
        self.GRU=nn.GRU(inputSize,hiddenSize,n_layers,dropout=dropout)

    def forward(self,x):
        """

        :param x: T*batch*N
        :return:
        """
        output,hidden=self.GRU(x)
        return output,hidden


class Decoder(nn.Module):
    def __init__(self,inputSize,hiddenSize,n_layers,dropout):
        super(Decoder, self).__init__()
        self.inputSize=inputSize
        self.hiddenSize=hiddenSize
        self.n_layers=n_layers
        self.dropout=nn.Dropout(dropout)
        self.GRU=nn.GRU(inputSize,hiddenSize,n_layers,dropout=dropout)

    def forward(self,x,hidden):
        """

        :param x: x是最后一个时刻的数据 1*batch*N
        :param hidden:
        :return:
        """
        output,hid=self.GRU(x,hidden)
        return output,hid


class Seq2Seq(nn.Module):
    def __init__(self,Tin,Tout,inputSize,hiddenSize,n_layers,dropout,device,trainMatrix1, trainMatrix2,hops, tradGcn=False):
        super(Seq2Seq, self).__init__()
        self.encoder=Encoder(inputSize=inputSize,hiddenSize=hiddenSize,n_layers=n_layers,dropout=dropout)
        self.decoder=Decoder(inputSize=inputSize,hiddenSize=hiddenSize,n_layers=n_layers,dropout=dropout)
        self.device=device
        self.attentionLinear=nn.ModuleList()
        self.outputT=Tout
        self.GCN = gcn.GCN(T=1, trainMatrix1=trainMatrix1, trainMatrix2=trainMatrix2, device=device, hops=hops,
                           tradGcn=tradGcn, dropout=dropout)
        self.dimCNN=nn.Conv2d(in_channels=2,out_channels=1,kernel_size=(1,1))
        for i in range(Tout):
            self.attentionLinear.append(nn.Linear(in_features=Tin+n_layers,out_features=n_layers))

    def forward(self,x,y,teacher_forcing_ratio=0.5):
        """

        :param x: x=[Tin*batch*N]
        :param y: y=[Tout*batch*N*2]
        :param teacher_forcing_ratio: 决定是否使用ground_truth,在train跟valid流程不一样
        :return:
        """
        batch_size=x.shape[1]
        target_len=y.shape[0]
        outputs=torch.zeros(y[...,0].shape).to(self.device)

        _,hidden=self.encoder(x) # 得到_做attention以避免误差累加 其中_:[Tin*batch*N] hidden:[n_layers*batch*N]
        decoder_input=x[-1:,:,:]
        for i in range(target_len):
            # 对decoder_input做一次图卷积
            decoder_input = self.GCN(decoder_input.permute(1, 2, 0).contiguous())  # 1*batch*N
            # if i>0:
            #     decoder_input=torch.cat([decoder_input.unsqueeze(dim=3),y[i-1:i,...,1:]],dim=3) # 1*batch*N*2
            #     decoder_input=self.dimCNN(decoder_input.permute(1,3,2,0).contiguous()).squeeze(dim=1) # batch*N*1
            #     decoder_input=decoder_input.permute(2,0,1).contiguous() # 1*batch*N
            #  attention部分
            hidden=torch.cat([_,hidden],dim=0).permute(1,2,0).contiguous() # batch*N*(Tin+1)
            hidden=self.attentionLinear[i](hidden).permute(2,0,1).contiguous() # 1*batch*N
            # run decode for one time step
            output,hidden=self.decoder(decoder_input,hidden)
            outputs[i]=output
            # decide if we are going to use teacher forcing or not
            teacher_forcing=random.random() < teacher_forcing_ratio
            decoder_input=y[i,...,0].unsqueeze(dim=0) if teacher_forcing else output
        return outputs # outputT*batch*N


