import torch
import torch.nn as nn
import model.GCN as GCN
import model.timeEmbedding as TE
import torch.nn.functional as F
from math import sqrt
import numpy as np


class GcnEncoderCell(nn.Module):
    def __init__(self,N, trainMatrix1, trainMatrix2,hops,device,tradGcn,dropout,dmodel,num_heads,Tin):
        """

        :param num_embedding: 有多少组时间，此处288
        :param embedding_dim:
        :param Tin:
        :param trainMatrix1:
        :param trainMatrix2:
        :param hops: gcn 跳数
        :param device:
        :param tradGcn:
        :param dropout:
        :param dmodel:
        :param num_heads:
        :param Tin: 输入时间的长度
        """
        super(GcnEncoderCell, self).__init__()
        self.temporalAttention=TemMulHeadAtte(dmodel=dmodel,num_heads=num_heads,dropout=dropout,device=device)
        self.f2=nn.Linear(in_features=2*dmodel,out_features=dmodel)
        self.f1=nn.Linear(in_features=2*dmodel,out_features=dmodel)
        self.f3=nn.Linear(in_features=dmodel,out_features=dmodel)

        self.device=device
        # 设置gate门
        self.gate=nn.Linear(in_features=2*dmodel,out_features=dmodel)
        self.batchNorm=nn.BatchNorm2d(num_features=dmodel)
        # 设置图卷积层捕获空间特征
        self.Gcn=GCN.GCN(T=Tin,trainMatrix1=trainMatrix1,trainMatrix2=trainMatrix2,device=device,tradGcn=tradGcn,dropout=dropout,hops=hops,dmodel=dmodel)
        self.spaceF=nn.Linear(2*dmodel,dmodel)


    def forward(self,x,hidden,tXin):
        """

        :param x: 只含流量值的embed batch*N*Tin*dmodel
        :param hidden: 此次输入的hidden:batch*N*Tin*dmodel
        :param tXin: 加了timeEmbedding的x值：tXin:[batch*N*Tin*dmodel]
        :return:
        """
        # 先捕获空间依赖
        gcnInput=torch.cat([x,hidden],dim=3) # batch*N*Tin*(2*dmodel)
        gcnInput=self.spaceF(gcnInput) # batch*N*Tin*dmodel
        gcnOutput=self.Gcn(gcnInput.permute(0,3,1,2).contiguous()) # batch*dmodel*N*Tin
        gcnOutput=gcnOutput.permute(0,2,3,1).contiguous() # batch*N*Tin*dmodel
        # 捕获时间依赖
        f2Input=torch.cat([hidden,tXin],dim=3) # batch*N*Tin*(2dmodel)
        key=self.f2(f2Input) # batch*N*Tin*dmodel

        f1Input=torch.cat([hidden,tXin],dim=3) # batch*N*Tin*(2dmodel)
        query=self.f1(f1Input)# batch*N*Tin*dmodel

        value=self.f3(hidden) # batch*N*Tin*dmodel

        # 做attention
        atten_mask=GcnEncoderCell.generate_square_subsequent_mask(B=query.size(0),N=query.size(1),T=query.size(2)).to(self.device) # batch*N*1*Tq*Ts
        out,atten=self.temporalAttention.forward(query=query,key=key,value=value,atten_mask=atten_mask) # batch*N*T*dmodel

        # 做gate
        # gateInput=torch.cat([gcnOutput,out],dim=3) # batch*N*Tin*2dmodel
        # gateInput=self.gate(gateInput) # batch*N*Tin*dmodel
        # gateInput=gateInput.permute(0,3,1,2).contiguous() # batch*dmodel*N*Tin
        # z=torch.sigmoid(self.batchNorm(gateInput).permute(0,2,3,1).contiguous()) # batch*N*Tin*dmodel
        # finalHidden=z*gcnOutput+(1-z)*out # batch*N*Tin*dmodel
        finalHidden=torch.sigmoid(gcnOutput+out)*torch.tanh(gcnOutput+out)

        return finalHidden # batch*N*Tin*dmodel

    @staticmethod
    def generate_square_subsequent_mask(B,N,T) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        mask_shape=[B,N,1,T,T]
        with torch.no_grad():
            return torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)


class GcnEncoder(nn.Module):
    def __init__(self,num_embedding,N, trainMatrix1, trainMatrix2,hops,device,tradGcn,
                 dropout,dmodel,num_heads,Tin,encoderBlocks):
        super(GcnEncoder, self).__init__()
        self.encoderBlock=nn.ModuleList()
        for i in range(encoderBlocks):
            self.encoderBlock.append(GcnEncoderCell(N=N,trainMatrix1=trainMatrix1,trainMatrix2=trainMatrix2,hops=hops,device=device,
                                           tradGcn=tradGcn,dropout=dropout,dmodel=dmodel,num_heads=num_heads,Tin=Tin))
        self.timeEmbed=TE.timeEmbedding(num_embedding=num_embedding,embedding_dim=dmodel,dropout=dropout)
        self.device=device
        self.encoderBlocks=encoderBlocks

        self.xCNN=nn.Conv2d(in_channels=1,out_channels=dmodel,kernel_size=(1,1))

    def forward(self,x,tx,ty):
        """

        :param x: 流量数据:[batch*N*Tin]
        :param tx: 时间数据:[batch*N*Tin]
        :return:
        """
        x=self.xCNN(x.unsqueeze(dim=1)) # batch*dmodel*N*Tin
        x=x.permute(0,2,3,1).contiguous() # batch*N*Tin*dmodel

        tx=self.timeEmbed(tx) # batch*N*Tin*dmodel
        ty=self.timeEmbed(ty) # batch*N*Tout*dmodel

        tXin=x+tx # batch*N*Tin*dmodel
        hidden=x.clone() # batch*N*Tin*dmodel
        skip=0
        for i in range(self.encoderBlocks):
            hidden=self.encoderBlock[i].forward(x=x,hidden=hidden,tXin=tXin) # Tin*batch*N
            skip = skip + hidden

        return skip+x,ty


class GcnDecoder(nn.Module):
    def __init__(self,dmodel,Tout,Tin):
        super(GcnDecoder, self).__init__()
        self.predict=nn.Linear(Tin,Tout)
        self.yCNN=nn.Conv2d(in_channels=dmodel,out_channels=1,kernel_size=(1,1))

    def forward(self,x,ty):
        """

        :param x: # batch*N*Tin*dmodel
        :param ty: batch*N*Tout*dmodel
        :return:
        """
        y=self.yCNN(x.permute(0,3,1,2).contiguous()) # batch*1*N*Tin
        y=y.squeeze(dim=1) # batch*N*Tin
        return self.predict(y) # batch*N*Tout


class TemMulHeadAtte(nn.Module):
    def __init__(self,dmodel,num_heads,dropout,device):
        """

        :param dmodel: embeddings之后的每个V每个T时刻的size
        :param num_heads: 多头注意力机制的head count
        """
        super(TemMulHeadAtte, self).__init__()
        self.dmodel=dmodel
        self.num_heads=num_heads
        self.dropout=nn.Dropout(p=dropout)
        self.device=device

        d_keys=dmodel//num_heads
        d_values=dmodel//num_heads

        self.query_projection=nn.Linear(in_features=dmodel,out_features=d_keys*num_heads)
        self.key_projection=nn.Linear(in_features=dmodel,out_features=d_keys*num_heads)
        self.value_projection=nn.Linear(in_features=dmodel,out_features=d_values*num_heads)
        self.out_projection=nn.Linear(in_features=d_values*num_heads,out_features=dmodel)

    def forward(self,query,key,value,atten_mask):
        """

        :param query: [batch*N*T*dmodel]
        :param key: [batch*N*T*dmodel]
        :param value: [batch*N*T*dmodel]
        :param atten_mask: [batch*N*1*Tq*Ts]
        :return: [batch*N*T*dmodel]
        """
        B,N,T,E=query.shape
        H=self.num_heads

        query=self.query_projection(query).view(B,N,T,H,-1) # batch*N*T*heads*d_keys
        key=self.key_projection(key).view(B,N,T,H,-1) # batch*N*T*heads*d_keys
        value=self.value_projection(value).view(B,N,T,H,-1) # batch*N*T*heads*d_values

        scale=1./sqrt(query.size(4))

        scores=torch.einsum("bnthe,bnshe->bnhts",(query,key)) # batch*N*head*Tq*Ts
        scores.masked_fill_(atten_mask,-np.inf) # batch*N*head*Tq*Ts
        scores=self.dropout(torch.softmax(scale*scores,dim=-1))

        out=torch.einsum("bnhts,bnshd->bnthd",(scores,value)) # batch*N*T*heads*d_values
        out=out.contiguous()
        out=out.view(B,N,T,-1) # batch*N*T*dmodel
        out=self.out_projection(out)

        # 返回最后的向量和得到的attention分数
        return out,scores






