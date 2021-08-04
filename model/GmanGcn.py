import torch
import torch.nn as nn
import model.GCN as GCN
import model.timeEmbedding as TE
import torch.nn.functional as F
import model.PositionEmbedding as PE


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
        self.temporalAttention=nn.MultiheadAttention(embed_dim=dmodel,num_heads=num_heads,dropout=dropout)
        self.f2=nn.Linear(in_features=2*Tin,out_features=Tin)
        self.f1=nn.Linear(in_features=2*Tin,out_features=Tin)
        self.f3=nn.Linear(in_features=Tin,out_features=Tin)
        # 对mutliheadAttention的输入做一个维度变换以保证他是2的幂次
        self.multiAttCNN1=nn.Conv1d(in_channels=N,out_channels=dmodel,kernel_size=1)
        self.multiAttCNN2=nn.Conv1d(in_channels=dmodel,out_channels=N,kernel_size=1)
        self.device=device
        # 设置gate门
        self.gate=nn.Linear(in_features=2*Tin,out_features=Tin)
        # 设置图卷积层捕获空间特征
        self.Gcn=GCN.GCN(T=Tin,trainMatrix1=trainMatrix1,trainMatrix2=trainMatrix2,device=device,tradGcn=tradGcn,dropout=dropout,hops=hops)
        self.spaceF=nn.Linear(2*Tin,Tin)
        # 设置TSCNN
        self.tsCNN=TSCNN(Tin=Tin,N=N,device=device)


    def forward(self,x,hidden,tXin):
        """

        :param x: 只含流量值的embed Tin*batch*N
        :param hidden: 此次输入的hidden:[Tin*batch*N]
        :param tXin: 加了timeEmbedding的x值：tXin:[Tin*batch*N]
        :return:
        """
        # 先捕获空间依赖
        gcnInput=torch.cat([x.permute(1,2,0).contiguous(),hidden.permute(1,2,0).contiguous()],dim=2) # batch*N*(2*Tin)
        gcnInput=F.relu(self.spaceF(gcnInput)) # batch*N*Tin
        gcnOutput=self.Gcn(gcnInput) # Tin*batch*N
        # 捕获时间依赖
        f2Input=torch.cat([hidden,tXin],dim=0) # (2Tin)*batch*N
        key=F.relu(self.f2(f2Input.permute(1,2,0).contiguous())) # batch*N*Tin
        key=self.multiAttCNN1(key) # batch*dmodel*Tin
        key=key.permute(2,0,1).contiguous() # Tin*batch*dmodel

        f1Input=torch.cat([hidden,tXin],dim=0) # (2Tin)*batch*N
        query=F.relu(self.f1(f1Input.permute(1,2,0).contiguous())) # batch*N*Tin
        query=self.multiAttCNN1(query) # batch*dmodel*Tin
        query=query.permute(2,0,1).contiguous() # Tin*batch*dmodel

        value=F.relu(self.f3(hidden.permute(1,2,0).contiguous())) # batch*N*Tin
        value=self.multiAttCNN1(value) # batch*dmodel*Tin
        value=value.permute(2,0,1).contiguous() # Tin*batch*dmodel

        # 做attention
        atten_mask=GcnEncoderCell.generate_square_subsequent_mask(value.size(0)).to(self.device)
        atten_output,atten=self.temporalAttention.forward(query=query,key=key,value=value,attn_mask=atten_mask) # Tin*batch*dmodel
        atten_output=atten_output.permute(1,2,0).contiguous() # batch*dmodel*Tin
        atten_output=self.multiAttCNN2(atten_output) # batch*N*Tin

        # tsCNN
        tsCnnOutput=self.tsCNN(hidden.permute(1,2,0).contiguous()) # batch*N*Tin

        # 做gate
        gcnOutput=gcnOutput.permute(1,2,0).contiguous() # batch*N*Tin
        finalHidden=torch.sigmoid(tsCnnOutput*torch.sigmoid(gcnOutput+atten_output)+gcnOutput*torch.sigmoid(tsCnnOutput+atten_output)+atten_output*torch.sigmoid(tsCnnOutput+gcnOutput))

        return finalHidden.permute(2,0,1).contiguous() # Tin*batch*N

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class GcnEncoder(nn.Module):
    def __init__(self,num_embedding,embedding_dim,N, trainMatrix1, trainMatrix2,hops,device,tradGcn,
                 dropout,dmodel,num_heads,Tin,encoderBlocks):
        super(GcnEncoder, self).__init__()
        self.encoderBlock=nn.ModuleList()
        for i in range(encoderBlocks):
            self.encoderBlock.append(GcnEncoderCell(N=N,trainMatrix1=trainMatrix1,trainMatrix2=trainMatrix2,hops=hops,device=device,
                                           tradGcn=tradGcn,dropout=dropout,dmodel=dmodel,num_heads=num_heads,Tin=Tin))
        self.timeEmbed=TE.timeEmbedding(num_embedding=num_embedding,embedding_dim=embedding_dim,dropout=dropout)
        self.device=device
        self.encoderBlocks=encoderBlocks
        self.bn=nn.ModuleList()
        for i in range(encoderBlocks):
            self.bn.append(nn.BatchNorm1d(num_features=N))

    def forward(self,x,tx,ty):
        """

        :param x: 流量数据:[batch*T*N]
        :param tx: 时间数据:[batch*T]
        :return:
        """
        x=x.permute(1,0,2).contiguous() # T*batch*N
        tx=self.timeEmbed(tx) # batch*T*N
        tx=tx.permute(1,0,2).contiguous() # T*batch*N
        ty=self.timeEmbed(ty) # batch*T*N

        tXin=x+tx
        hidden=x.clone()
        skip=0
        for i in range(self.encoderBlocks):
            hidden=self.encoderBlock[i].forward(x=x,hidden=hidden,tXin=tXin) # Tin*batch*N
            skip=skip+hidden
            hidden=self.bn[i](hidden.permute(1,2,0).contiguous()) # batch*N*Tin
            hidden=hidden.permute(2,0,1).contiguous()
        return skip,ty


class GcnDecoder(nn.Module):
    def __init__(self,dmodel, cnn_in_channels, cnn_out_channels, nhead,
                 num_layers, dropout, device,Tout,Tin,num_embedding):
        super(GcnDecoder, self).__init__()
        # timeEmbedding
        self.timeEmbed=TE.timeEmbedding(num_embedding=num_embedding,embedding_dim=cnn_in_channels,dropout=dropout)
        # encoder
        self.encoderLayer = nn.TransformerEncoderLayer(d_model=dmodel, nhead=nhead, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoderLayer, num_layers=num_layers)
        self.positionEmbedding = PE.PositionalEncoding(d_model=dmodel, dropout=dropout)
        # CNN
        self.decoderCNN1=nn.Conv1d(in_channels=cnn_in_channels,out_channels=cnn_out_channels,kernel_size=1)
        self.decoderCNN2=nn.Conv1d(in_channels=cnn_out_channels,out_channels=cnn_in_channels,kernel_size=1)
        # predict
        self.predict=nn.Linear(Tin+Tout,Tout)

    def forward(self,x,ty):
        """

        :param x: # Tin*batch*N
        :param ty: batch*Tout*N
        :return:
        """
        # ty=self.timeEmbed(ty) # batch*Tout*N

        x=self.decoderCNN1(x.permute(1,2,0).contiguous()) # batch*dmodel*T

        x=x.permute(2,0,1).contiguous() # T*batch*dmodel
        x=self.positionEmbedding(x) # T*batch*dmodel
        x=self.encoder(x) # Tin*batch*dmodel
        x=x.permute(1,2,0).contiguous() # batch*dmodel*Tin

        x=self.decoderCNN2(x) # batch*N*Tin

        ty=ty.permute(0,2,1).contiguous() # batch*N*tout
        input=torch.cat([x,ty],dim=2) # batch*N*(Tin+Tout)
        output=self.predict(input) # batch*N*Tout
        return output


class TSCNN(nn.Module):
    def __init__(self,Tin,N,device):
        super(TSCNN, self).__init__()
        self.Tin=Tin
        self.N=N
        self.device=device
        self.CNN=nn.ModuleList()
        for i in range(Tin):
            self.CNN.append(nn.Conv2d(in_channels=1,out_channels=N,kernel_size=[N,i+1]))

    def forward(self,x):
        """
        :param x: batch*N*Tin
        :return: batch*N*Tin
        """
        y=torch.zeros_like(x).to(self.device)
        for i in range(self.Tin):
            input=x[...,0:i+1].clone() # batch*N*t
            output=self.CNN[i](input.unsqueeze(dim=1)) # batch*N*1*1
            output=output.squeeze(dim=3) # batch*N*1
            y[...,i]=output.squeeze(dim=2)
        return y



