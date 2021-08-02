import torch
import torch.nn as nn
import model.GCN as GCN
import model.timeEmbedding as TE
import torch.nn.functional as F
import model.PositionEmbedding as PE


class GcnEncoderCell(nn.Module):
    def __init__(self,N, trainMatrix1, trainMatrix2,hops,device,tradGcn,dropout,dmodel,num_heads):
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
        """
        super(GcnEncoderCell, self).__init__()
        self.Gcn=GCN.GCN(T=1,trainMatrix1=trainMatrix1,trainMatrix2=trainMatrix2,device=device,tradGcn=tradGcn,dropout=dropout,hops=hops)
        self.temporalAttention=nn.MultiheadAttention(embed_dim=dmodel,num_heads=num_heads,dropout=dropout)
        self.f2=nn.Linear(in_features=2,out_features=1)
        self.f1=nn.Linear(in_features=2,out_features=1)
        # 对mutliheadAttention的输入做一个维度变换以保证他是2的幂次
        self.multiAttCNN1=nn.Conv1d(in_channels=N,out_channels=dmodel,kernel_size=1)
        self.multiAttCNN2=nn.Conv1d(in_channels=dmodel,out_channels=N,kernel_size=1)
        self.device=device
        #
        self.gate=nn.Linear(in_features=2,out_features=1)

    def forward(self,x,hidden,tXin,tHidden):
        """

        :param x: 1*batch*N
        :param hidden: 1*batch*N
        :param tXin: 在此次时间之前的真实数据值（包括此次时间）tXin:[tBefore*batch*N]
        :param tHidden : 在此次时间之前已输出的hidden(不包括此次时间)tHidden:[tBefore-1*batch*N]
        :return:
        """
        # gcn提取空间特征
        spaceAttenX=self.Gcn(x.permute(1,2,0).contiguous()) # 1*batch*N
        # 做temporalAttention
        temporalInput=torch.zeros_like(tXin).to(self.device) # [tBefore*batch*N]
        for i in range(tXin.size(0)-1):
            f2input=torch.cat([tXin[i,...].unsqueeze(dim=0),tHidden[i,...].unsqueeze(dim=0)]) # 2*batch*N
            f2output=F.relu(self.f2(f2input.permute(1,2,0).contiguous()).permute(2,0,1).contiguous()) # 1*batch*N
            temporalInput[i]=f2output
        f1input=torch.cat([tXin[tXin.size(0)-1,...].unsqueeze(dim=0),hidden]) # 2*batch*N
        f1output=F.relu(self.f1(f1input.permute(1,2,0).contiguous()).permute(2,0,1).contiguous()) # 1*batch*N
        temporalInput[tXin.size(0)-1]=f1output # tBefore*batch*N
        # 做维度变换
        temporalInput=self.multiAttCNN1(temporalInput.permute(1,2,0).contiguous()) # batch*dmodel*tBefore
        temporalInput=temporalInput.permute(2,0,1).contiguous() # tBefore*batch*dmodel
        # 计算atten_mask以及得到temporalOutput:[tBefore*batch*dmodel]
        atten_mask=GcnEncoderCell.generate_square_subsequent_mask(sz=temporalInput.size(0)).to(self.device) #tBefore*tBefore
        temporalOutput,output_atten=self.temporalAttention.forward(query=temporalInput,key=temporalInput,value=temporalInput,attn_mask=atten_mask)
        # 做维度变换变回N
        temporalOutput=self.multiAttCNN2(temporalOutput.permute(1,2,0).contiguous())
        temporalOutput=temporalOutput.permute(2,0,1).contiguous() # tBefore*batch*N
        #
        temporalAttenT=temporalOutput[temporalOutput.size(0)-1,...].unsqueeze(dim=0) # 1*batch*N

        # 经过一个gate Fusion
        attenSum=torch.cat([spaceAttenX,temporalAttenT],dim=0) # 2*batch*N
        z=F.sigmoid(self.gate(attenSum.permute(1,2,0).contiguous())) # batch*N*1
        z=z.permute(2,0,1).contiguous() # 1*batch*N
        finalHidden=z*spaceAttenX+(1-z)*temporalAttenT # 1*batch*N

        return finalHidden

    @staticmethod
    def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class GcnEncoder(nn.Module):
    def __init__(self,num_embedding,embedding_dim,N, trainMatrix1, trainMatrix2,hops,device,tradGcn,
                 dropout,dmodel,num_heads):
        super(GcnEncoder, self).__init__()
        self.GcnEncoderCell=GcnEncoderCell(N=N,trainMatrix1=trainMatrix1,trainMatrix2=trainMatrix2,hops=hops,device=device,
                                           tradGcn=tradGcn,dropout=dropout,dmodel=dmodel,num_heads=num_heads)
        self.timeEmbed=TE.timeEmbedding(num_embedding=num_embedding,embedding_dim=embedding_dim,dropout=dropout)
        self.device=device

    def forward(self,x,tx):
        """

        :param x: 流量数据:[batch*T*N]
        :param tx: 时间数据:[batch*T]
        :return:
        """
        x=x.permute(1,0,2).contiguous() # T*batch*N
        tx=self.timeEmbed(tx) # batch*T*N
        tx=tx.permute(1,0,2).contiguous() # T*batch*N

        allHidden=torch.zeros_like(x).to(self.device) # T*batch*N
        hidden=torch.zeros(1,x.size(1),x.size(2)).to(self.device) # 1*batch*N
        embedX=x+tx # T*batch*N
        for i in range(x.size(0)):
            hidden=self.GcnEncoderCell(x[i].unsqueeze(dim=0),hidden,embedX[0:i+1,...],allHidden[0:i,...]) # 1*batch*N
            allHidden[i]=hidden
        return allHidden,hidden


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
        :param ty: batch*Tout
        :return:
        """
        ty=self.timeEmbed(ty) # batch*Tout*N

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


