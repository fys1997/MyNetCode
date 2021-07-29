# 此处定义自己的模型
import torch.nn as nn
import torch
import model.SequenceToSequence as S2S
import model.Transformer as tr
import model.TCN as tcn
import model.GCN as gcn


class mixNet(nn.Module):
    def __init__(self, args, device, T, N, outputT):
        '''

        :param args: 一些设置的参数
        :param data: 训练的数据
        '''
        super(mixNet, self).__init__()
        self.N = N
        self.T = T
        self.outputT = outputT  # output sequence length
        self.device = device
        # 此处m指预训练矩阵的维度数
        self.m = args.M
        # 节点数
        self.num_nodes = N
        self.trainMatrix1 = nn.Parameter(torch.randn(self.num_nodes, self.m).to(device), requires_grad=True).to(device)
        self.trainMatrix2 = nn.Parameter(torch.randn(self.m, self.num_nodes).to(device), requires_grad=True).to(device)
        # hops的值
        self.hops = args.hops
        # AR模块窗口的大小
        self.arSize = args.arSize
        self.dropout = nn.Dropout(p=args.dropout)
        # 运用图卷积
        self.GCN = gcn.GCN(T=T, trainMatrix1=self.trainMatrix1, trainMatrix2=self.trainMatrix2, device=device,
                           hops=self.hops, tradGcn=args.tradGcn, dropout=args.dropout)
        # 定义 Seq2Seq GRU部分
        self.seq2seq = S2S.Seq2Seq(inputSize=self.N, hiddenSize=self.N, n_layers=args.n_layers, dropout=args.dropout,
                                   device=device,Tout=outputT,Tin=T,trainMatrix1=self.trainMatrix1,trainMatrix2=self.trainMatrix2,
                                   hops=self.hops,tradGcn=args.tradGcn)
        # self.gruLinear = nn.Linear(self.outputT, self.outputT)
        # 定义AutoRegressive部分
        if (self.arSize > 0):
           self.tcn=tcn.TCN(Tin=self.T,Tout=self.outputT,dropout=args.dropout,channels=self.N)
        # 定义transformer
        self.transformer = tr.Transformer(dmodel=args.dmodel, cnn_in_channels=self.N, cnn_out_channels=args.dmodel,
                                          nhead=args.head, num_layers=args.transformerLayers, dropout=args.dropout,
                                          device=device,Tout=outputT)
        # self.transformerLinear = nn.Linear(self.outputT, self.outputT)
        # 此处定义将X维度变为1
        self.in_dims=args.in_dims
        self.dimCNN1=nn.Conv2d(in_channels=self.in_dims,out_channels=1,kernel_size=(1,1))
        self.dimCNN2=nn.Conv2d(in_channels=self.in_dims,out_channels=64,kernel_size=(1,1))
        self.dimCNN3=nn.Conv2d(in_channels=64,out_channels=256,kernel_size=(1,1))
        self.dimCNN4=nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(1,1))
        self.dimCNN5=nn.Conv2d(in_channels=512,out_channels=256,kernel_size=(1,1))
        self.dimCNN6 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=(1, 1))
        self.dimCNN7 = nn.Conv2d(in_channels=64, out_channels=self.in_dims, kernel_size=(1, 1))
        # 定义batchNorm
        self.batchnormS2S=nn.BatchNorm2d(num_features=1)
        self.batchnormTran=nn.BatchNorm2d(num_features=1)
        self.predict=nn.Linear(in_features=3*outputT+self.arSize if self.arSize>0 else 2*outputT+self.arSize,out_features=outputT)

    def forward(self, X, Y, teacher_forcing_ratio):
        """

        :param X: 输入数据，X:batch*node*T*2
        :param Y: 真实值，Y:outputT*batch*node*2
        :return: 输出数据: Y:batch*node*T
        """
        # 原数据记录在此做tcn
        # tcnX=X
        # 先维度变换
        x=X.clone()
        # 先图卷积
        Hout=self.GCN(X[...,0]) # Tin*batch*N
        x[...,0]=Hout.permute(1,2,0).contiguous() # x:[batch*N*Tin*2]
        # 维度变化测试
        x=self.dimCNN2(x)
        x=self.dimCNN3(x)
        x=self.dimCNN4(x)
        x=self.dimCNN5(x)
        x=self.dimCNN6(x)
        x=self.dimCNN7(x)
        x=self.dimCNN1(x.permute(0,3,1,2).contiguous()).squeeze(dim=1) # batch*N*Tin

        x=x.permute(2,0,1).contiguous() # Tin*batch*N
        # Seq2Seq GRU部分
        y1 = self.seq2seq(x, Y, teacher_forcing_ratio=teacher_forcing_ratio)  # outputT*batch*N
        y1 = y1.permute(1, 2, 0).contiguous()  # batch*N*outputT
        y1 = self.batchnormS2S(y1.unsqueeze(dim=1)).squeeze(dim=1)
        y1 = torch.relu(y1) # batch*N*outputT
        # y1 = self.gruLinear(y1)

        # Transformer部分
        y2 = self.transformer(x, Y[...,1], teacher_forcing_ratio=teacher_forcing_ratio)  # batch*N*outputT
        y2 = self.batchnormTran(y2.unsqueeze(dim=1)).squeeze(dim=1)
        y2=torch.relu(y2) # batch*N*outputT
        # y2 = self.transformerLinear(y2)  # batch*N*outputT

        # TCN部分
        y3=0
        if self.arSize>0:
            y3 = self.tcn(x.permute(1,2,0).contiguous()) # batch*N*outputT
        y=torch.cat([y1,y2,y3,X[:,:,-self.arSize:,0]],dim=2)
        y=self.predict(y)
        return y
