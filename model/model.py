# 此处定义自己的模型
import torch.nn as nn
import torch
import model.GmanGcn as GG

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
        # 定义GCNEncoder
        self.GcnEncoder=GG.GcnEncoder(num_embedding=args.num_embedding,embedding_dim=N,N=N,trainMatrix1=self.trainMatrix1,
                                      trainMatrix2=self.trainMatrix2,hops=self.hops,device=device,tradGcn=args.tradGcn,
                                      dropout=args.dropout,dmodel=args.dmodel,num_heads=args.head,Tin=T,encoderBlocks=args.encoderBlocks,
                                      M=args.M)
        self.GcnDecoder=GG.GcnDecoder(dmodel=args.dmodel,cnn_in_channels=N,cnn_out_channels=args.dmodel,
                                      nhead=args.head,num_layers=args.transformerLayers,dropout=args.dropout,
                                      device=device,Tout=outputT,Tin=T,num_embedding=args.num_embedding)
        # predict layer
        self.predict=nn.Linear(in_features=outputT+self.arSize,out_features=outputT)

    def forward(self, X, Y, teacher_forcing_ratio):
        """

        :param X: 输入数据，X:batch*node*T*2
        :param Y: 真实值，Y:outputT*batch*node*2
        :return: 输出数据: Y:batch*node*T
        """
        vx=X[...,0] # batch*node*Tin 表示X车流量
        tx=X[:,0,:,1] # batch*T 表示输入X的时间index
        Y=Y.permute(1,2,0,3).contiguous() # batch*node*Tout*2
        ty=Y[:,0,:,1] # batch*T 表示Y的时间index
        # 开始encoder
        output,ty=self.GcnEncoder(vx.permute(0,2,1).contiguous(),tx,ty) # T*batch*N
        result=self.GcnDecoder(output,ty) # batch*N*Tout
        result=torch.cat([result,vx[...,-self.arSize:]],dim=2)
        result=self.predict(result)


        return result
