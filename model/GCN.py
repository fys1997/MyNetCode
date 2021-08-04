import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self,T, trainMatrix1, trainMatrix2, device, hops, tradGcn=False, dropout=0.1):
        super().__init__()
        self.T=T
        self.trainMatrix1 = trainMatrix1
        self.trainMatrix2 = trainMatrix2
        self.device = device
        self.hops = hops
        self.tradGcn = tradGcn
        self.dropout = nn.Dropout(p=dropout)
        # 门
        self.gate=nn.Linear(in_features=2*T,out_features=T)
        # 运用传统图卷积
        self.tradGcn = tradGcn
        if tradGcn:
            self.tradGcnW = nn.ModuleList()
            for i in range(self.hops):
                self.tradGcnW.append(nn.Linear(self.T, self.T))
        else:
            self.gcnLinear = nn.Linear(self.T * (self.hops + 1), self.T)
        # 设置low mid high filter的p与a
        self.pLow=nn.Parameter(torch.rand(1).to(device)).to(device)
        self.aLow=nn.Parameter(torch.rand(1).to(device)).to(device)
        self.pMiddle=nn.Parameter(torch.rand(1).to(device)).to(device)
        self.aMiddle=nn.Parameter(torch.rand(1).to(device)).to(device)
        self.pHigh=nn.Parameter(torch.rand(1).to(device)).to(device)
        self.aHigh=nn.Parameter(torch.rand(1).to(device)).to(device)

    def forward(self,X):
        """

        :param X: batch*node*T
        :return: Hout:[T*batch*N]
        """
        adjMat = torch.mm(self.trainMatrix1, self.trainMatrix2)
        adjMat = F.softmax(adjMat, dim=1)
        I=torch.ones(adjMat.shape[0]).to(self.device)
        # 计算邻接矩阵的度矩阵
        rowsum = torch.sum(adjMat, dim=1)  # 每一行相加求和
        degreeMat = torch.pow(rowsum, -0.5)
        degreeMat[torch.isinf(degreeMat)] = 0

        degreeMat = torch.diag(degreeMat)
        # D^-1/2*A*D^-1/2
        A = torch.mm(torch.mm(degreeMat, adjMat), degreeMat)

        H = list()
        H.append(X)
        Hbefore = X  # X batches*node*T
        # 开始图卷积部分
        if self.tradGcn == False:
            for k in range(self.hops):
                # low filter
                HnowLow=torch.einsum("ik,bkj->bij", (self.pLow*(self.aLow*A+(1-self.aLow)*I), Hbefore)) # batch*node*T
                # middle filter
                HnowMiddle=torch.einsum("ik,bkj->bij", (self.pMiddle*(torch.pow(A,2)-self.aMiddle*I), Hbefore)) # batch*node*T
                # High filter
                HnowHigh=torch.einsum("ik,bkj->bij", (self.pHigh*(-self.aHigh*A+(1-self.aHigh)*I), Hbefore)) # batch*node*T
                # gate
                Hnow=torch.sigmoid(HnowLow*torch.sigmoid(HnowMiddle+HnowHigh)+HnowMiddle*torch.sigmoid(HnowLow+HnowHigh)
                                   +HnowHigh*torch.sigmoid(HnowLow+HnowMiddle)) # batch*node*T

                # Hnow=torch.einsum("ik,bkj->bij", (A, Hbefore)) # batch*node*T
                gateInput=torch.cat([X,Hnow],dim=2) # batch*node*2T
                z=torch.sigmoid(self.gate(gateInput)) # batch*node*T
                Hnow=z*Hnow+(1-z)*X # batch*node*T
                H.append(Hnow)
                Hbefore = Hnow
            H = torch.cat(H, dim=2)  # batch*N*(T*(hops+1))
            Hout = self.gcnLinear(H)  # batch*N*T
            Hout = self.dropout(Hout).permute(2, 0, 1).contiguous()  # T*batch*N
        else:
            Hout = Hbefore
            for k in range(self.hops):
                Hout = torch.einsum("ik,bkj->bij", (A, Hout))  # batch*N*T A*H
                Hout = self.tradGcnW[k](Hout)  # batch*N*T A*H*W
                Hout = F.relu(Hout)  # relu(A*H*w)
            Hout = self.dropout(Hout).permute(2, 0, 1).contiguous()  # T*batch*N
        return Hout
