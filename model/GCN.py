import torch
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self,T, trainMatrix1, trainMatrix2, device, hops,dmodel, tradGcn=False, dropout=0.1):
        super().__init__()
        self.T=T
        self.trainMatrix1 = trainMatrix1
        self.trainMatrix2 = trainMatrix2
        self.device = device
        self.hops = hops
        self.tradGcn = tradGcn
        self.dropout = nn.Dropout(p=dropout)
        # 运用传统图卷积
        self.tradGcn = tradGcn
        if tradGcn:
            self.tradGcnW = nn.ModuleList()
            for i in range(self.hops):
                self.tradGcnW.append(nn.Linear(self.T, self.T))
        else:
            self.gcnLinear = nn.Linear(self.T * (self.hops + 1), self.T)

    def forward(self,X):
        """

        :param X: batch*dmodel*node*T
        :return: Hout:[batch*dmodel*node*T]
        """
        adjMat = F.relu(torch.mm(self.trainMatrix1, self.trainMatrix2))
        adjMat = F.softmax(adjMat, dim=1)
        A=adjMat

        H = list()
        H.append(X)
        Hbefore = X  # X batches*dmodel*node*T
        # 开始图卷积部分
        if self.tradGcn == False:
            for k in range(self.hops):
                Hnow=torch.einsum("nk,bdkt->bdnt", (A, Hbefore)) # batch*dmodel*node*T
                Hnow=torch.sigmoid(X+Hnow)*torch.tanh(X+Hnow)
                H.append(Hnow)
                Hbefore = Hnow
            H = torch.cat(H, dim=3)  # batch*dmodel*N*(T*(hops+1))
            Hout = self.gcnLinear(H)  # batch*dmodel*N*T
            Hout = self.dropout(Hout) # batch*dmodel*N*T
        else:
            Hout = Hbefore
            for k in range(self.hops):
                Hout = torch.einsum("nk,bdkt->bdnt", (A, Hout))  # batch*N*T A*H
                Hout = self.tradGcnW[k](Hout)  # batch*dmodel*N*T A*H*W
                Hout = F.relu(Hout)  # relu(A*H*w)
            Hout = self.dropout(Hout)  # batch*dmodel*N*T
        return Hout
