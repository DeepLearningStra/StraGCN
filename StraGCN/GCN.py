import sys
import time
import argparse
import os.path as osp
from scipy.sparse import *
from tqdm import *
import heapq

import torch
import torch.nn as nn
import torch.optim as optim
import GCN_ST
from dataset import *
import torch.nn.functional as F


datastr = ["cora", "citeseer", "pubmed", "ppi", "PROTEINS_full", "OVCAR-8H", "Yeast", "DD", "TWITTER-Real-Graph-Partial", "SW-620H", "amazon0505", "amazon0601", "artist", "com-amazon", "soc-BlogCatalog"]
dims = [1433, 3703, 500, 50, 29, 66, 74, 89, 1323, 66, 96, 96, 12, 96, 39]
classes1 = [7, 6, 3, 121, 2, 2, 2, 2, 2, 2, 22, 22, 12, 22, 39]
hidden = [16, 32, 64, 128, 256, 512, 1024]
hiss = 0
iss = 0
print(datastr[iss])

parser = argparse.ArgumentParser()

parser.add_argument("--dataDir", type=str, default="./osdi-ae-graphs", help="the path to data")
parser.add_argument("--dataname", type=str, default=datastr[iss], help="dataset name")
parser.add_argument("--dim", type=int, default=dims[iss], help="input embedding dimension")
parser.add_argument("--hidden", type=int, default=hidden[hiss], help="hidden dimension size")
parser.add_argument("--classes", type=int, default=classes1[iss], help="output classes size")

parser.add_argument("--epoches", type=int, default=200, help="epoch")
args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load dataset


# preadd data
path_string = datastr[iss]
A_rowPtr = np.load("./presumdata/" + path_string + "/A_rowPtr.npy", allow_pickle=True)
A_colIdx = np.load("./presumdata/" + path_string + "/A_colIdx.npy", allow_pickle=True)
A_values = np.load("./presumdata/" + path_string + "/A_values.npy", allow_pickle=True)
num_nodes = np.load("./presumdata/" + path_string + "/node_num.npy", allow_pickle=True)
AR = []
AC = []
AVa = []
offset = []
offtmp = 0
for i in range(7):
    offset.append(offtmp)
    offtmp += A_rowPtr[i][-1]
    for it in A_rowPtr[i]:
        AR.append(it)
    for it in A_colIdx[i]:
        AC.append(it)
    for it in A_values[i]:
        AVa.append(it)

offset = torch.tensor(offset, dtype=torch.int32).to(device)
AR = torch.tensor(AR, dtype=torch.int32).to(device)
AC = torch.tensor(AC, dtype=torch.int32).to(device)
AVa = torch.tensor(AVa, dtype=torch.float32).to(device)


    
    

# sparse and dense
# path = osp.join("./osdi-ae-graphs/", datastr[iss] + ".npz")
# graph_obj = np.load(path)
# src_li = graph_obj['src_li']
# dst_li = graph_obj['dst_li']
# num_edges = len(src_li)
# edge_index = np.stack([src_li, dst_li])
# val = [3.1] * num_edges

# indices = torch.tensor(edge_index).to(device)
# val = torch.tensor(val).to(device)
# size = (num_nodes, num_nodes)
# sparse_matrix = torch.sparse_coo_tensor(indices, val, size)

# scipy_coo = coo_matrix((val, edge_index), shape=(num_nodes, num_nodes))
# A_dense = torch.tensor(scipy_coo.toarray(), dtype=torch.float32).to(device)
# print(A_dense.shape)

# partition_row = torch.empty(0).to(device)
# partitions_list = torch.empty(0).to(device)
# max_p = 0
# num_p = 0


class FunctionLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        # dense
        # return torch.matmul(A_dense, torch.matmul(input, weight))
        
        # my
        return GCN_ST.forward(AR, AC, AVa, offset, input, weight)
        # return GCN_ST.forward(AR, AC, AVa, partition_row, partitions_list, max_p, num_p, offset, input, weight)
        
        # cusparse
        # xw = torch.matmul(input, weight)
        # return torch.sparse.mm(sparse_matrix, xw)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors

        # dense
        # inputt = input.t()
        # grad_weight = torch.matmul(inputt, torch.matmul(A_dense, grad_output))

        # my  
        grad_input, grad_weight = GCN_ST.backward(AR, AC, AVa, offset, grad_output, input, weight)
        # grad_weight = GCN_ST.backward(AR, AC, AVa, partition_row, partitions_list, max_p, num_p, offset, grad_output, input)

        # cusparse
        # inputt = input.t()
        # tmp = torch.sparse.mm(sparse_matrix, grad_output)
        # grad_weight = torch.matmul(inputt, tmp)
        return grad_input, grad_weight
        

        # return grad_input, grad_weight

class myLinear(nn.Module):
    def __init__(self, input_size, output_size):
        super(myLinear, self).__init__()
        self.w = nn.Parameter(torch.randn(input_size, output_size))
    def forward(self, x):
        return FunctionLinear.apply(x, self.w)


class myGCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(myGCN, self).__init__()
        self.fc1 = myLinear(input_size, hidden_size)
        self.fc2 = myLinear(hidden_size, output_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x

def train_model():
    
    net = myGCN(args.dim, args.hidden, args.classes).to(device)  

    # set loss
    criterion = nn.MSELoss()

    optimizer = optim.SGD(net.parameters(), lr=0.01)

    inputs = torch.randn(num_nodes, args.dim).to(device)
    targets = torch.randn(args.classes).to(device)

    # train
    # dry run
    for _ in range(10):
        net.train()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


    epoch = 200
    start = time.perf_counter()
    for _ in tqdm(range(epoch)):
        net.train()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    end = time.perf_counter()
    print('Time (ms): {:.3f}'.format((end-start)*1e3/epoch))


def test_small_dataset():

    A = torch.tensor([[0, 0, 0, 1, 1, 0],  
                      [0, 4, 0, 0, 0, 2],  
                      [0, 0, 3, 0, 0, 0], 
                      [1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 5, 0, 0]], dtype=torch.float32).to(device)
    
    offset = torch.tensor([0, 3, 6, 8, 10, 15, 18], dtype=torch.int32).to(device)
    AR = torch.tensor([0, 0, 1, 3, 0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1, 2, 0, 2, 4, 5, 0, 1, 2, 3, 0, 2, 4, 5], dtype=torch.int32).to(device)
    AC = torch.tensor([1, 0, 2, 0, 1, 0, 1, 2, 1, 0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 1, 2, 0], dtype=torch.int32).to(device)
    AVa = torch.tensor([5., 5., 3., 1., 1., 5., 4., 3., 1., 5., 1., 1., 4., 2., 3., 1., -4., -3., 1., 1., -1., 2., -5.], dtype=torch.float32).to(device)





    X = []
    for i in range(6):
        tmp = []
        for j in range(4):
            tmp.append(j + 2.4 + i)
        X.append(tmp)
    W = []
    for i in range(4):
        tmp = []
        for j in range(4):
            tmp.append(j + 1.4 + i)
        W.append(tmp)
    grad_out = []
    for i in range(6):
        tmp = []
        for j in range(4):
            tmp.append(j + 1.8 + i)
        grad_out.append(tmp)

    
    X = torch.tensor(X).to(device)
    W = torch.tensor(W).to(device)
    grad_out = torch.tensor(grad_out).to(device)
    XT = X.t()



    # tmp1 = GCN_ST.backward(AR, AC, AVa,
    #                       grad_out, X)
    # tmp2 = torch.matmul(XT, torch.matmul(A, grad_out))
    
    

    tmp1 = GCN_ST.forward(AR, AC, AVa, offset, X, W)
    tmp2 = torch.matmul(A, torch.matmul(X, W))
    
    print("tmp1")
    print(tmp1)
    print("tmp2")
    print(tmp2)
    





def test_spmm():

    path = osp.join("./osdi-ae-graphs/", datastr[iss] + ".npz")
    graph_obj = np.load(path)
    src_li = graph_obj['src_li']
    dst_li = graph_obj['dst_li']
    num_edges = len(src_li)
    edge_index = np.stack([src_li, dst_li])
    val = [3.1] * num_edges

    indices = torch.tensor(edge_index)
    val = torch.tensor(val)
    size = (num_nodes, num_nodes)
    sparse_matrix = torch.sparse_coo_tensor(indices, val, size).to(device)

    # dry run
    # args.dim
    X = torch.randn(num_nodes, dims[iss]).to(device)

    for _ in range(10):
        tmp = torch.sparse.mm(sparse_matrix, X)
    start = time.perf_counter()
    for i in range(200):
        tmp = torch.sparse.mm(sparse_matrix, X)
    end = time.perf_counter()
    print('cuSPARSE Time (ms): {:.3f}'.format((end-start)*1e3/200))

    
    # dry run
    for i in range(10):
        tmp = GCN_ST.spmmstra(AR, AC, AVa, offset, X)
    start = time.perf_counter()
    for i in range(200):
        tmp = GCN_ST.spmmstra(AR, AC, AVa, offset, X)
    end = time.perf_counter()
    print('SPMM_STRA Time (ms): {:.3f}'.format((end-start)*1e3/200))




if __name__ == "__main__":
    train_model()
    # test_small_dataset()
    # test_strassen_normal()

    # test_spmm()
    







