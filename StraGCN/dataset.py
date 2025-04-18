#!/usr/bin/env python3
import torch
import numpy as np
import time
import os.path as osp
import os

from scipy.sparse import *

def func(x):
    '''
    node degrees function
    '''
    if x > 0:
        return x
    else:
        return 1

class Dataset(torch.nn.Module):
    """
    data loading for more graphs
    """
    def __init__(self, path, dim, num_class, load_from_txt=False, verbose=False):
        super(Dataset, self).__init__()

        self.nodes = set()

        self.load_from_txt = load_from_txt
        self.num_nodes = 0
        self.num_features = dim 
        self.num_classes = num_class
        self.edge_index = None
        
        self.reorder_flag = False
        self.verbose_flag = verbose

        self.avg_degree = -1
        self.avg_edgeSpan = -1

        self.init_edges(path)
        self.init_embedding(dim)
        self.init_labels(num_class)

        train = 1
        val = 0.3
        test = 0.1
        self.train_mask = [1] * int(self.num_nodes * train) + [0] * (self.num_nodes  - int(self.num_nodes * train))
        self.val_mask = [1] * int(self.num_nodes * val)+ [0] * (self.num_nodes  - int(self.num_nodes * val))
        self.test_mask = [1] * int(self.num_nodes * test) + [0] * (self.num_nodes  - int(self.num_nodes * test))
        self.train_mask = torch.BoolTensor(self.train_mask).cuda()
        self.val_mask = torch.BoolTensor(self.val_mask).cuda()
        self.test_mask = torch.BoolTensor(self.test_mask).cuda()

    def init_edges(self, path):

        # loading from a txt graph file
        if self.load_from_txt:
            fp = open(path, "r")
            src_li = []
            dst_li = []
            start = time.perf_counter()
            for line in fp:
                src, dst = line.strip('\n').split()
                src, dst = int(src), int(dst)
                src_li.append(src)
                dst_li.append(dst)
                self.nodes.add(src)
                self.nodes.add(dst)
            
            # self.g.add_edges(src_li, dst_li)
            self.num_edges = len(src_li)
            self.num_nodes = max(self.nodes) + 1
            self.edge_index = np.stack([src_li, dst_li])

            dur = time.perf_counter() - start
            if self.verbose_flag:
                print("# Loading (txt) {:.3f}s ".format(dur))

        # loading from a .npz graph file
        else: 
            if not path.endswith('.npz'):
                raise ValueError("graph file must be a .npz file")

            start = time.perf_counter()
            graph_obj = np.load(path)
            src_li = graph_obj['src_li']
            dst_li = graph_obj['dst_li']

            self.num_nodes = graph_obj['num_nodes']
            # self.g.add_edges(src_li, dst_li)
            self.num_edges = len(src_li)
            self.edge_index = np.stack([src_li, dst_li])
            dur = time.perf_counter() - start
            if self.verbose_flag:
                print("# Loading (npz)(s): {:.3f}".format(dur))
        
        self.avg_degree = self.num_edges / self.num_nodes
        self.avg_edgeSpan = np.mean(np.abs(np.subtract(src_li, dst_li)))

        if self.verbose_flag:
            print('# nodes: {}'.format(self.num_nodes))
            print("# avg_degree: {:.2f}".format(self.avg_degree))
            print("# avg_edgeSpan: {}".format(int(self.avg_edgeSpan)))

        # Build graph CSR.
        self.val = [3.0] * self.num_edges
        start = time.perf_counter()
        scipy_coo = coo_matrix((self.val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
        # self.A_dense = scipy_coo.toarray()
        scipy_csr = scipy_coo.tocsr()
        build_csr = time.perf_counter() - start

        if self.verbose_flag:
            print("# Build CSR after reordering (s): {:.3f}".format(build_csr))

        self.column_index = torch.IntTensor(scipy_csr.indices)
        self.row_pointers = torch.IntTensor(scipy_csr.indptr)

        # Get degrees array.
        degrees = (self.row_pointers[1:] - self.row_pointers[:-1]).tolist()
        self.degrees = torch.sqrt(torch.FloatTensor(list(map(func, degrees)))).cuda()

    def init_embedding(self, dim):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes, dim).cuda()
    
    def init_labels(self, num_class):
        '''
        Generate the node label.
        Called from __init__.
        '''
        self.y = torch.ones(self.num_nodes).long().cuda()


def split_CSR(row_pointers, column_index, values):
    A11_rowPtr = []
    A11_colIdx = []
    A11_values = []
    A12_rowPtr = []
    A12_colIdx = []
    A12_values = []
    A21_rowPtr = []
    A21_colIdx = []
    A21_values = []
    A22_rowPtr = []
    A22_colIdx = []
    A22_values = []

    nnzA11 = 0
    nnzA12 = 0
    nnzA21 = 0
    nnzA22 = 0

    A11_rowPtr.append(0)
    A12_rowPtr.append(0)
    A21_rowPtr.append(0)
    A22_rowPtr.append(0)

    non_zeros = row_pointers[-1]
    row = len(row_pointers) - 1
    print("row: ", row)
    half = int((row + 1) / 2)
    print("half", half)
    for i in range(row):
        start = row_pointers[i]
        end = row_pointers[i+1]
        # print(i)
        for j in range(start, end):
            col = column_index[j]
            val = values[j]
            # print(col)
            if i < half and col < half:
                nnzA11 += 1
                A11_colIdx.append(col)
                A11_values.append(val)
            elif i < half and col >= half :
                nnzA12 += 1
                A12_colIdx.append(col-half)
                A12_values.append(val)
            elif i >= half and col < half:
                nnzA21 += 1
                A21_colIdx.append(col)
                A21_values.append(val)
            else:
                nnzA22 += 1
                A22_colIdx.append(col-half)
                A22_values.append(val)
        if i < half:
            A11_rowPtr.append(A11_rowPtr[i]+nnzA11)
            A12_rowPtr.append(A12_rowPtr[i]+nnzA12)
            nnzA11 = 0
            nnzA12 = 0
        else:
            A21_rowPtr.append(A21_rowPtr[i-half]+nnzA21)
            A22_rowPtr.append(A22_rowPtr[i-half]+nnzA22)
            nnzA21 = 0
            nnzA22 = 0
    if row % 2 != 0:
        A21_rowPtr.append(A21_rowPtr[-1])
        A22_rowPtr.append(A22_rowPtr[-1])
    print(len(A11_rowPtr), len(A12_rowPtr), len(A21_rowPtr), len(A22_rowPtr))
    print(len(A11_colIdx), len(A12_colIdx), len(A21_colIdx), len(A22_colIdx))
    return A11_rowPtr, A11_colIdx, A11_values, A12_rowPtr, A12_colIdx, A12_values, A21_rowPtr, A21_colIdx, A21_values, A22_rowPtr, A22_colIdx, A22_values



def preAdd(A11_rowPtr, A11_colIdx, A11_values, A12_rowPtr, A12_colIdx, A12_values, A21_rowPtr, A21_colIdx, A21_values, A22_rowPtr, A22_colIdx, A22_values, m):
    # print("A11 type: ", type(A11_rowPtr[0]), type(A11_colIdx[0]), type(A11_values[0]))
    mat1 = csr_matrix((A11_values, A11_colIdx, A11_rowPtr), shape=(m, m))
    mat2 = csr_matrix((A22_values, A22_colIdx, A22_rowPtr), shape=(m, m))
    M1 = mat1 + mat2

    mat1 = csr_matrix((A21_values, A21_colIdx, A21_rowPtr), shape=(m, m))
    mat2 = csr_matrix((A22_values, A22_colIdx, A22_rowPtr), shape=(m, m))
    M2 = mat1 + mat2

    mat1 = csr_matrix((A11_values, A11_colIdx, A11_rowPtr), shape=(m, m))
    mat2 = csr_matrix((A12_values, A12_colIdx, A12_rowPtr), shape=(m, m))
    M5 = mat1 + mat2

    mat1 = csr_matrix((A21_values, A21_colIdx, A21_rowPtr), shape=(m, m))
    mat2 = csr_matrix((A11_values, A11_colIdx, A11_rowPtr), shape=(m, m))
    M6 = mat1 - mat2

    mat1 = csr_matrix((A12_values, A12_colIdx, A12_rowPtr), shape=(m, m))
    mat2 = csr_matrix((A22_values, A22_colIdx, A22_rowPtr), shape=(m, m))
    M7 = mat1 - mat2

    return M1, M2, M5, M6, M7

    



if __name__ == '__main__':
    datastr = ["cora", "citeseer", "pubmed", "ppi", "PROTEINS_full", "OVCAR-8H", "Yeast", "DD", "TWITTER-Real-Graph-Partial", "SW-620H", "amazon0505", "amazon0601", "artist", "com-amazon", "soc-BlogCatalog"]
    path_string = datastr[14]
    print(path_string)
    path = osp.join("./osdi-ae-graphs/", path_string + ".npz")
    dataset = Dataset(path, 16, 10)

    row_coo = list(dataset.edge_index[0])
    col_coo = list(dataset.edge_index[1])
    print("row: ", len(row_coo))
    print("col: ", len(col_coo))

    num_nodes = dataset.num_nodes
    halfm = int((num_nodes + 1) / 2)

    print("number nodes: ", num_nodes)



    row_ptr = np.array(dataset.row_pointers, dtype=object)
    # print(row_ptr.shape)
    col_index = np.array(dataset.column_index, dtype=object)
    # print(col_index.shape)

    
    degrees = row_ptr[1:] - row_ptr[:-1]
    print("degrees: ", degrees.shape)
    mask = degrees > 0 
    deg_inv = np.zeros_like(degrees, dtype=np.float32)
    deg_inv[mask] = np.power(degrees[mask], -0.5)
    print("deg_inv: ", deg_inv.shape)
    values = np.array(deg_inv[row_coo] * deg_inv[col_coo], dtype=np.float32)

    A11_rowPtr, A11_colIdx, A11_values, A12_rowPtr, A12_colIdx, A12_values, A21_rowPtr, A21_colIdx, A21_values, A22_rowPtr, A22_colIdx, A22_values = split_CSR(row_ptr, col_index, values)

    print("split over")
    M1, M2, M5, M6, M7 = preAdd(A11_rowPtr, A11_colIdx, A11_values, A12_rowPtr, A12_colIdx, A12_values, A21_rowPtr, A21_colIdx, A21_values, A22_rowPtr, A22_colIdx, A22_values, halfm)
    
    print("add over")
    A_rowPtr = []
    A_colIdx = []
    A_values = []
    M1_rowPtr = M1.indptr
    M2_rowPtr = M2.indptr
    M3_rowPtr = A11_rowPtr
    M4_rowPtr = A22_rowPtr
    M5_rowPtr = M5.indptr
    M6_rowPtr = M6.indptr
    M7_rowPtr = M7.indptr
    print("save begin")
    
    A_rowPtr.append(M1.indptr)
    A_rowPtr.append(M2.indptr)
    A_rowPtr.append(A11_rowPtr)
    A_rowPtr.append(A22_rowPtr)
    A_rowPtr.append(M5.indptr)
    A_rowPtr.append(M6.indptr)
    A_rowPtr.append(M7.indptr)

    A_colIdx.append(M1.indices)
    A_colIdx.append(M2.indices)
    A_colIdx.append(A11_colIdx)
    A_colIdx.append(A22_colIdx)
    A_colIdx.append(M5.indices)
    A_colIdx.append(M6.indices)
    A_colIdx.append(M7.indices)

    A_values.append(M1.data)
    A_values.append(M2.data)
    A_values.append(A11_values)
    A_values.append(A22_values)
    A_values.append(M5.data)
    A_values.append(M6.data)
    A_values.append(M7.data)

    A_rowPtr = np.array(A_rowPtr, dtype=object)
    A_colIdx = np.array(A_colIdx, dtype=object)
    A_values = np.array(A_values, dtype=object)

    os.makedirs("./presumdata/" + path_string, exist_ok=True)
    np.save("./presumdata/" + path_string + "/A_rowPtr.npy", A_rowPtr)
    np.save("./presumdata/" + path_string + "/A_colIdx.npy", A_colIdx)
    np.save("./presumdata/" + path_string + "/A_values.npy", A_values)
    np.save("./presumdata/" + path_string + "/node_num.npy", num_nodes)
    print("saved!!")






