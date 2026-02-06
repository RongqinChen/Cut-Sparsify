import os
import os.path as osp
import shutil
from ogb.utils import smiles2graph
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
import pandas as pd
from tqdm import tqdm
import torch

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


class PygPCQM4Mv2Dataset(InMemoryDataset):
    def __init__(self, root='dataset', smiles2graph=smiles2graph, transform=None, pre_transform=None):
        '''
            Pytorch Geometric PCQM4Mv2 dataset object
                - root (str): the dataset folder will be located at root/pcqm4m_kddcup2021
                - smiles2graph (callable): A callable function that converts a SMILES string into a graph object
                    * The default smiles2graph requires rdkit to be installed
        '''

        self.original_root = root
        self.smiles2graph = smiles2graph
        self.folder = osp.join(root, 'pcqm4m-v2')
        self.version = 1

        # Old url hosted at Stanford
        # md5sum: 65b742bafca5670be4497499db7d361b
        # self.url = f'http://ogb-data.stanford.edu/data/lsc/pcqm4m-v2.zip'
        # New url hosted by DGL team at AWS--much faster to download
        self.url = 'https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip'

        # check version and update if necessary
        if osp.isdir(self.folder) and (not osp.exists(osp.join(self.folder, f'RELEASE_v{self.version}.txt'))):
            print('PCQM4Mv2 dataset has been updated.')
            if input('Will you update the dataset now? (y/N)\n').lower() == 'y':
                shutil.rmtree(self.folder)

        super(PygPCQM4Mv2Dataset, self).__init__(self.folder, transform, pre_transform)

        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

        for key, idxs in self.get_idx_split().items():
            print(key, len(idxs))
        for key, idxs in self.get_idx_split2().items():
            print(key, len(idxs))

    @property
    def raw_file_names(self):
        return 'data.csv.gz'

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print('Stop download.')
            exit(-1)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, 'data.csv.gz'))
        smiles_list = data_df['smiles']
        homolumogap_list = data_df['homolumogap']

        print('Converting SMILES strings into graphs...')

        # double-check prediction target
        split_dict = self.get_idx_split()
        for key, ten in split_dict.items():
            split_dict[key] = set(ten.tolist())

        # assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['train']]))
        # assert(all([not torch.isnan(data_list[i].y)[0] for i in split_dict['valid']]))
        # assert(all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-dev']]))
        # assert(all([torch.isnan(data_list[i].y)[0] for i in split_dict['test-challenge']]))

        data_list = []
        invalid_idx_list = []
        logfile = open("pcqm4m.log", 'w')
        for i in tqdm(range(len(smiles_list))):
            data = Data()

            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]
            try:
                graph = self.smiles2graph(smiles)
            except Exception:
                print(f"{i}, {smiles}, {homolumogap}", file=logfile)

                data.__num_nodes__ = 0
                data.edge_index = torch.empty((2, 0), dtype=torch.int64)
                data.edge_attr = torch.empty((0, 3), dtype=torch.int64)
                data.x = torch.empty((0, 9), dtype=torch.int64)
                data.y = torch.Tensor([0])

                data_list.append(data)
                invalid_idx_list.append(i)
                continue

            assert (len(graph['edge_feat']) == graph['edge_index'].shape[1])
            assert (len(graph['node_feat']) == graph['num_nodes'])

            data.__num_nodes__ = int(graph['num_nodes'])
            data.edge_index = torch.from_numpy(graph['edge_index']).to(torch.int64)
            data.edge_attr = torch.from_numpy(graph['edge_feat']).to(torch.int64)
            data.x = torch.from_numpy(graph['node_feat']).to(torch.int64)
            data.y = torch.Tensor([homolumogap])

            data_list.append(data)

        logfile.close()

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)

        print('Saving...')
        torch.save((data, slices), self.processed_paths[0])

        for i in invalid_idx_list:
            for key, idxs in split_dict.items():
                if i in idxs:
                    idxs.remove(i)

        for key, idxs in split_dict.items():
            split_dict[key] = torch.tensor(sorted(idxs))

        torch.save(split_dict, osp.join(self.root, 'split_dict2.pt'))

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(torch.load(osp.join(self.root, 'split_dict.pt'), weights_only=False))
        return split_dict

    def get_idx_split2(self):
        return torch.load(osp.join(self.root, 'split_dict2.pt'), weights_only=False)


if __name__ == '__main__':
    dataset = PygPCQM4Mv2Dataset()
    print(dataset)
    print(dataset.data.edge_index)
    print(dataset.data.edge_index.shape)
    print(dataset.data.x.shape)
    print(dataset[100])
    print(dataset[100].y)
    print(dataset.get_idx_split())
