# protein_encoders.py
import torch
import esm

import os
import shutil
import argparse
import torch
import pandas as pd
import re
import torch.utils.tensorboard
from embed import *
# from utils.datasets import *
from embed.transforms import *
from embed.misc import *
from embed.train import *
from embed.early_stop import *

from torch_geometric.transforms.add_positional_encoding import AddLaplacianEigenvectorPE
from torch_geometric.nn import radius_graph, knn_graph
from embed.common import GaussianSmearing,Gaussian
from torch_geometric.utils import to_undirected

from embed.protein_ligand import PDBProtein
from embed.data import ProteinLigandData, torchify_dict

class SimpleProteinTokenizer:
    def __init__(self, max_len):
        self.max_len = max_len
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.aa_to_idx = {aa: i + 1 for i, aa in enumerate(self.amino_acids)}  # 0 = padding

    def __call__(self, seq):
        indices = [self.aa_to_idx.get(aa, 0) for aa in seq[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long)

class ESMEncoder:
    def __init__(self, max_len):
        self.max_len = max_len
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval()

    def __call__(self, seq):

        data = [(f"protein_{i}", seq) for i, seq in enumerate(seq)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[33])

        token_representations = results["representations"][33]

        embeddings = []
        for i, (_, seq) in enumerate(data):
            seq_len = len(seq)
            emb = token_representations[i, 1:seq_len + 1].cpu()
            embeddings.append(emb)

        return embeddings

class CProMG:
    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, seq):
        config = load_config(args.config)
        config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
        seed_all(config.train.seed)

        # Logging
        log_dir = get_new_log_dir(args.logdir, )
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        logger.info(args)
        logger.info(config)
        shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
        shutil.copytree('./models', os.path.join(log_dir, 'models'))

        # Transforms
        protein_featurizer = FeaturizeProteinAtom(config)
        residue_featurizer = FeaturizeProteinResidue(config)
        transform = Compose([
            protein_featurizer,
            residue_featurizer,
        ])


        # Model
        logger.info('Building model...')
        model = Transformer(
            config.model,
            protein_featurizer.feature_dim,
            config.train.num_props,
        ).to(args.device)

        # Optimizer and scheduler
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = get_optimizer(config.train.optimizer, model)
        scheduler = get_scheduler(config.train.scheduler, optimizer)

        # 加载模型
        checkpoint = torch.load(args.model, map_location='cuda:0')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        it = checkpoint['iteration']
        model.eval()

        #data
        aa_laplacian = AddLaplacianEigenvectorPE(k=config.model.encoder.lap_dim,attr_name='protein_aa_laplacian')
        atom_laplacian = AddLaplacianEigenvectorPE(k=config.model.encoder.lap_dim,attr_name='protein_atom_laplacian')
        distance_expansion = GaussianSmearing(stop=10, num_gaussians=2)
        aa_distance_expansion = GaussianSmearing(stop=5, num_gaussians=2)
        gaussian = Gaussian(sigma=15)
        aa_gaussian = Gaussian(sigma=30)


        pocket_dict = PDBProtein(args.input).to_dict_atom()
        residue_dict = PDBProtein(args.input).to_dict_residue()

        data = ProteinLigandData.from_protein_ligand_dicts(
            protein_dict=torchify_dict(pocket_dict),
            residue_dict=torchify_dict(residue_dict),
        )

        data.protein_filename = 'pocket1'
        edge_index = knn_graph(data.protein_pos, 8, flow='target_to_source')     
        edge_length = torch.norm(data.protein_pos[edge_index[0]] - data.protein_pos[edge_index[1]], dim=1)   
        edge_attr =  gaussian(edge_length)                 
        edge_index,edge_attr = to_undirected(edge_index,edge_attr,reduce='mean')  
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data = atom_laplacian(data)

        #data.protein_atom_laplacian = atom_laplacian(data.protein_element.size(0), edge_index, edge_attr)


        aa_edge_index = knn_graph(data.residue_center_of_mass, 30,flow='target_to_source')                     
        aa_edge_length = torch.norm(data.residue_center_of_mass[aa_edge_index[0]] -data.residue_center_of_mass[aa_edge_index[1]], dim=1)
        aa_edge_attr = aa_gaussian(aa_edge_length) 
        aa_edge_index,aa_edge_attr = to_undirected(aa_edge_index,aa_edge_attr,reduce='mean')
        data.edge_index = aa_edge_index
        data.edge_attr = aa_edge_attr

        data = aa_laplacian(data)


        transform(data)
        data.protein_element_batch = torch.zeros([len(data.protein_element)]).long()
        data.residue_amino_acid_batch = torch.zeros([len(data.residue_amino_acid)]).long()

        data.to(args.device)


        batch_size = 1
        num_beams = 20  
        topk = 1
        filename = data.protein_filename


        if config.train.num_props:
            prop = torch.tensor([config.generate.prop for i in range(batch_size*num_beams)],dtype = torch.float).to(args.device)
            assert prop.shape[-1] == config.train.num_props
            num = int(bool(config.train.num_props))
        else:
            num = 0
            prop = None
        beam_output = beam_embed(model, config.model.decoder.smiVoc, num_beams, 
                                        batch_size, config.model.decoder.tgt_len + num, topk, data, prop)
        return beam_output