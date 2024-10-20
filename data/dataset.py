from torch.utils.data import Dataset
import json 
import numpy as np 
import pandas as pd
import torch 
from tqdm import tqdm
import ast
from Bio.PDB import PDBParser

AMINO_ACIDS = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


class RLDIFDataset(Dataset):
    def __init__(
        self,
        config,
    ):
        super().__init__()

        self.config = config
        self.chosen_ids = None
        self.test_ids = None
        self.RS = []

        self.pifold = config.pifold
        # redis version
        index_for_redis = 0
        if config.dataset_name == 'ts50':
            data_paths = "./data/raw_data/ts50.json"
        elif config.dataset_name == 'ts500': 
            data_paths =  "./data/raw_data/ts500.json"
        elif config.dataset_name == 'casp15':
            data_paths = "./data/raw_data/casp15.json"
        elif config.dataset_name == 'cath_test':
            data_paths = "./data/raw_data/cath_test.csv"
        elif config.dataset_name == 'cath_single':
            data_paths = "./data/raw_data/cath_single_chain.csv"
        elif config.dataset_name == 'cath_sub100':
            data_paths = "./data/raw_data/cath_sub_100.csv"
        elif config.dataset_name == 'custom_pdbs':
            data_paths = config.custom_pdb_path
        else:
            raise Exception("Invalid dataset name")

        dp = data_paths
        if 'cath' in config.dataset_name:
            x = pd.read_csv(dp)
            iterator = tqdm(x.iterrows())
        elif config.dataset_name == 'custom_pdbs':
            x = [i.rstrip() for i in open(dp, 'r').readlines()]
            iterator = tqdm(x)
        else:
            x = json.load(open(dp, 'r'))
            iterator = tqdm(x)

        num = 0
        for block in iterator:
            if 'ts50' in config.dataset_name or 'ts500' in config.dataset_name:
                artifical_block = {}
                artifical_block['name'] = block['name']
                artifical_block['seq'] = block['seq']
                if len(block['seq']) < 800:
                    artifical_block['coords'] = {}
                    coords = np.array(block['coords'])
                    artifical_block['coords']['CA'] = coords[:,1,:]
                    artifical_block['coords']['C'] = coords[:,2,:]
                    artifical_block['coords']['O'] = coords[:,3,:]
                    artifical_block['coords']['N'] = coords[:,0,:]

                    sample_dict = {
                    int(index_for_redis + num): self.get_sample_dict(artifical_block)
                    }

                    self.RS.append(sample_dict) 
                    num += 1
                else:
                    print("Filtered out a large one")
            elif 'casp15' in config.dataset_name:
                artifical_block = {}
                artifical_block['name'] = block['name']
                artifical_block['seq'] = block['seq']
                artifical_block['coords'] = {}
                artifical_block['CA'] = block['coords']['CA']
                artifical_block['C'] = block['coords']['C']
                artifical_block['O'] = block['coords']['O']
                artifical_block['N'] = block['coords']['N']

                self.RS.append(artifical_block) 
                num += 1
            elif config.custom_pdb:
                self.RS.append(self.preprocess_pdb(block))
                num += 1
            else:
                coords = ast.literal_eval(block[1]['coords'].replace("'", '"').replace('\n', '').replace(' ', '').replace('array(', '').replace(')','').replace(',dtype=object','').replace('nan', 'None'))
                for key in coords:
                    coords[key] = [np.nan if x is None else x for x in coords[key]]
                artifical_block = {}
                artifical_block['name'] = block[1]['name']
                artifical_block['seq'] = block[1]['seq']
                artifical_block['coords'] = {}
                artifical_block['CA'] = np.stack(coords['CA'])
                artifical_block['C'] = np.stack(coords['C'])
                artifical_block['O'] = np.stack(coords['O'])
                artifical_block['N'] = np.stack(coords['N'])
                self.RS.append(artifical_block)
                
        print(f"Entered {len(self.RS)} samples into Redis")
        self.batch = None
    
    def preprocess_pdb(self, pdb_path):
        parser = PDBParser()
        structure = parser.get_structure("name", pdb_path)
        model = structure[0]
        chain = model["A"]
        seq = ""
        coords = {'CA': [], 'C': [], 'N': [], 'O': []}
        for residue in chain:
            if residue.get_resname() not in AMINO_ACIDS.keys():
                continue
            seq += AMINO_ACIDS[residue.get_resname()]
            coords['CA'].append(residue["CA"].get_coord())
            coords['C'].append(residue["C"].get_coord())
            coords['N'].append(residue["N"].get_coord())
            coords['O'].append(residue["O"].get_coord())
    
        return {
            "name": pdb_path.split("/")[-1].split('.')[0],
            "seq": seq,
            "coords": coords,
        }

    def preprocess_src(
        self,
        src: pd.DataFrame,
        index_for_redis: int = 0,
    ) -> list:

        src = src.reset_index()
        sample_dict_list = {
            int(index_for_redis + idx): self.get_sample_dict(src.loc[idx])
            for idx in src.index.values
        }
        return sample_dict_list

    def preprocess_src_samples_dict(
        self, src: pd.DataFrame, index_for_redis: int = 0
    ) -> dict:
        # reset the index as src dataframe might be filtered, these needs to be linear for redis indexing
        src = src.reset_index()
        sample_dict_dict = {
            int(index_for_redis + idx): self.get_sample_dict(src.loc[idx])
            for idx in src.index.values
        }
        return sample_dict_dict
    
    def get_sample_dict(self, row) -> dict:
        if self.pifold:
            return {
                "title": row["name"],
                "seq": row["seq"],
                "CA": np.stack(row["CA"]),
                "N": np.stack(row["N"]),
                "C": np.stack(row["C"]),
                "O": np.stack(row["O"]),
                "score": 100.0,
            }
        else:
            name = row["name"]

            coords = torch.stack(
                [
                    torch.tensor(np.stack(row["coords"]["N"])),
                    torch.tensor(np.stack(row["coords"]["CA"])),
                    torch.tensor(np.stack(row["coords"]["C"])),
                    torch.tensor(np.stack(row["coords"]["O"])),
                ],
                axis=1,
            ).float()

            seq = row["seq"]
            seq_one_hot_features = self._get_amino_acid_features(seq)
            node_features = seq_one_hot_features

            isnan = np.isnan(coords)
            mask = np.isfinite(torch.sum(coords, (1, 2)))
            coords[isnan.bool()] = 0.0
            node_features[np.isnan(node_features).bool()] = 0.0
            node_features[np.isinf(node_features).bool()] = 0.0
            
            return {
                "name": name,
                "bb": coords,
                "feat": node_features,
                "seq": seq,
                "coords_mask": mask,
            }

    def __len__(self) -> int:
        return len(self.RS)

    def __getitem__(self, idx):
        return self.RS[idx]

    def collate_fn(
        self, samples
    ):  # in this case samples is a list of dictionaries of parsed pdbs
        (
            X,
            S,
            mask,
            lengths,
            chain_M,
            residue_idx,
            mask_self,
            chain_encoding_all,
        ) = featurize(samples)

        batch = {
            "X": X,
            "S": S,
            "mask": mask,
            "chain_M": chain_M,
            "residue_idx": residue_idx,
            "chain_encoding_all": chain_encoding_all,
            "lengths": lengths,
        }

        for key in ["name", "seq", "folded_true_seq"]:
            batch[key] = get_key(samples, key)

        if not self.batch:
            self.batch = batch

        return batch