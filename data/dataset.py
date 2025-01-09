from torch.utils.data import Dataset
import json 
import numpy as np 
import pandas as pd
import torch 
from tqdm import tqdm
import ast
from Bio.PDB import PDBParser
import random

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

        self.rldif = config.rldif
        self.protein_mpnn = config.protein_mpnn
        # redis version
        index_for_redis = 0
        if config.custom_pdb_input is not False:
            data_paths = config.custom_pdb_input
        elif config.dataset_name == 'ts50':
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
        else:
            raise Exception("Invalid dataset name")

        dp = data_paths
        if config.custom_pdb_input is not False:
            x = [i.rstrip().split(',') for i in open(dp, 'r').readlines()]
            iterator = tqdm(x)
        elif 'cath' in config.dataset_name:
            x = pd.read_csv(dp)
            iterator = tqdm(x.iterrows())
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
            elif config.custom_pdb_input is not False:
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
    
    def preprocess_pdb(self, data):
        pdb_path, chain = data

        parser = PDBParser()
        structure = parser.get_structure("name", pdb_path)
        model = structure[0]

        if chain == 'all' or chain == 'All' or chain == 'ALL':
            data = [model[chain.id] for chain in model]
            chain_letter = "_".join([chain.id for chain in model])
        else:
            data = [model[chain]]
            chain_letter = chain

        seq = ""
        coords = {'CA': [], 'C': [], 'N': [], 'O': []}
        for chain in data:
            for residue in chain:
                if residue.get_resname() not in AMINO_ACIDS.keys():
                    continue
                try:
                    ca_to_add = residue["CA"].get_coord()
                    c_to_add = residue["C"].get_coord()
                    n_to_add = residue["N"].get_coord()
                    o_to_add = residue["O"].get_coord()
                    coords['CA'].append(ca_to_add)
                    coords['C'].append(c_to_add)
                    coords['N'].append(n_to_add)
                    coords['O'].append(o_to_add)
                    seq += AMINO_ACIDS[residue.get_resname()]
                except:
                    #One of the residues is malformed so we exlude it
                    continue
        
        if self.protein_mpnn:
            chain_letter = chain_letter
            name = pdb_path.split("/")[-1].split('.')[0]
            sequence = seq

            # The order of backbone atoms is [N, CA, C, O]
            chain_coords = {}
            chain_coords["N_chain_" + chain_letter] = coords["N"]
            chain_coords["CA_chain_" + chain_letter] = coords['CA']
            chain_coords["C_chain_" + chain_letter] = coords["C"]
            chain_coords["O_chain_" + chain_letter] = coords["O"]

            return {"name": name,
                "chain_id": chain_letter,
                "seq": sequence,
                "coords_chain_" + chain_letter: chain_coords,
                "num_of_chains": 1,
                "seq_chain_" + chain_letter: sequence,
                "masked_list": [chain_letter],
                "visible_list": []}
        else:
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
        if self.rldif:
            return {
                "title": row["name"],
                "seq": row["seq"],
                "CA": np.stack(row["CA"]),
                "N": np.stack(row["N"]),
                "C": np.stack(row["C"]),
                "O": np.stack(row["O"]),
                "score": 100.0,
            }
        elif self.protein_mpnn:
            chain_letter = row["name"].split(".")
            if len(chain_letter) == 1:
                chain_letter = None
            else:
                chain_letter = chain_letter[1]
            name = row["name"]
            if chain_letter is None:
                chain_letter = "X"
            if name is None:
                name = "XXXX"
            # some chains are encoded as numbers and/or letters
            chain_letter = str(chain_letter)
            sequence = "".join([aa for aa in row["seq"]])

            # The order of backbone atoms is [N, CA, C, O]
            chain_coords = {}
            chain_coords["N_chain_" + chain_letter] = row["coords"]["N"]
            chain_coords["CA_chain_" + chain_letter] = row["coords"]["CA"]
            chain_coords["C_chain_" + chain_letter] = row["coords"]["C"]
            chain_coords["O_chain_" + chain_letter] = row["coords"]["O"]

            return {"name": name,
                "chain_id": chain_letter,
                "seq": sequence,
                "coords_chain_" + chain_letter: chain_coords,
                "num_of_chains": 1,
                "seq_chain_" + chain_letter: sequence,
                "masked_list": [chain_letter],
                "visible_list": []}
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

def featurize(batch):
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"
    B = len(batch)
    lengths = np.array(
        [len(b["seq"]) for b in batch], dtype=np.int32
    )  # sum of chain seq lengths
    L_max = max([len(b["seq"]) for b in batch])
    # L_max = 100
    X = np.zeros([B, L_max, 4, 3])
    residue_idx = -100 * np.ones(
        [B, L_max], dtype=np.int32
    )  # residue idx with jumps across chains
    chain_M = np.zeros(
        [B, L_max], dtype=np.int32
    )  # 1.0 for the bits that need to be predicted, 0.0 for the bits that are given
    mask_self = np.ones(
        [B, L_max, L_max], dtype=np.int32
    )  # for interface loss calculation - 0.0 for self interaction, 1.0 for other
    chain_encoding_all = np.zeros(
        [B, L_max], dtype=np.int32
    )  # integer encoding for chains 0, 0, 0,...0, 1, 1,..., 1, 2, 2, 2...
    S = np.zeros([B, L_max], dtype=np.int32)  # sequence AAs integers
    init_alphabet = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]
    extra_alphabet = [str(item) for item in list(np.arange(300))]
    chain_letters = init_alphabet + extra_alphabet
    for i, b in enumerate(batch):
        masked_chains = b["masked_list"]
        visible_chains = b["visible_list"]
        all_chains = masked_chains + visible_chains
        visible_temp_dict = {}
        masked_temp_dict = {}
        for step, letter in enumerate(all_chains):
            chain_seq = b[f"seq_chain_{letter}"]
            if letter in visible_chains:
                visible_temp_dict[letter] = chain_seq
            elif letter in masked_chains:
                masked_temp_dict[letter] = chain_seq
        for km, vm in masked_temp_dict.items():
            for kv, vv in visible_temp_dict.items():
                if vm == vv:
                    if kv not in masked_chains:
                        masked_chains.append(kv)
                    if kv in visible_chains:
                        visible_chains.remove(kv)
        all_chains = masked_chains + visible_chains
        random.shuffle(all_chains)  # randomly shuffle chain order
        num_chains = b["num_of_chains"]
        mask_dict = {}
        x_chain_list = []
        chain_mask_list = []
        chain_seq_list = []
        chain_encoding_list = []
        c = 1
        l0 = 0
        l1 = 0
        for step, letter in enumerate(all_chains):
            if letter in visible_chains:
                chain_seq = b[f"seq_chain_{letter}"]
                chain_length = len(chain_seq)
                chain_coords = b[f"coords_chain_{letter}"]  # this is a dictionary
                chain_mask = np.zeros(chain_length)  # 0.0 for visible chains
                x_chain = np.stack(
                    [
                        chain_coords[c]
                        for c in [
                            f"N_chain_{letter}",
                            f"CA_chain_{letter}",
                            f"C_chain_{letter}",
                            f"O_chain_{letter}",
                        ]
                    ],
                    1,
                )  # [chain_length,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1
            elif letter in masked_chains:
                chain_seq = b[f"seq_chain_{letter}"]
                chain_length = len(chain_seq)
                chain_coords = b[f"coords_chain_{letter}"]  # this is a dictionary
                chain_mask = np.ones(chain_length)  # 0.0 for visible chains
                x_chain = np.stack(
                    [
                        # chain_coords[c]
                        np.array([np.array(array).astype(float) for array in chain_coords[c]])
                        for c in [
                            f"N_chain_{letter}",
                            f"CA_chain_{letter}",
                            f"C_chain_{letter}",
                            f"O_chain_{letter}",
                        ]
                    ],
                    1,
                )  # [chain_lenght,4,3]
                x_chain_list.append(x_chain)
                chain_mask_list.append(chain_mask)
                chain_seq_list.append(chain_seq)
                chain_encoding_list.append(c * np.ones(np.array(chain_mask).shape[0]))
                l1 += chain_length
                mask_self[i, l0:l1, l0:l1] = np.zeros([chain_length, chain_length])
                residue_idx[i, l0:l1] = 100 * (c - 1) + np.arange(l0, l1)
                l0 += chain_length
                c += 1

        x = np.concatenate(x_chain_list, 0)  # [L, 4, 3]
        all_sequence = "".join(chain_seq_list)
        m = np.concatenate(
            chain_mask_list, 0
        )  # [L,], 1.0 for places that need to be predicted
        chain_encoding = np.concatenate(chain_encoding_list, 0)

        l = len(all_sequence)
        x_pad = np.pad(
            x, [[0, L_max - l], [0, 0], [0, 0]], "constant", constant_values=(np.nan,)
        )
        X[i, :, :, :] = x_pad

        m_pad = np.pad(m, [[0, L_max - l]], "constant", constant_values=(0.0,))
        chain_M[i, :] = m_pad

        chain_encoding_pad = np.pad(
            chain_encoding, [[0, L_max - l]], "constant", constant_values=(0.0,)
        )
        chain_encoding_all[i, :] = chain_encoding_pad

        # Convert to labels
        indices = np.asarray([alphabet.index(a) for a in all_sequence], dtype=np.int32)
        S[i, :l] = indices

    isnan = np.isnan(X)
    mask = np.isfinite(np.sum(X, (2, 3))).astype(np.float32)
    X[isnan] = 0.0

    # Conversion
    residue_idx = torch.from_numpy(residue_idx).to(dtype=torch.long)
    S = torch.from_numpy(S).to(dtype=torch.long)
    X = torch.from_numpy(X).to(dtype=torch.float32)
    mask = torch.from_numpy(mask).to(dtype=torch.float32)
    mask_self = torch.from_numpy(mask_self).to(dtype=torch.float32)
    chain_M = torch.from_numpy(chain_M).to(dtype=torch.float32)
    chain_encoding_all = torch.from_numpy(chain_encoding_all).to(dtype=torch.long)
    if torch.cuda.is_available():
        return X.cuda(), S.cuda(), mask.cuda(), lengths, chain_M.cuda(), residue_idx.cuda(), mask_self.cuda(), chain_encoding_all.cuda()
    else:
        return X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all

def get_key(iterable, key: str, strict: bool = False, allow_none: bool = True):
    if strict:
        values = [i[key] for i in iterable]
    else:
        values = [i.get(key) for i in iterable]
    if not allow_none and all(v is None for v in values):
        return None
    else:
        return values