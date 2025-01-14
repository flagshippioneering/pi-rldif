from model.mod_pifold import InverseFoldingDiffusionPiFoldModel
import os
import torch
from data.dataset import RLDIFDataset
from torch.utils.data import DataLoader
from utils.utils import load_config, featurize_GTrans, mpnn_index_to_AA, slice_dict
from tqdm import tqdm
import numpy as np
import pandas as pd
import esm
from transformers import AutoTokenizer
from model.protein_mpnn import ProteinMPNN
from run.trainer import train

def test(config, model, dataloader, split_name, foldfunction = None):

    if config.protein_mpnn:
        index_to_AA = mpnn_index_to_AA
    elif config.rldif or config.dif_large:
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        index_to_AA = {i: a for i, a in enumerate(alphabet)}
    elif config.esmif:
        index_to_AA = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
        valid_indices = index_to_AA.encode('ARNDCQEGHILKMFPSTWYV')[1:-1]
        alphabet='ARNDCQEGHILKMFPSTWYV'
        mask_token = []
        bad_tokens = []
        for token, idx in index_to_AA.get_vocab().items():
            if idx not in valid_indices:
                mask_token.append(idx)
                bad_tokens.append(token)

    results = []

    per_batch_accuracy = []

    per_batch_tm_vec_output = []
    per_batch_tm_scores = []

    for batch in tqdm(dataloader):

        for i in range(config.num_samples):
            if config.protein_mpnn:
                names = batch["name"]
                batch = slice_dict(
                    batch,
                    keys=[
                        "X",
                        "S",
                        "mask",
                        "chain_M",
                        "residue_idx",
                        "chain_encoding_all",
                    ],
                )

                batch["decoding_order"] = torch.arange(torch.prod(torch.tensor(batch["chain_M"].shape))).reshape(batch["chain_M"].shape).cuda()

                sample_args = {**batch}

                if config.free_positions:
                    pos_unfixed_mask = []
                    S_true = batch["S"]
                    for pos in range(S_true.shape[1]):
                        if pos in config.free_positions:
                            pos_unfixed_mask.append(1)
                        else:
                            pos_unfixed_mask.append(0)
                    sample_args["pos_unfixed_mask"] = torch.tensor(pos_unfixed_mask).cuda()

                if bool(config.temp):
                    sample_args['temperature'] = config.temp
                else:
                    sample_args["decoding_order"] = torch.rand(batch["chain_M"].shape).cuda()
                    sample_args['temperature'] = 0.000000000001
                    
                samples = model.simplest_sample(**sample_args)

                out = {}
                out["features_0_step"] = samples["S"]
                out["features_true"] = batch["S"]
                out["mask"] = batch["mask"]
                batch["name"] = names
            elif config.rldif or config.dif_large:
                out = model.sample(batch.clone().cuda(), closure=True)
                names = batch["names"]
            elif config.esmif:
                coords = batch['X'][:,:,:3, :]
                pred_sequences = []
                true_seq = [i for i in index_to_AA.decode([str(i.item()) for i in list(batch['S'][0])])][::2]
                
                for i in coords:
                    if bool(config.free_positions):
                        fixedprot = []
                        for pos in range(len(true_seq)):
                            if pos in config.free_positions:
                                fixedprot.append('<mask>')
                            else:
                                fixedprot.append(true_seq[pos])

                        res = model.sample(i, partial_seq = fixedprot, temperature = 1.0)
                    else:
                        res = model.sample(i, temperature = 1.0)
                    pred_sequences.append(index_to_AA.encode(res)[1:-1])

                out = {}
                out['features_0_step'] = torch.stack([torch.tensor(i) for i in pred_sequences])
                out['features_true'] = batch['S']
                out['mask'] = batch['mask']
                names = batch['name']
            else:
                out = model.sample(batch.clone().cuda(), closure=True)
                names = batch["names"]

            accs = []
            tm_vec_res = []
            counter = 0
            num = 0
            for ft, fp, mask, name in zip(
                out["features_true"],
                out["features_0_step"],
                out["mask"],
                names,
            ):
                if config.protein_mpnn or config.esmif:
                    mask = mask.to(torch.bool)
                else:
                    mask = mask.astype(bool)
                
                fp, ft = fp[mask], ft[mask]

                n = ft.shape[0]

                if config.protein_mpnn or config.esmif:
                    acc = (ft == fp).sum(axis=-1).sum() / float(n)
                    if config.esmif:
                        pred_sequence= "".join(index_to_AA.decode(fp).split(" "))
                        real_sequence = "".join(index_to_AA.decode(ft).split(" "))
                        for l in bad_tokens:
                            if l in pred_sequence:
                                pred_sequence = pred_sequence.replace(l, 'A')
                            if l in real_sequence:
                                real_sequence = real_sequence.replace(l, 'A')
                    else:
                        mpnn_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
                        pred_sequence = "".join([mpnn_alphabet[i] for i in fp])
                        real_sequence = "".join([mpnn_alphabet[i] for i in ft])

                    coordinates_res = batch["X"][batch["name"].index(name)]
                    mask_res = batch["mask"][batch["name"].index(name)].bool()
                else:
                    acc = (
                        ft.argmax(axis=-1) == fp.argmax(axis=-1)
                    ).sum() / float(n)
                    ft = ft.argmax(axis=-1)
                    fp = fp.argmax(axis=-1)

                    pred_sequence = "".join(
                        np.vectorize(index_to_AA.get)(fp).tolist()
                    )
                    real_sequence = "".join(
                        np.vectorize(index_to_AA.get)(ft).tolist()
                    )

                    if "name" not in batch:
                        coordinates_res = batch["x"][
                            batch["batch"] == batch["names"].index(name)
                        ]
                        mask_res = batch["mask"][
                            batch["batch"] == batch["names"].index(name)
                        ]
                    else:
                        coordinates_res = batch["x"][
                            batch["batch"] == batch["name"].index(name)
                        ]
                        mask_res = batch["mask"][
                            batch["batch"] == batch["name"].index(name)
                        ]

                accs.append(acc)

                if foldfunction is not None:
                    tm_scores = foldfunction(pred_sequence, real_sequence)
                    print(tm_scores)
                else:
                    tm_scores = None

                results.append(
                    {
                        "name": name,
                        "pred": pred_sequence,
                        "real": real_sequence,
                        "tm_score": tm_scores,
                        "split_name": split_name,
                    }
                )

                num += 1
        
        if type(accs[0]) is torch.Tensor:
            accs = [i.cpu().numpy() for i in accs]
        acc = np.mean(accs)
        print(f"Accuracy: {acc}")

        if tm_scores is not None:
            tm_scores = np.mean(tm_scores)
        else:
            print(f"TM-Score Output: {tm_scores}")
 
        per_batch_accuracy.append(acc)
        per_batch_tm_scores.append(tm_scores)

    print(f"Average Accuracy: {np.mean(per_batch_accuracy)}")
    if foldfunction is not None:
        print(f"Average TM-Score: {np.mean(per_batch_tm_scores)}")

    df = pd.DataFrame(results)
    if config.docker:
        pdb_base_path = os.environ.get('PDB_BASE_PATH', '/usr/src/app/input_files/')
        df.to_csv(pdb_base_path + str(config.name) + "_results.csv")
    else:
        df.to_csv(str(config.name) + "_results.csv")

if __name__ == '__main__':
    args = load_config('./configs/config.yaml')

    if args.docker is True:
        pdb_base_path = os.environ.get('PDB_BASE_PATH', '/usr/src/app/input_files/')
        args = load_config(pdb_base_path + 'config.yaml')
        args.data.custom_pdb_input = pdb_base_path + args.data.custom_pdb_input
        args.data.docker = True
    else:
        args.data.docker = False

    if sum([args.rldif, args.dif_large, args.esmif, args.protein_mpnn]) != 1:
        raise ValueError("Exactly one model must be selected.")

    if args.rldif is True or args.dif_large is True:
        if args.dif_large:
            args.pifold_model.network.num_encoder_layers = 20
        args.pifold_model.free_positions = args.free_positions
        model = InverseFoldingDiffusionPiFoldModel(args.pifold_model).cuda()
        
        if args.dif_large:
            if not os.path.exists('RLDIF_8M.ckpt'):
                os.system("wget https://zenodo.org/records/14509073/files/RLDIF_8M.ckpt")
            state_dict = torch.load('RLDIF_8M.ckpt')['state_dict']
        else:
            #This is the link to the last.ckpt file
            if not os.path.exists('last.ckpt'):
                os.system('wget https://zenodo.org/records/11304952/files/last.ckpt')
            state_dict = torch.load('last.ckpt')['state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace('model.', '')] = v
        model.load_state_dict(new_state_dict)
    elif args.esmif is True:
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    elif args.protein_mpnn is True:
        args.model_mpnn.k_neighbors = 38
        model = ProteinMPNN(args.model_mpnn).cuda() 
        if not os.path.exists('protein_mpnn_checkpoint.ckpt'):
            os.system('wget https://zenodo.org/records/14509073/files/protein_mpnn_checkpoint.ckpt')
        checkpoint = torch.load('protein_mpnn_checkpoint.ckpt')
        ckpt_dict_new = dict()
        for k in checkpoint["state_dict"].keys():
            res = k.replace("model.mpnn.", "")
            ckpt_dict_new[res] = checkpoint["state_dict"][k]
        model.load_state_dict(ckpt_dict_new)

    model = model.eval()
    args.data.rldif = args.rldif
    args.data.protein_mpnn = args.protein_mpnn
    dataset = RLDIFDataset(args.data)

    if args.rldif or args.dif_large:
        collate_function = model.collate_fn
    elif args.esmif:
        collate_function = featurize_GTrans
    elif args.protein_mpnn:
        collate_function = dataset.collate_fn

    dataloader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=False,
                collate_fn=collate_function,
            )

    if args.inference:
        test(args, model, dataloader, args.data.split_name)
    else:
        if not args.rldif and not args.dif_large:
            raise ValueError("Finetuning is only supported for RLDIF and DIF-Large model.")
        else:
            master_config = load_config('./configs/master_config.yaml')
            master_config.EnvironmentConfig.n_gpus = torch.cuda.device_count()

            #n_gpus: !expr torch.cuda.device_count()

            # Go through args and update master_config
            for key, value in args.items():
                master_config[key] = value


            train(master_config, model, dataloader)
