import argparse
import yaml
import json
from model.mod_pifold import InverseFoldingDiffusionPiFoldModel
import os
import torch
from data.dataset import RLDIFDataset
from torch.utils.data import DataLoader
from utils.utils import Config, load_config
from tqdm import tqdm
import numpy as np
import pandas as pd

def test(config, model, dataloader, split_name, foldfunction = None):

    if config.protein_mpnn:
        index_to_AA = proteins.IndexedAminoAcids._bwd_map()
    elif config.pifold:
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        index_to_AA = {i: a for i, a in enumerate(alphabet)}
    elif config.kwdesign:
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

        # Sample 4 times

        for i in range(4):
            if config.protein_mpnn:
                names = batch["name"]
                batch = utils.data.slice_dict(
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

                # Running MPNN with random decoding order and zero temperature
                batch["decoding_order"] = torch.rand(batch["chain_M"].shape)

                samples = model.simplest_sample(
                    **batch,
                    temperature=0.000000000001,
                )

                out = {}
                out["features_0_step"] = samples["S"]
                out["features_true"] = batch["S"]
                out["mask"] = batch["mask"]
                batch["name"] = names
            elif config.pifold:
                if config.pifold_model.diffusion_mode:
                    out = model.sample(batch.clone().cuda(), closure=True)
                    names = batch["names"]
                else:
                    out = {}
                    out["features_0_step"] = model(batch).detach().cpu()
                    out["features_true"] = batch[1]
                    out["mask"] = batch[3]
                    names = batch[5]
            elif config.kwdesign:
                #SETTING TEMP TO 0.1 FOR BEST RESULT FROM PAPER
                temperature = 0.1
                names = batch['name']
                pred_results = []
                true_results = []

                for i in names:
                    sub_res, true_res = model(i)
                    if temperature==0:
                        sub_res = sub_res.argmax(dim=-1)
                    else:
                        sub_res[:, mask_token] = -100000000000000
                        sub_res = torch.multinomial(torch.softmax(sub_res/temperature, dim=-1), 1).squeeze(1)
                        replace_values = [0, 1, 2, 3, 29, 30, 31, 32]
                        for val in replace_values:
                            sub_res = torch.where(sub_res == val, torch.tensor(28, dtype=torch.long), sub_res)
                    pred_results.append(sub_res)
                    true_results.append(true_res)
                    
        
                padded_pred_results = pad_sequence(pred_results, batch_first=True, padding_value=-1)
                padded_true_results = pad_sequence(true_results, batch_first=True, padding_value=-1)
                mask = padded_pred_results != -1
                out = {}
                out['features_0_step'] = padded_pred_results
                out['features_true'] = padded_true_results
                out['mask'] = mask
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
                if config.pifold and not config.pifold_model.diffusion_mode:
                    ft = ft[mask.bool()]
                    n = ft.shape[0]
                    fp = out["features_0_step"][counter : counter + n]
                    counter += n

                else:
                    if config.protein_mpnn or config.kwdesign:
                        mask = mask.to(torch.bool)
                    else:
                        mask = mask.astype(bool)
                    
                    fp, ft = fp[mask], ft[mask]

                    n = ft.shape[0]

                if config.protein_mpnn or config.kwdesign:
                    acc = (ft == fp).sum(axis=-1).sum() / float(n)
                    if config.kwdesign:
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
                    if config.pifold and not config.pifold_model.diffusion_mode:
                        acc = (ft == fp.argmax(axis=-1)).sum() / float(n)
                        fp = fp.argmax(axis=-1)
                    
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

                    if config.pifold and not config.pifold_model.diffusion_mode:
                        coordinates_res = batch[0][num][mask.bool(), :, :]
                        mask_res = mask.bool()
                    else:
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
                #print(acc)

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
    df.to_csv(str(config.name) + "_results.csv")





if __name__ == '__main__':
    args = load_config('./configs/config.yaml')
    model = InverseFoldingDiffusionPiFoldModel(args.pifold_model).cuda()
    #Check if last.ckpt exists in current directory then wget
    if not os.path.exists('last.ckpt'):
        os.system('wget https://zenodo.org/records/11304952/files/last.ckpt')
    state_dict = torch.load('last.ckpt')['state_dict']
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k.replace('model.', '')] = v
    model.load_state_dict(new_state_dict)
    dataset = RLDIFDataset(args.data)

    dataloader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=False,
                collate_fn=model.collate_fn,
            )
    
    test(args, model, dataloader, args.data.split_name)

