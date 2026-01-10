# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib
# Agent Selection Module for Heterogeneous Collaboration.

import numpy as np
import random
import os
from collections import OrderedDict
import json

class Adaptor:
    def __init__(self, 
                ego_modality, 
                model_modality_list, 
                modality_assignment,
                lidar_channels_dict,
                mapping_dict,
                cav_preference,
                train):
        self.ego_modality = ego_modality
        self.model_modality_list = model_modality_list
        self.modality_assignment = modality_assignment
        self.lidar_channels_dict = lidar_channels_dict
        self.mapping_dict = mapping_dict
        if cav_preference is None:
            cav_preference = dict.fromkeys(model_modality_list, 1/len(model_modality_list))
        self.cav_preferece = cav_preference # training, probability for setting non-ego cav modality
        self.train = train


    def reorder_cav_list(self, cav_list, scenario_name):
        """
        When evaluation, make the cav that could be ego modality after mapping be the first.

        This can check the training effect of aligner.

        work in basedataset -> reinitialize
        """
        # if self.train:
        #     # shuffle the cav list
        #     random.shuffle(cav_list)
        #     return cav_list

        # assignment = self.modality_assignment[scenario_name]
        # # Check if the first CAV's mapped modality matches ego_modality
        # # Use exact string comparison instead of substring check
        # first_cav_mapped_modality = self.mapping_dict[assignment[cav_list[0]]]
        
        # # Split ego_modality if it contains "&" (e.g., "m1&m2")
        # ego_modality_list = self.ego_modality.split("&") if "&" in self.ego_modality else [self.ego_modality]
        
        # if first_cav_mapped_modality not in ego_modality_list:
        #     ego_cav = None
        #     for cav_id, modality in assignment.items():
        #         mapped_modality = self.mapping_dict.get(modality, modality)
        #         if mapped_modality in ego_modality_list:  # after mapping the modality is ego
        #             ego_cav = cav_id
        #             break

        #     if ego_cav is None:
        #         return cav_list

        #     other_cav = sorted(list(assignment.keys()))
        #     other_cav.remove(ego_cav)
        #     cav_list = [ego_cav] + other_cav

        return cav_list
    
    def reassign_cav_modality(self, modality_name, idx_in_cav_list):
        """
        work in basedataset -> reinitialize
        """
        # if self.train: 
        #     # always assign the ego_modality to idx 0 in cav_list
        #     if idx_in_cav_list == 0:
        #         return np.random.choice(self.ego_modality.split("&"))
        #     return random.choices(list(self.cav_preferece.keys()), weights=self.cav_preferece.values())[0]
        # else:
        #     return self.mapping_dict[modality_name]
        return self.mapping_dict[modality_name]

    def unmatched_modality(self, cav_modality):
        """
        work in 
            intermediate_heter_fusion_dataset -> __getitem__
            late_heter_fusion_dataset -> get_item_test

        Returns:
            True/False. If the input modality is in the model_modality_list
        """
        cav_modality_list = cav_modality.split("&")
        for cav_modality_item in cav_modality_list:
            if cav_modality_item not in self.model_modality_list:
                return True
        return False
        # return cav_modality not in self.model_modality_list


    def switch_lidar_channels(self, cav_modality, lidar_file_path):
        """
        Currently only support OPV2V
        """
        if self.lidar_channels_dict.get(cav_modality, None) == 32:
            return lidar_file_path.replace("OPV2V","OPV2V_Hetero").replace(".pcd", "_32.pcd")
        if self.lidar_channels_dict.get(cav_modality, None) == 16:
            return lidar_file_path.replace("OPV2V","OPV2V_Hetero").replace(".pcd", "_16.pcd")
        return lidar_file_path


def assign_modality(root_dir="dataset/OPV2V", 
                   output_path="opencood/logs/heter_modality_assign/modality.json",
                   num_modalities=4,
                   splits=None,
                   in_order=False,
                   check_path=True,
                   random_seed=303):
    """
    General function to assign modalities to a dataset.
    
    Args:
        root_dir: Root directory of the dataset
        output_path: Output JSON file path
        num_modalities: Number of modalities (2 or 4)
        splits: List of splits to process, default is ['train', 'validate', 'test']
        in_order: Whether to assign in order (True: m1->m2->m3->m4 circularly, False: random assignment)
        check_path: Whether to check if path exists
        random_seed: Random seed (only used when not assigning in order)
    """
    if splits is None:
        splits = ['train', 'validate', 'test']
    
    if not in_order:
        np.random.seed(random_seed)
    
    scenario_cav_modality_dict = OrderedDict()

    for split in splits:
        split_path = os.path.join(root_dir, split)
        if check_path and not os.path.exists(split_path):
            print(f"Warning: {split_path} does not exist, skipping...")
            continue
            
        scenario_folders = sorted([os.path.join(split_path, x)
                                    for x in os.listdir(split_path) if
                                    os.path.isdir(os.path.join(split_path, x))])

        for scenario_folder in scenario_folders:
            scenario_name = scenario_folder.split('/')[-1]
            scenario_cav_modality_dict[scenario_name] = OrderedDict()

            cav_list = sorted([x for x in os.listdir(scenario_folder) \
                                if os.path.isdir(os.path.join(scenario_folder, x))])
            
            # Handle special case of '-1' (when assigning in order)
            if in_order and cav_list and cav_list[0] == '-1':
                cav_list = cav_list[1:] + cav_list[:1]

            if in_order:
                # Assign in circular order
                for j, cav_id in enumerate(cav_list):
                    scenario_cav_modality_dict[scenario_name][cav_id] = 'm'+str(j % num_modalities + 1)
            else:
                # Random assignment
                perm = np.random.permutation(num_modalities) + 1
                for j, cav_id in enumerate(cav_list):
                    scenario_cav_modality_dict[scenario_name][cav_id] = 'm'+str(perm[j % num_modalities])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(scenario_cav_modality_dict, f, indent=4, sort_keys=True)
    print(f"Modality assignment saved to {output_path}")

# Convenience functions
def assign_modality_4(root_dir="dataset/OPV2V", output_path="opencood/logs/heter_modality_assign/opv2v_4modality.json"):
    assign_modality(root_dir, output_path, num_modalities=4, splits=['train', 'test', 'validate'])

def assign_modality_4_in_order(root_dir="dataset/OPV2V", output_path="opencood/logs/heter_modality_assign/opv2v_4modality_in_order.json"):
    assign_modality(root_dir, output_path, num_modalities=4, splits=['train', 'test', 'validate'], in_order=True)

def assign_modality_2(root_dir="dataset/OPV2V", output_path="opencood/logs/heter_modality_assign/opv2v_2modality.json"):
    assign_modality(root_dir, output_path, num_modalities=2, splits=['train', 'test', 'validate'])

if __name__ == "__main__":
    # Example usage:
    # assign_modality_4('dataset/V2XSET', output_path='opencood/logs/heter_modality_assign/v2xset_4modality.json')
    # assign_modality_2('datasets/v2x-radar-v1.0', output_path='opencood/logs/heter_modality_assign/v2xset_4modality_ours-v1.0.json')
    # assign_modality('/mnt/ssd8T/Cooperative_Perception/V2X-R', 'docs/modality_assign/v2x_r_4modality.json', num_modalities=4)
    # assign_modality('/mnt/ssd8T/Cooperative_Perception/V2X-R', 'docs/modality_assign/v2x_r_2modality.json', num_modalities=2)
    
    # Assign modalities for V2X-R
    assign_modality_4_in_order('data/v2x-r', 'docs/modality_assign/v2x_r_4modality.json')
    