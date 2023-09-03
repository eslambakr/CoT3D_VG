import numpy as np 
import math

def get3d_box_from_pcs(pc):
    """
    Given point-clouds that represent object or scene return the 3D dimension of the 3D box that contains the PCs.
    """
    w = pc[:, 0].max() - pc[:, 0].min()
    l = pc[:, 1].max() - pc[:, 1].min()
    h = pc[:, 2].max() - pc[:, 2].min()
    return w, l, h


def get3d_box_center_from_pcs(pc):
    """
    Given point-clouds that represent object or scene return the 3D center of the 3D box that contains the PCs.
    """
    w, l, h = get3d_box_from_pcs(pc)
    return np.array([pc[:, 0].max() - w / 2, pc[:, 1].max() - l / 2, pc[:, 2].max() - h / 2])


def extract_target_loc_from_pred_objs_from_description(pred_objs_list, target_class):
    indices = [c for c, x in enumerate(pred_objs_list) if x == target_class]  # find indices of the target class
    if len(indices) == 1:
        return indices[0]
    else:  
        return indices[-1] 

