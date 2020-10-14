import numpy as np

def reorder_markers(biorbd_model, c3d, frames, step_size=1, broken_dofs=None):
    markers = c3d['data']['points'][:3, :95, frames.start:frames.stop:step_size] / 1000
    c3d_labels = c3d['parameters']['POINT']['LABELS']['value'][:95]
    model_labels = [label.to_string() for label in biorbd_model.markerNames()]

    labels_index = []
    missing_markers_index = []
    for index_model, model_label in enumerate(model_labels):
        missing_markers_bool = True
        for index_c3d, c3d_label in enumerate(c3d_labels):
            if model_label in c3d_label:
                labels_index.append(index_c3d)
                missing_markers_bool = False
        if missing_markers_bool:
            labels_index.append(index_model)
            missing_markers_index.append(index_model)

    markers_reordered = np.zeros((3, markers.shape[1], markers.shape[2]))
    for index, label_index in enumerate(labels_index):
        if index in missing_markers_index:
            markers_reordered[:, index, :] = np.nan
        else:
            markers_reordered[:, index, :] = markers[:, label_index, :]

    model_segments = {
        'pelvis': {'markers': ['EIASD', 'CID', 'EIPSD', 'EIPSG', 'CIG', 'EIASG'], 'dofs': range(0, 6)},
        'thorax': {'markers': ['MANU', 'MIDSTERNUM', 'XIPHOIDE', 'C7', 'D3', 'D10'], 'dofs': range(6, 9)},
        'head': {'markers': ['ZYGD', 'TEMPD', 'GLABELLE', 'TEMPG', 'ZYGG'], 'dofs': range(9, 12)},
        'right_shoulder': {'markers': ['CLAV1D', 'CLAV2D', 'CLAV3D', 'ACRANTD', 'ACRPOSTD', 'SCAPD'], 'dofs': range(12, 14)},
        'right_arm': {'markers': ['DELTD', 'BICEPSD', 'TRICEPSD', 'EPICOND', 'EPITROD'], 'dofs': range(14, 17)},
        'right_forearm': {'markers': ['OLE1D', 'OLE2D', 'BRACHD', 'BRACHANTD', 'ABRAPOSTD', 'ABRASANTD', 'ULNAD', 'RADIUSD'], 'dofs': range(17, 19)},
        'right_hand': {'markers': ['METAC5D', 'METAC2D', 'MIDMETAC3D'], 'dofs': range(19, 21)},
        'left_shoulder': {'markers': ['CLAV1G', 'CLAV2G', 'CLAV3G', 'ACRANTG', 'ACRPOSTG', 'SCAPG'], 'dofs': range(21, 23)},
        'left_arm': {'markers': ['DELTG', 'BICEPSG', 'TRICEPSG', 'EPICONG', 'EPITROG'], 'dofs': range(23, 26)},
        'left_forearm': {'markers': ['OLE1G', 'OLE2G', 'BRACHG', 'BRACHANTG', 'ABRAPOSTG', 'ABRANTG', 'ULNAG', 'RADIUSG'], 'dofs': range(26, 28)},
        'left_hand': {'markers': ['METAC5G', 'METAC2G', 'MIDMETAC3G'], 'dofs': range(28, 30)},
        'right_thigh': {'markers': ['ISCHIO1D', 'TFLD', 'ISCHIO2D', 'CONDEXTD', 'CONDINTD'], 'dofs': range(30, 33)},
        'right_leg': {'markers': ['CRETED', 'JAMBLATD', 'TUBD', 'ACHILED', 'MALEXTD', 'MALINTD'], 'dofs': range(33, 34)},
        'right_foot': {'markers': ['CALCD', 'MIDMETA4D', 'MIDMETA1D', 'SCAPHOIDED', 'METAT5D', 'METAT1D'], 'dofs': range(34, 36)},
        'left_thigh': {'markers': ['ISCHIO1G', 'TFLG', 'ISCHIO2G', 'CONEXTG', 'CONDINTG'], 'dofs': range(36, 39)},
        'left_leg': {'markers': ['CRETEG', 'JAMBLATG', 'TUBG', 'ACHILLEG', 'MALEXTG', 'MALINTG', 'CALCG'], 'dofs': range(39, 40)},
        'left_foot': {'markers': ['MIDMETA4G', 'MIDMETA1G', 'SCAPHOIDEG', 'METAT5G', 'METAT1G'], 'dofs': range(40, 42)},
    }

    markers_idx_broken_dofs = []
    if broken_dofs is not None:
        for dof in broken_dofs:
            for segment in model_segments.values():
                if dof in segment['dofs']:
                    marker_positions = [index_model for marker_label in segment['markers'] for index_model, model_label in enumerate(model_labels) if marker_label in model_label]
                    if range(min(marker_positions), max(marker_positions) + 1) not in markers_idx_broken_dofs:
                        markers_idx_broken_dofs.append(range(min(marker_positions), max(marker_positions) + 1))

    return markers_reordered, markers_idx_broken_dofs
