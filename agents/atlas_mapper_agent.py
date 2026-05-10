from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from agents.schemas import AtlasRegion

_FS_TO_AAL: dict[str, str] = {
    "10": "Thalamus_L",         "left-thalamus": "Thalamus_L",
    "49": "Thalamus_R",         "right-thalamus": "Thalamus_R",
    "left-thalamus-proper": "Thalamus_L",
    "right-thalamus-proper": "Thalamus_R",
    "11": "Caudate_L",          "left-caudate": "Caudate_L",
    "50": "Caudate_R",          "right-caudate": "Caudate_R",
    "12": "Putamen_L",          "left-putamen": "Putamen_L",
    "51": "Putamen_R",          "right-putamen": "Putamen_R",
    "13": "Pallidum_L",         "left-pallidum": "Pallidum_L",
    "52": "Pallidum_R",         "right-pallidum": "Pallidum_R",
    "17": "Hippocampus_L",      "left-hippocampus": "Hippocampus_L",
    "53": "Hippocampus_R",      "right-hippocampus": "Hippocampus_R",
    "left-hippocampus-head": "Hippocampus_L",
    "left-hippocampus-body": "Hippocampus_L",
    "left-hippocampus-tail": "Hippocampus_L",
    "right-hippocampus-head": "Hippocampus_R",
    "right-hippocampus-body": "Hippocampus_R",
    "right-hippocampus-tail": "Hippocampus_R",
    "18": "Amygdala_L",         "left-amygdala": "Amygdala_L",
    "54": "Amygdala_R",         "right-amygdala": "Amygdala_R",
    "1024": "Precentral_L",     "ctx-lh-precentral": "Precentral_L",
    "2024": "Precentral_R",     "ctx-rh-precentral": "Precentral_R",
    "1022": "Postcentral_L",    "ctx-lh-postcentral": "Postcentral_L",
    "2022": "Postcentral_R",    "ctx-rh-postcentral": "Postcentral_R",
    "1028": "Frontal_Sup_L",    "ctx-lh-superiorfrontal": "Frontal_Sup_L",
    "2028": "Frontal_Sup_R",    "ctx-rh-superiorfrontal": "Frontal_Sup_R",
    "1003": "Frontal_Mid_L",    "ctx-lh-caudalmiddlefrontal":  "Frontal_Mid_L",
    "1027": "Frontal_Mid_L",    "ctx-lh-rostralmiddlefrontal": "Frontal_Mid_L",
    "2003": "Frontal_Mid_R",    "ctx-rh-caudalmiddlefrontal":  "Frontal_Mid_R",
    "2027": "Frontal_Mid_R",    "ctx-rh-rostralmiddlefrontal": "Frontal_Mid_R",
    "1018": "Frontal_Inf_Oper_L", "ctx-lh-parsopercularis":  "Frontal_Inf_Oper_L",
    "1019": "Frontal_Inf_Orb_L",  "ctx-lh-parsorbitalis":    "Frontal_Inf_Orb_L",
    "1020": "Frontal_Inf_Tri_L",  "ctx-lh-parstriangularis": "Frontal_Inf_Tri_L",
    "2018": "Frontal_Inf_Oper_R", "ctx-rh-parsopercularis":  "Frontal_Inf_Oper_R",
    "2019": "Frontal_Inf_Orb_R",  "ctx-rh-parsorbitalis":    "Frontal_Inf_Orb_R",
    "2020": "Frontal_Inf_Tri_R",  "ctx-rh-parstriangularis": "Frontal_Inf_Tri_R",
    "1014": "Frontal_Med_Orb_L",    "ctx-lh-medialorbitofrontal":  "Frontal_Med_Orb_L",
    "1012": "Frontal_Inf_Orb_L",    "ctx-lh-lateralorbitofrontal": "Frontal_Inf_Orb_L",
    "2014": "Frontal_Med_Orb_R",    "ctx-rh-medialorbitofrontal":  "Frontal_Med_Orb_R",
    "2012": "Frontal_Inf_Orb_R",    "ctx-rh-lateralorbitofrontal": "Frontal_Inf_Orb_R",
    "1032": "Frontal_Sup_Medial_L", "ctx-lh-frontalpole": "Frontal_Sup_Medial_L",
    "2032": "Frontal_Sup_Medial_R", "ctx-rh-frontalpole": "Frontal_Sup_Medial_R",
    "1030": "Temporal_Sup_L",   "ctx-lh-superiortemporal": "Temporal_Sup_L",
    "2030": "Temporal_Sup_R",   "ctx-rh-superiortemporal": "Temporal_Sup_R",
    "1001": "Temporal_Sup_L",   "ctx-lh-bankssts": "Temporal_Sup_L",
    "2001": "Temporal_Sup_R",
    "1015": "Temporal_Mid_L",   "ctx-lh-middletemporal":   "Temporal_Mid_L",
    "2015": "Temporal_Mid_R",   "ctx-rh-middletemporal":   "Temporal_Mid_R",
    "1009": "Temporal_Inf_L",   "ctx-lh-inferiortemporal": "Temporal_Inf_L",
    "2009": "Temporal_Inf_R",   "ctx-rh-inferiortemporal": "Temporal_Inf_R",
    "1033": "Temporal_Pole_Sup_L", "ctx-lh-temporalpole":  "Temporal_Pole_Sup_L",
    "2033": "Temporal_Pole_Sup_R", "ctx-rh-temporalpole":  "Temporal_Pole_Sup_R",
    "1034": "Heschl_L",         "ctx-lh-transversetemporal": "Heschl_L",
    "2034": "Heschl_R",         "ctx-rh-transversetemporal": "Heschl_R",
    "1008": "Parietal_Inf_L",   "ctx-lh-inferiorparietal": "Parietal_Inf_L",
    "2008": "Parietal_Inf_R",   "ctx-rh-inferiorparietal": "Parietal_Inf_R",
    "1031": "SupraMarginal_L",  "ctx-lh-supramarginal":    "SupraMarginal_L",
    "2031": "SupraMarginal_R",  "ctx-rh-supramarginal":    "SupraMarginal_R",
    "1029": "Parietal_Sup_L",   "ctx-lh-superiorparietal": "Parietal_Sup_L",
    "2029": "Parietal_Sup_R",   "ctx-rh-superiorparietal": "Parietal_Sup_R",
    "1025": "Precuneus_L",      "ctx-lh-precuneus": "Precuneus_L",
    "2025": "Precuneus_R",      "ctx-rh-precuneus": "Precuneus_R",
    "1023": "Cingulum_Post_L",  "ctx-lh-posteriorcingulate":       "Cingulum_Post_L",
    "1010": "Cingulum_Post_L",  "ctx-lh-isthmuscingulate":         "Cingulum_Post_L",
    "2023": "Cingulum_Post_R",  "ctx-rh-posteriorcingulate":       "Cingulum_Post_R",
    "2010": "Cingulum_Post_R",  "ctx-rh-isthmuscingulate":         "Cingulum_Post_R",
    "1002": "Cingulum_Ant_L",   "ctx-lh-caudalanteriorcingulate":  "Cingulum_Ant_L",
    "1026": "Cingulum_Ant_L",   "ctx-lh-rostralanteriorcingulate": "Cingulum_Ant_L",
    "2002": "Cingulum_Ant_R",   "ctx-rh-caudalanteriorcingulate":  "Cingulum_Ant_R",
    "2026": "Cingulum_Ant_R",   "ctx-rh-rostralanteriorcingulate": "Cingulum_Ant_R",
    "1035": "Insula_L",         "ctx-lh-insula": "Insula_L",
    "2035": "Insula_R",         "ctx-rh-insula": "Insula_R",
    "1011": "Occipital_Mid_L",  "ctx-lh-lateraloccipital": "Occipital_Mid_L",
    "2011": "Occipital_Mid_R",  "ctx-rh-lateraloccipital": "Occipital_Mid_R",
    "1013": "Lingual_L",        "ctx-lh-lingual":          "Lingual_L",
    "2013": "Lingual_R",        "ctx-rh-lingual":          "Lingual_R",
    "1005": "Cuneus_L",         "ctx-lh-cuneus":            "Cuneus_L",
    "2005": "Cuneus_R",         "ctx-rh-cuneus":            "Cuneus_R",
    "1021": "Calcarine_L",      "ctx-lh-pericalcarine":    "Calcarine_L",
    "2021": "Calcarine_R",      "ctx-rh-pericalcarine":    "Calcarine_R",
    "1007": "Fusiform_L",       "ctx-lh-fusiform":         "Fusiform_L",
    "2007": "Fusiform_R",       "ctx-rh-fusiform":         "Fusiform_R",
    "1016": "ParaHippocampal_L", "ctx-lh-parahippocampal": "ParaHippocampal_L",
    "2016": "ParaHippocampal_R", "ctx-rh-parahippocampal": "ParaHippocampal_R",
    "1006": "ParaHippocampal_L", "ctx-lh-entorhinal":      "ParaHippocampal_L",
    "2006": "ParaHippocampal_R", "ctx-rh-entorhinal":      "ParaHippocampal_R",
    "1017": "Paracentral_Lobule_L", "ctx-lh-paracentral": "Paracentral_Lobule_L",
    "2017": "Paracentral_Lobule_R", "ctx-rh-paracentral": "Paracentral_Lobule_R",
}

_SPECIAL_REGIONS: dict[str, dict] = {
    "brainstem":                   {"canonical_name": "Brainstem",                    "hemisphere": "midline", "lobe": "brainstem",    "networks": ["Reticular Activating System"]},
    "brain-stem":                  {"canonical_name": "Brainstem",                    "hemisphere": "midline", "lobe": "brainstem",    "networks": ["Reticular Activating System"]},
    "corpus-callosum":             {"canonical_name": "Corpus Callosum",              "hemisphere": "midline", "lobe": "white_matter", "networks": ["Interhemispheric Pathway"]},
    "cc_anterior":                 {"canonical_name": "Corpus Callosum (Genu)",       "hemisphere": "midline", "lobe": "white_matter", "networks": ["Interhemispheric Pathway"]},
    "cc_posterior":                {"canonical_name": "Corpus Callosum (Splenium)",   "hemisphere": "midline", "lobe": "white_matter", "networks": ["Interhemispheric Pathway"]},
    "left-ventraldc":              {"canonical_name": "Left VentralDC / Internal Capsule",  "hemisphere": "left",  "lobe": "subcortical", "networks": ["Corticospinal Tract"]},
    "right-ventraldc":             {"canonical_name": "Right VentralDC / Internal Capsule", "hemisphere": "right", "lobe": "subcortical", "networks": ["Corticospinal Tract"]},
    "left-accumbens-area":         {"canonical_name": "Left Accumbens",               "hemisphere": "left",  "lobe": "subcortical",  "networks": ["Reward Network", "Limbic Network"]},
    "right-accumbens-area":        {"canonical_name": "Right Accumbens",              "hemisphere": "right", "lobe": "subcortical",  "networks": ["Reward Network", "Limbic Network"]},
    "left-cerebellum-cortex":      {"canonical_name": "Left Cerebellum",              "hemisphere": "left",  "lobe": "cerebellum",   "networks": ["Cerebellar Network"]},
    "right-cerebellum-cortex":     {"canonical_name": "Right Cerebellum",             "hemisphere": "right", "lobe": "cerebellum",   "networks": ["Cerebellar Network"]},
    "left-cerebellum-white-matter":{"canonical_name": "Left Cerebellum White Matter", "hemisphere": "left",  "lobe": "cerebellum",   "networks": ["Cerebellar Network"]},
    "right-cerebellum-white-matter":{"canonical_name": "Right Cerebellum White Matter","hemisphere": "right", "lobe": "cerebellum",   "networks": ["Cerebellar Network"]},
}

_SPECIAL_SEGID_MAP: dict[str, str] = {
    "16":  "brainstem",
    "7":   "left-cerebellum-white-matter",
    "8":   "left-cerebellum-cortex",
    "46":  "right-cerebellum-white-matter",
    "47":  "right-cerebellum-cortex",
    "251": "cc_posterior",
    "252": "corpus-callosum",
    "253": "corpus-callosum",
    "254": "corpus-callosum",
    "255": "cc_anterior",
    "26":  "left-accumbens-area",
    "58":  "right-accumbens-area",
    "28":  "left-ventraldc",
    "60":  "right-ventraldc",
}

_LOBE_MAP: dict[str, str] = {
    "Frontal_Sup_Medial": "frontal",
    "Frontal_Med_Orb":    "frontal",
    "Frontal_Inf_Oper":   "frontal",
    "Frontal_Inf_Tri":    "frontal",
    "Frontal_Inf_Orb":    "frontal",
    "Frontal_Sup_Orb":    "frontal",
    "Frontal_Mid_Orb":    "frontal",
    "Frontal_Sup":        "frontal",
    "Frontal_Mid":        "frontal",
    "Supp_Motor_Area":    "frontal",
    "Rolandic_Oper":      "frontal",
    "Olfactory":          "frontal",
    "Rectus":             "frontal",
    "Precentral":         "frontal",
    "Paracentral_Lobule": "parietal",
    "Parietal_Sup":       "parietal",
    "Parietal_Inf":       "parietal",
    "SupraMarginal":      "parietal",
    "Angular":            "parietal",
    "Precuneus":          "parietal",
    "Postcentral":        "parietal",
    "Cingulum_Ant":       "cingulate",
    "Cingulum_Mid":       "cingulate",
    "Cingulum_Post":      "cingulate",
    "Hippocampus":        "limbic",
    "ParaHippocampal":    "limbic",
    "Amygdala":           "limbic",
    "Temporal_Pole_Sup":  "temporal",
    "Temporal_Pole_Mid":  "temporal",
    "Temporal_Sup":       "temporal",
    "Temporal_Mid":       "temporal",
    "Temporal_Inf":       "temporal",
    "Heschl":             "temporal",
    "Fusiform":           "temporal",
    "Insula":             "insular",
    "Caudate":            "subcortical",
    "Putamen":            "subcortical",
    "Pallidum":           "subcortical",
    "Thalamus":           "subcortical",
    "Calcarine":          "occipital",
    "Cuneus":             "occipital",
    "Lingual":            "occipital",
    "Occipital_Sup":      "occipital",
    "Occipital_Mid":      "occipital",
    "Occipital_Inf":      "occipital",
    "Cerebelum":          "cerebellum",
    "Vermis":             "cerebellum",
}

_LOBE_MAP_SORTED = sorted(_LOBE_MAP.items(), key=lambda kv: len(kv[0]), reverse=True)

_NETWORKS: dict[str, list[str]] = {
    "Cingulum_Post_L":       ["Default Mode Network"],
    "Cingulum_Post_R":       ["Default Mode Network"],
    "Precuneus_L":           ["Default Mode Network"],
    "Precuneus_R":           ["Default Mode Network"],
    "Angular_L":             ["Default Mode Network"],
    "Angular_R":             ["Default Mode Network"],
    "Frontal_Med_Orb_L":     ["Default Mode Network"],
    "Frontal_Med_Orb_R":     ["Default Mode Network"],
    "Frontal_Sup_Medial_L":  ["Default Mode Network"],
    "Frontal_Sup_Medial_R":  ["Default Mode Network"],
    "Hippocampus_L":         ["Default Mode Network", "Memory Network"],
    "Hippocampus_R":         ["Default Mode Network", "Memory Network"],
    "Caudate_L":             ["Frontostriatal Network", "Executive Network"],
    "Caudate_R":             ["Frontostriatal Network", "Executive Network"],
    "Putamen_L":             ["Frontostriatal Network", "Sensorimotor Network"],
    "Putamen_R":             ["Frontostriatal Network", "Sensorimotor Network"],
    "Pallidum_L":            ["Frontostriatal Network"],
    "Pallidum_R":            ["Frontostriatal Network"],
    "Thalamus_L":            ["Thalamo-Cortical Network"],
    "Thalamus_R":            ["Thalamo-Cortical Network"],
    "Frontal_Mid_L":         ["Executive Network", "Frontoparietal Control Network"],
    "Frontal_Mid_R":         ["Executive Network", "Frontoparietal Control Network"],
    "Frontal_Sup_L":         ["Executive Network", "Default Mode Network"],
    "Frontal_Sup_R":         ["Executive Network", "Default Mode Network"],
    "Frontal_Sup_Medial_L":  ["Executive Network"],
    "Frontal_Sup_Medial_R":  ["Executive Network"],
    "Frontal_Inf_Oper_L":    ["Language Network", "Frontoparietal Control Network"],
    "Frontal_Inf_Oper_R":    ["Language Network", "Frontoparietal Control Network"],
    "Frontal_Inf_Tri_L":     ["Language Network"],
    "Frontal_Inf_Tri_R":     ["Language Network"],
    "Temporal_Sup_L":        ["Language Network", "Auditory Network"],
    "Temporal_Sup_R":        ["Language Network", "Auditory Network"],
    "Heschl_L":              ["Auditory Network"],
    "Heschl_R":              ["Auditory Network"],
    "Insula_L":              ["Salience Network"],
    "Insula_R":              ["Salience Network"],
    "Cingulum_Ant_L":        ["Salience Network"],
    "Cingulum_Ant_R":        ["Salience Network"],
    "Amygdala_L":            ["Limbic Network", "Salience Network"],
    "Amygdala_R":            ["Limbic Network", "Salience Network"],
    "Precentral_L":          ["Sensorimotor Network"],
    "Precentral_R":          ["Sensorimotor Network"],
    "Postcentral_L":         ["Sensorimotor Network"],
    "Postcentral_R":         ["Sensorimotor Network"],
    "Supp_Motor_Area_L":     ["Sensorimotor Network"],
    "Supp_Motor_Area_R":     ["Sensorimotor Network"],
    "Paracentral_Lobule_L":  ["Sensorimotor Network"],
    "Paracentral_Lobule_R":  ["Sensorimotor Network"],
    "Parietal_Sup_L":        ["Dorsal Attention Network"],
    "Parietal_Sup_R":        ["Dorsal Attention Network"],
    "Parietal_Inf_L":        ["Dorsal Attention Network", "Default Mode Network"],
    "Parietal_Inf_R":        ["Dorsal Attention Network"],
    "Calcarine_L":           ["Visual Network"],
    "Calcarine_R":           ["Visual Network"],
    "Cuneus_L":              ["Visual Network"],
    "Cuneus_R":              ["Visual Network"],
    "Lingual_L":             ["Visual Network"],
    "Lingual_R":             ["Visual Network"],
    "Occipital_Sup_L":       ["Visual Network"],
    "Occipital_Sup_R":       ["Visual Network"],
    "Occipital_Mid_L":       ["Visual Network"],
    "Occipital_Mid_R":       ["Visual Network"],
    "Occipital_Inf_L":       ["Visual Network"],
    "Occipital_Inf_R":       ["Visual Network"],
    "Fusiform_L":            ["Ventral Visual Stream"],
    "Fusiform_R":            ["Ventral Visual Stream"],
}

_ADHD_KEY_LABELS: set[str] = {
    "Caudate_L", "Caudate_R",
    "Putamen_L", "Putamen_R",
    "Frontal_Inf_Oper_R",
    "Frontal_Mid_L", "Frontal_Mid_R",
    "Frontal_Sup_L", "Frontal_Sup_R",
    "Cingulum_Post_L", "Cingulum_Post_R",
    "Cingulum_Ant_L", "Cingulum_Ant_R",
    "Insula_R",
    "Precuneus_L", "Precuneus_R",
    "Thalamus_L", "Thalamus_R",
}

_ELOQUENT_LABELS: set[str] = {
    "Precentral_L", "Precentral_R",
    "Postcentral_L", "Postcentral_R",
    "Temporal_Sup_L",
    "Frontal_Inf_Oper_L",
    "Frontal_Inf_Tri_L",
    "Supp_Motor_Area_L", "Supp_Motor_Area_R",
    "Calcarine_L", "Calcarine_R",
}

_VASCULAR_TERRITORY: dict[str, set[str]] = {
    "MCA": {
        "Precentral_L", "Precentral_R",
        "Postcentral_L", "Postcentral_R",
        "Frontal_Mid_L", "Frontal_Mid_R",
        "Frontal_Sup_L", "Frontal_Sup_R",
        "Frontal_Inf_Oper_L", "Frontal_Inf_Oper_R",
        "Frontal_Inf_Tri_L", "Frontal_Inf_Tri_R",
        "Temporal_Sup_L", "Temporal_Sup_R",
        "Temporal_Mid_L", "Temporal_Mid_R",
        "Parietal_Inf_L", "Parietal_Inf_R",
        "Parietal_Sup_L", "Parietal_Sup_R",
        "Insula_L", "Insula_R",
        "Caudate_L", "Caudate_R",
        "Putamen_L", "Putamen_R",
    },
    "ACA": {
        "Frontal_Sup_L", "Frontal_Sup_R",
        "Frontal_Sup_Medial_L", "Frontal_Sup_Medial_R",
        "Frontal_Med_Orb_L", "Frontal_Med_Orb_R",
        "Supp_Motor_Area_L", "Supp_Motor_Area_R",
        "Cingulum_Ant_L", "Cingulum_Ant_R",
        "Paracentral_Lobule_L", "Paracentral_Lobule_R",
    },
    "PCA": {
        "Calcarine_L", "Calcarine_R",
        "Cuneus_L", "Cuneus_R",
        "Lingual_L", "Lingual_R",
        "Occipital_Sup_L", "Occipital_Sup_R",
        "Occipital_Mid_L", "Occipital_Mid_R",
        "Occipital_Inf_L", "Occipital_Inf_R",
        "Fusiform_L", "Fusiform_R",
        "Hippocampus_L", "Hippocampus_R",
        "Amygdala_L", "Amygdala_R",
        "Thalamus_L", "Thalamus_R",
        "Cingulum_Post_L", "Cingulum_Post_R",
        "Precuneus_L", "Precuneus_R",
    },
    "PICA":    set(),
    "BASILAR": set(),
}

_TEXT_KEYWORDS: dict[str, str] = {
    "motor cortex": "Precentral_L",
    "primary motor": "Precentral_L",
    "precentral": "Precentral_L",
    "m1": "Precentral_L",
    "sensory cortex": "Postcentral_L",
    "primary somatosensory": "Postcentral_L",
    "postcentral": "Postcentral_L",
    "s1": "Postcentral_L",
    "broca": "Frontal_Inf_Oper_L",
    "broca's": "Frontal_Inf_Oper_L",
    "brocas": "Frontal_Inf_Oper_L",
    "wernicke": "Temporal_Sup_L",
    "wernicke's": "Temporal_Sup_L",
    "wernickes": "Temporal_Sup_L",
    "left dlpfc": "Frontal_Mid_L",
    "right dlpfc": "Frontal_Mid_R",
    "dlpfc": "Frontal_Mid_L",
    "left ifg": "Frontal_Inf_Oper_L",
    "right ifg": "Frontal_Inf_Oper_R",
    "inferior frontal gyrus": "Frontal_Inf_Oper_L",
    "posterior cingulate": "Cingulum_Post_L",
    "pcc": "Cingulum_Post_L",
    "anterior cingulate": "Cingulum_Ant_L",
    "acc": "Cingulum_Ant_L",
    "medial prefrontal": "Frontal_Med_Orb_L",
    "mpfc": "Frontal_Med_Orb_L",
    "vmPFC": "Frontal_Med_Orb_L",
    "orbitofrontal": "Frontal_Med_Orb_L",
    "frontal pole": "Frontal_Sup_Medial_L",
    "hippocampus": "Hippocampus_L",
    "caudate": "Caudate_L",
    "putamen": "Putamen_L",
    "pallidum": "Pallidum_L",
    "globus pallidus": "Pallidum_L",
    "thalamus": "Thalamus_L",
    "amygdala": "Amygdala_L",
    "insula": "Insula_L",
    "insular": "Insula_L",
    "right insula": "Insula_R",
    "left insula": "Insula_L",
    "anterior insula": "Insula_L",
    "precuneus": "Precuneus_L",
    "angular gyrus": "Angular_L",
    "angular": "Angular_L",
    "supramarginal": "SupraMarginal_L",
    "inferior parietal": "Parietal_Inf_L",
    "superior parietal": "Parietal_Sup_L",
    "superior frontal": "Frontal_Sup_L",
    "middle frontal": "Frontal_Mid_L",
    "inferior frontal": "Frontal_Inf_Oper_L",
    "superior temporal": "Temporal_Sup_L",
    "middle temporal": "Temporal_Mid_L",
    "inferior temporal": "Temporal_Inf_L",
    "fusiform": "Fusiform_L",
    "parahippocampal": "ParaHippocampal_L",
    "entorhinal": "ParaHippocampal_L",
    "occipital": "Occipital_Mid_L",
    "calcarine": "Calcarine_L",
    "cuneus": "Cuneus_L",
    "lingual": "Lingual_L",
    "paracentral": "Paracentral_Lobule_L",
    "sma": "Supp_Motor_Area_L",
    "supplementary motor": "Supp_Motor_Area_L",
    "primary auditory": "Heschl_L",
    "heschl": "Heschl_L",
    "brainstem": "brainstem",
    "brain stem": "brainstem",
    "pons": "brainstem",
    "medulla": "brainstem",
    "midbrain": "brainstem",
    "mesencephalon": "brainstem",
    "pontine": "brainstem",
    "corpus callosum": "corpus-callosum",
    "butterfly glioma": "corpus-callosum",
    "callosal": "corpus-callosum",
    "internal capsule": "left-ventraldc",
    "corticospinal tract": "left-ventraldc",
    "posterior limb": "left-ventraldc",
    "accumbens": "left-accumbens-area",
    "nucleus accumbens": "left-accumbens-area",
    "cerebellum": "left-cerebellum-cortex",
    "cerebellar": "left-cerebellum-cortex",
}

@dataclass
class MappingResult:
    success:    bool
    region:     Optional[AtlasRegion]
    confidence: float = 0.0
    method:     str = ""
    warnings:   list = field(default_factory=list)
    error:      Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success":    self.success,
            "region":     self.region.to_dict() if self.region else None,
            "confidence": self.confidence,
            "method":     self.method,
            "warnings":   self.warnings,
            "error":      self.error,
        }

class AtlasMapper:

    def __init__(self):
        self._initialized = False
        self._idx_to_label: dict[int, str] = {}
        self._label_to_idx: dict[str, int] = {}
        self._label_centroids: dict[str, np.ndarray] = {}
        self._all_labels: list[str] = []
        self._all_centroids: np.ndarray | None = None
        self._atlas_data: np.ndarray | None = None
        self._affine_inv: np.ndarray | None = None

    def _init(self) -> None:
        if self._initialized:
            return
        import requests
        import urllib3
        from nilearn import datasets
        import nibabel as nib

        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        _orig_send = requests.Session.send
        def _send_no_verify(self_s, req, **kw):
            kw["verify"] = False
            return _orig_send(self_s, req, **kw)
        requests.Session.send = _send_no_verify
        try:
            atlas = datasets.fetch_atlas_aal()
        finally:
            requests.Session.send = _orig_send
        img = nib.load(atlas.maps)
        data = np.asarray(img.get_fdata(), dtype=int)
        affine = img.affine

        self._affine_inv = np.linalg.inv(affine)
        self._atlas_data = data

        labels  = atlas.labels
        indices = [int(i) for i in atlas.indices]

        for idx, label in zip(indices, labels):
            self._idx_to_label[idx]  = label
            self._label_to_idx[label] = idx

        for label, idx in self._label_to_idx.items():
            voxels = np.array(np.where(data == idx)).T
            if len(voxels) == 0:
                continue
            vox_centroid = voxels.mean(axis=0)
            mni = (affine @ np.append(vox_centroid, 1.0))[:3]
            self._label_centroids[label] = mni

        self._all_labels    = list(self._label_centroids.keys())
        self._all_centroids = np.array([self._label_centroids[l] for l in self._all_labels])
        self._initialized   = True

    def _mni_to_aal(self, x: float, y: float, z: float) -> tuple[str, float]:
        self._init()

        vox = (self._affine_inv @ np.array([x, y, z, 1.0]))[:3]
        vi, vj, vk = int(round(vox[0])), int(round(vox[1])), int(round(vox[2]))
        h, w, d = self._atlas_data.shape
        if 0 <= vi < h and 0 <= vj < w and 0 <= vk < d:
            atlas_val = int(self._atlas_data[vi, vj, vk])
            if atlas_val != 0 and atlas_val in self._idx_to_label:
                label = self._idx_to_label[atlas_val]
                cent  = self._label_centroids.get(label, np.array([x, y, z]))
                dist  = float(np.linalg.norm(np.array([x, y, z]) - cent))
                return label, dist

        coord = np.array([x, y, z])
        dists = np.linalg.norm(self._all_centroids - coord, axis=1)
        best_i = int(np.argmin(dists))
        return self._all_labels[best_i], float(dists[best_i])

    def _label_to_region(self, aal_label: str) -> AtlasRegion:
        self._init()

        if aal_label.endswith("_L"):
            hemisphere = "left"
        elif aal_label.endswith("_R"):
            hemisphere = "right"
        else:
            hemisphere = "midline"

        lobe = "unknown"
        for prefix, lobe_name in _LOBE_MAP_SORTED:
            if aal_label.startswith(prefix):
                lobe = lobe_name
                break

        mni_arr = self._label_centroids.get(aal_label, np.zeros(3))
        mni = (float(mni_arr[0]), float(mni_arr[1]), float(mni_arr[2]))

        networks = _NETWORKS.get(aal_label, [])

        canonical = re.sub(r"_L$", " (Left)",  aal_label)
        canonical = re.sub(r"_R$", " (Right)", canonical)
        canonical = re.sub(r"_+", " ", canonical)

        return AtlasRegion(
            canonical_name=canonical,
            hemisphere=hemisphere,
            lobe=lobe,
            atlas_label=aal_label,
            mni_centroid=mni,
            functional_role="",
            tumor_relevance="",
            stroke_territory="",
            adhd_relevance="",
            networks=networks,
        )

    def _special_to_region(self, key: str) -> AtlasRegion:
        rec = _SPECIAL_REGIONS[key]
        return AtlasRegion(
            canonical_name=rec["canonical_name"],
            hemisphere=rec["hemisphere"],
            lobe=rec["lobe"],
            atlas_label=key,
            mni_centroid=(0.0, 0.0, 0.0),
            functional_role="",
            tumor_relevance="",
            stroke_territory="",
            adhd_relevance="",
            networks=rec.get("networks", []),
        )

    def map_freesurfer(self, label: str) -> MappingResult:
        self._init()
        label_str = str(label).strip()

        if label_str in _SPECIAL_SEGID_MAP:
            key = _SPECIAL_SEGID_MAP[label_str]
            return MappingResult(
                success=True, region=self._special_to_region(key),
                confidence=1.0, method="segid_special",
            )

        if label_str.lstrip("-").isdigit():
            aal = _FS_TO_AAL.get(label_str)
            if aal:
                return MappingResult(
                    success=True, region=self._label_to_region(aal),
                    confidence=1.0, method="segid_exact",
                )
            return MappingResult(
                success=False, region=None, confidence=0.0, method="none",
                error=f"No atlas match for FreeSurfer SegId: {label_str}",
            )

        key = label_str.lower()

        if key in _SPECIAL_REGIONS:
            return MappingResult(
                success=True, region=self._special_to_region(key),
                confidence=1.0, method="alias_special",
            )

        aal = _FS_TO_AAL.get(key)
        if aal:
            return MappingResult(
                success=True, region=self._label_to_region(aal),
                confidence=1.0, method="alias_exact",
            )

        stem = re.sub(r"^(ctx-[lr]h-|[lr]h\.|[lr]h_|left-|right-)", "", key)
        for fs_key, aal_val in _FS_TO_AAL.items():
            if stem and fs_key.endswith(stem):
                return MappingResult(
                    success=True, region=self._label_to_region(aal_val),
                    confidence=0.85, method="alias_partial",
                    warnings=[f"Partial alias match: '{label}' -> '{aal_val}'"],
                )

        for aal_label in self._label_to_idx:
            if stem and stem in aal_label.lower():
                return MappingResult(
                    success=True, region=self._label_to_region(aal_label),
                    confidence=0.75, method="aal_fuzzy",
                    warnings=[f"Fuzzy AAL match: '{label}' -> '{aal_label}'"],
                )

        return MappingResult(
            success=False, region=None, confidence=0.0, method="none",
            error=f"No atlas match for FreeSurfer label: '{label}'",
        )

    def map_mni(self, x: float, y: float, z: float) -> MappingResult:
        self._init()
        aal_label, dist_mm = self._mni_to_aal(x, y, z)
        confidence = max(0.0, 1.0 - dist_mm / 60.0)
        warnings = []
        if dist_mm > 25:
            warnings.append(
                f"Nearest region is {dist_mm:.1f} mm away, "
                "coordinate may be in unlisted region or white matter."
            )
        return MappingResult(
            success=True,
            region=self._label_to_region(aal_label),
            confidence=confidence,
            method=f"mni_aal_{dist_mm:.1f}mm",
            warnings=warnings,
        )

    def map_text(self, text: str) -> MappingResult:
        self._init()
        text_lower = text.lower().strip()

        for kw in sorted(_TEXT_KEYWORDS.keys(), key=len, reverse=True):
            if kw in text_lower:
                target = _TEXT_KEYWORDS[kw]
                if target in _SPECIAL_REGIONS:
                    return MappingResult(
                        success=True, region=self._special_to_region(target),
                        confidence=0.9, method="text_keyword",
                    )
                return MappingResult(
                    success=True, region=self._label_to_region(target),
                    confidence=0.9, method="text_keyword",
                )

        tokens = set(re.findall(r"\b\w+\b", text_lower))
        best_label, best_score = "", 0.0
        for aal_label in self._all_labels:
            aal_tokens = set(re.findall(r"\b\w+\b", aal_label.lower()))
            score = len(tokens & aal_tokens) / max(len(aal_tokens), 1)
            if score > best_score:
                best_score, best_label = score, aal_label

        if best_score >= 0.5 and best_label:
            return MappingResult(
                success=True, region=self._label_to_region(best_label),
                confidence=best_score * 0.75, method="text_token_overlap",
                warnings=[f"Low-confidence text match (score={best_score:.2f})"],
            )

        return MappingResult(
            success=False, region=None, confidence=0.0, method="none",
            error=f"No atlas match found for text: '{text}'",
        )

    def ground_structural_output(
        self, vol_table: dict[str, float]
    ) -> dict[str, MappingResult]:
        return {label: self.map_freesurfer(label) for label in vol_table}

    def ground_lesion_coords(
        self, coord_list: list[tuple[float, float, float]]
    ) -> list[MappingResult]:
        return [self.map_mni(*coord) for coord in coord_list]

    def list_regions(
        self,
        lobe: Optional[str] = None,
        network: Optional[str] = None,
        hemisphere: Optional[str] = None,
    ) -> list[AtlasRegion]:
        self._init()
        results = []
        for label in self._all_labels:
            region = self._label_to_region(label)
            if lobe and region.lobe != lobe:
                continue
            if hemisphere and region.hemisphere not in (hemisphere, "bilateral", "midline"):
                continue
            if network and not any(network.lower() in n.lower() for n in region.networks):
                continue
            results.append(region)
        return results

    def adhd_key_regions(self) -> list[AtlasRegion]:
        self._init()
        return [
            self._label_to_region(label)
            for label in _ADHD_KEY_LABELS
            if label in self._label_centroids
        ]

    def tumor_eloquent_regions(self) -> list[AtlasRegion]:
        self._init()
        return [
            self._label_to_region(label)
            for label in _ELOQUENT_LABELS
            if label in self._label_centroids
        ]

    def stroke_territory_regions(self, territory: str) -> list[AtlasRegion]:
        self._init()
        labels = _VASCULAR_TERRITORY.get(territory.upper(), set())
        return [
            self._label_to_region(label)
            for label in labels
            if label in self._label_centroids
        ]

    def stroke_territory(self, territory: str) -> list[AtlasRegion]:
        return self.stroke_territory_regions(territory)

def _run_tests(mapper: AtlasMapper) -> None:
    tests = [
        ("fs",   "Left-Caudate"),
        ("fs",   "ctx-rh-precentral"),
        ("fs",   "17"),
        ("fs",   "Left-Hippocampus"),
        ("fs",   "CC_Anterior"),
        ("mni",  (-36, 0, 4)),
        ("mni",  (0, 28, 20)),
        ("mni",  (26, -18, -18)),
        ("text", "motor cortex near the hand area"),
        ("text", "posterior cingulate"),
        ("text", "right insula"),
        ("text", "broca's area"),
    ]

    print("AtlasMapper NeuroAgent test suite")
    for mode, inp in tests:
        if mode == "fs":
            r = mapper.map_freesurfer(inp)
            label = inp
        elif mode == "mni":
            r = mapper.map_mni(*inp)
            label = str(inp)
        else:
            r = mapper.map_text(inp)
            label = f'"{inp}"'

        status = "OK" if r.success else "FAIL"
        name   = r.region.canonical_name if r.region else "---"
        print(f"[{status}] {label:<40} -> {name} (conf={r.confidence:.2f}, method={r.method})")
        for w in r.warnings:
            print(f"      WARN: {w}")

    print()
    print("ADHD key regions:", [r.canonical_name for r in mapper.adhd_key_regions()])
    print()
    print("Eloquent cortex:", [r.canonical_name for r in mapper.tumor_eloquent_regions()])
    print()
    print("MCA territory:", [r.canonical_name for r in mapper.stroke_territory("MCA")])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroAgent Atlas Mapper")
    parser.add_argument("--test",     action="store_true")
    parser.add_argument("--map-fs",   type=str, metavar="LABEL")
    parser.add_argument("--map-mni",  type=str, metavar="X,Y,Z")
    parser.add_argument("--map-text", type=str, metavar="TEXT")
    args = parser.parse_args()

    mapper = AtlasMapper()

    if args.test:
        _run_tests(mapper)
    elif args.map_fs:
        r = mapper.map_freesurfer(args.map_fs)
        print(json.dumps(r.to_dict(), indent=2, default=str))
    elif args.map_mni:
        xyz = [float(v) for v in args.map_mni.split(",")]
        r = mapper.map_mni(*xyz)
        print(json.dumps(r.to_dict(), indent=2, default=str))
    elif args.map_text:
        r = mapper.map_text(args.map_text)
        print(json.dumps(r.to_dict(), indent=2, default=str))
    else:
        parser.print_help()
