"""数据/特征层：manifest、npz 特征、bbox 几何、关系负采样、头输入拼装。

对外常用入口（全部只依赖 numpy，不依赖 torch）::

    from starlife_eval.dataset import (
        load_manifest, load_features, build_relation_data,
        build_slot_feat, build_type_feat, build_pair_feat, relation_onehot,
        pair_geometry, bbox_map_from_manifest,
    )
"""

from .features import (
    FeatureError,
    feature_dim,
    load_features,
    pair_base_dim,
    relation_onehot_dim,
    slot_feature_dim,
    type_feature_dim,
)
from .geometry import (
    GEOMETRY_DIM,
    IMAGE_H,
    IMAGE_W,
    bbox_map_from_manifest,
    bboxes_for_global_slots,
    pair_geometry,
    pair_geometry_from_manifest,
)
from .manifest import (
    MANIFEST_NAME,
    ManifestError,
    iter_global_slots,
    load_manifest,
    manifest_pair_map,
    split_pairs,
)
from .negative_sampling import (
    DEFAULT_SEED,
    SAMPLE_MULT,
    build_relation_data,
    n_negative,
    n_positive,
)
from .pair_data import (
    build_pair_base,
    build_pair_feat,
    build_pair_inputs,
    build_rel_pos_feat,
    build_slot_feat,
    build_slot_inputs,
    build_type_feat,
    head_input_dims,
    relation_onehot,
)

__all__ = [
    "FeatureError",
    "GEOMETRY_DIM",
    "IMAGE_H",
    "IMAGE_W",
    "MANIFEST_NAME",
    "ManifestError",
    "DEFAULT_SEED",
    "SAMPLE_MULT",
    "bbox_map_from_manifest",
    "bboxes_for_global_slots",
    "build_pair_base",
    "build_pair_feat",
    "build_pair_inputs",
    "build_rel_pos_feat",
    "build_relation_data",
    "build_slot_feat",
    "build_slot_inputs",
    "build_type_feat",
    "feature_dim",
    "head_input_dims",
    "iter_global_slots",
    "load_features",
    "load_manifest",
    "manifest_pair_map",
    "n_negative",
    "n_positive",
    "pair_base_dim",
    "pair_geometry",
    "pair_geometry_from_manifest",
    "relation_onehot",
    "relation_onehot_dim",
    "slot_feature_dim",
    "split_pairs",
    "type_feature_dim",
]
