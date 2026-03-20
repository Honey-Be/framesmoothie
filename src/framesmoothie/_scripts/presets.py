from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable

from framesmoothie.predictor import ZoneEdgeSpec


def _baseline() -> Dict[str, Any]:
    return {
        "pre_bridge_map": "log1p_arsinh",
        "pre_bridge_channels": None,
        "use_zoning": False,
        "use_zone_prediction": False,
        "zone_pred_edges": (),
        "lambda_pred": 0.0,
    }


def _zoning() -> Dict[str, Any]:
    cfg = _baseline()
    cfg.update({
        "use_zoning": True,
    })
    return cfg


def _pred() -> Dict[str, Any]:
    cfg = _zoning()
    cfg.update({
        "use_zone_prediction": True,
        "lambda_pred": 0.1,
        "zone_pred_edges": (
            ZoneEdgeSpec(
                srcs=("structure",),
                tgt="boundary",
                weight=1.0,
                predictor_kind="dwsep_conv",
                loss_type="smooth_l1",
            ),
            ZoneEdgeSpec(
                srcs=("content", "structure"),
                tgt="label",
                weight=0.5,
                predictor_kind="bilinear_mlp",
                projector_kind="bnfree_mlp",
                projector_dim=16,
                projector_share_mode="shared",
                projector_post_norm="l2",
                loss_type="simsiam",
            ),
        ),
    })
    return cfg


def _full() -> Dict[str, Any]:
    cfg = _pred()
    cfg.update({
        "use_lrca": True,
        "lrca_rank_shared": 4,
        "lrca_rank_private": 2,
        "lrca_fmlm_rank": 4,
        "lrca_fmlm_eta": 0.1,
    })
    return cfg

def _full_lite_instance_friendly() -> Dict[str, Any]:
    """
    A softer variant of `full` intended to preserve the semantic gains of the
    full preset while reducing the observed degradation on instance losses.

    Main changes versus `full`:
      - smaller global predictive weight (`lambda_pred`)
      - weaker label-edge coupling
      - lighter label-edge loss (`cosine` instead of representation-heavier SSL losses)
      - structure zone is reintroduced into the instance branch
      - boundary-aware HMC strength is reduced
    """
    cfg = _pred()
    cfg.update({
        # keep LRCA/FMLM adaptive capacity, but retain the same moderate ranks as `full`
        "use_lrca": True,
        "lrca_rank_shared": 4,
        "lrca_rank_private": 2,
        "lrca_fmlm_rank": 4,
        "lrca_fmlm_eta": 0.1,

        # reduce pressure from predictive losses so instance branch has more freedom
        "lambda_pred": 0.05,

        # give instance branch direct access to structural cues again
        "instance_zone_names": ("content", "structure", "label", "boundary"),

        # soften boundary prior influence inside HMC to avoid over-constraining masks
        "hmc_kwargs": {
            "lambda_region_boundary": 0.10,
            "lambda_superpixel_boundary": 0.25,
            "lambda_pixel_boundary": 0.25,
        },

        # predictive graph: keep structure->boundary strong, weaken label edge
        "zone_pred_edges": (
            ZoneEdgeSpec(
                srcs=("structure",),
                tgt="boundary",
                weight=1.0,
                predictor_kind="dwsep_conv",
                loss_type="smooth_l1",
            ),
            ZoneEdgeSpec(
                srcs=("content", "structure"),
                tgt="label",
                weight=0.2,
                predictor_kind="bilinear_mlp",
                loss_type="cosine",
            ),
        ),
    })
    return cfg

def _full_lite_target_instance_friendly() -> Dict[str, Any]:
    cfg = _full_lite_instance_friendly()
    cfg.update({
        # Further ease predictive pressure to help target-instance recovery.
        "lambda_pred": 0.05,
        # Keep instance structure access, but reduce label edge influence one more step.
        "zone_pred_edges": (
            ZoneEdgeSpec(
                srcs=("structure",),
                tgt="boundary",
                weight=0.75,
                predictor_kind="dwsep_conv",
                loss_type="smooth_l1",
            ),
            ZoneEdgeSpec(
                srcs=("content", "structure"),
                tgt="label",
                weight=0.15,
                predictor_kind="bilinear_mlp",
                loss_type="cosine",
            ),
        ),
        # Further soften target mask conservatism from boundary-aware HMC.
        "hmc_kwargs": {
            "lambda_region_boundary": 0.08,
            "lambda_superpixel_boundary": 0.20,
            "lambda_pixel_boundary": 0.20,
        },
    })
    return cfg


def _full_lite_target_instance_friendly_v3() -> Dict[str, Any]:
    """
    A v3 refinement of full_lite_target_instance_friendly.

    Rationale:
      - keep the better-balanced predictive pressure from the v2 family
      - preserve semantic performance
      - explicitly rebalance the instance branch by weighted zone fusion:
          * emphasize structure
          * de-emphasize label / boundary a bit
    """
    cfg = _full_lite_target_instance_friendly()
    cfg.update({
        # weighted instance fusion is the main new ingredient in v3
        "instance_zone_weights": {
            "content": 1.0,
            "structure": 1.25,
            "label": 0.50,
            "boundary": 0.75,
        },
        # keep semantic fusion uniform by default
        "semantic_zone_weights": None,
    })
    return cfg

def _full_lite_target_instance_friendly_v4() -> Dict[str, Any]:
    """
    A v4 refinement of full_lite_target_instance_friendly_v3.

    Goal:
      - keep the target-instance recovery observed in v3
      - recover some of the source-instance fitting capacity that v3 sacrificed

    Main idea:
      - keep the v3 global predictive / HMC settings (they were beneficial for target instance)
      - soften structure emphasis a bit
      - restore a little more label / boundary contribution to the instance branch
    """
    cfg = _full_lite_target_instance_friendly()
    cfg.update({
        # preserve the better-balanced predictive and HMC regime from v3/v2 family
        "lambda_pred": 0.05,
        "zone_pred_edges": (
            ZoneEdgeSpec(
                srcs=("structure",),
                tgt="boundary",
                weight=0.75,
                predictor_kind="dwsep_conv",
                loss_type="smooth_l1",
            ),
            ZoneEdgeSpec(
                srcs=("content", "structure"),
                tgt="label",
                weight=0.15,
                predictor_kind="bilinear_mlp",
                loss_type="cosine",
            ),
        ),
        "hmc_kwargs": {
            "lambda_region_boundary": 0.08,
            "lambda_superpixel_boundary": 0.20,
            "lambda_pixel_boundary": 0.20,
        },
        # weighted instance fusion: keep target-instance gains while giving source instance
        # a bit more direct label/boundary support than v3.
        "instance_zone_weights": {
            "content": 1.00,
            "structure": 1.15,
            "label": 0.65,
            "boundary": 0.85,
        },
        "semantic_zone_weights": None,
    })
    return cfg


PRESET_FACTORIES = {
    "baseline": _baseline,
    "zoning": _zoning,
    "pred": _pred,
    "full": _full,
    "full_lite_instance_friendly": _full_lite_instance_friendly,
    "full_lite_target_instance_friendly": _full_lite_target_instance_friendly,
    "full_lite_target_instance_friendly_v3": _full_lite_target_instance_friendly_v3,
    "full_lite_target_instance_friendly_v4": _full_lite_target_instance_friendly_v4,
}


def get_preset(name: str) -> Dict[str, Any]:
    if name not in PRESET_FACTORIES:
        raise KeyError(f"Unknown preset '{name}'. Available: {sorted(PRESET_FACTORIES)}")
    return deepcopy(PRESET_FACTORIES[name]())


def list_presets() -> Iterable[str]:
    return PRESET_FACTORIES.keys()
