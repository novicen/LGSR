from __future__ import annotations

import torch


def _head_inputs():
    torch.manual_seed(13)
    features = torch.randn(1, 32, 2, 3)
    target = torch.tensor([[[0, 1, 2], [2, 1, 0]]], dtype=torch.long)
    return features, target


def test_release_head_matches_reference_before_consistent_gmm():
    from lgsr.models.gmm_head import LGSRHead
    from nnunetv2.nets.UNet_imbalance import GMMPosteriorHead

    reference = GMMPosteriorHead(32, num_classes=3, components_per_class=(1, 3, 3))
    release = LGSRHead(32, num_classes=3, components_per_class=(1, 3, 3))
    release.load_state_dict(reference.state_dict())
    features, _ = _head_inputs()

    reference_payload = reference(features, return_components=True)
    release_payload = release(features, return_components=True)
    for key in ("logits", "component_scores", "mu", "log_alpha", "gmm_var", "valid_component_mask"):
        torch.testing.assert_close(release_payload[key], reference_payload[key], atol=0.0, rtol=0.0)


def test_release_head_matches_reference_after_consistent_gmm():
    from lgsr.models.gmm_head import LGSRHead
    from nnunetv2.nets.UNet_imbalance import GMMPosteriorHead

    reference = GMMPosteriorHead(32, num_classes=3, components_per_class=(1, 3, 3))
    release = LGSRHead(32, num_classes=3, components_per_class=(1, 3, 3))
    release.load_state_dict(reference.state_dict())
    torch.manual_seed(17)
    mu = torch.randn_like(reference.mu)
    var = torch.rand_like(reference.gmm_var).add_(0.25)
    log_pi = reference.log_alpha.detach().clone()
    log_pi[1, :3] = torch.log_softmax(torch.tensor([0.2, -0.1, 0.4]), dim=0)
    reference.set_consistent_gmm(mu, var, log_pi)
    release.set_consistent_gmm(mu, var, log_pi)
    features, _ = _head_inputs()

    reference_payload = reference(features, return_components=True)
    release_payload = release(features, return_components=True)
    for key in ("logits", "component_scores", "mu", "log_alpha", "gmm_var", "valid_component_mask"):
        torch.testing.assert_close(release_payload[key], reference_payload[key], atol=0.0, rtol=0.0)


def test_release_loss_matches_reference_without_gate():
    from lgsr.losses.lgsr_loss import LGSRSelectiveLoss
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    from nnunetv2.utilities.helpers import softmax_helper_dim1

    features, target = _head_inputs()
    torch.manual_seed(23)
    logits = torch.randn(1, 3, 2, 3)
    payload = {"logits": logits, "features": features}
    release = LGSRSelectiveLoss(
        use_gmm_nll=True,
        gmm_nll_weight=0.01,
        use_strict_safe_ball=False,
        use_dice=True,
        dice_weight=1.0,
        dice_batch_dice=True,
        feature_dim=32,
    )
    ce = torch.nn.functional.cross_entropy(logits, target, reduction="none").mean()
    class_nll = torch.stack([(-logits[:, cls][target == cls]).mean() for cls in (0, 1, 2)]).mean()
    dice = MemoryEfficientSoftDiceLoss(
        apply_nonlin=softmax_helper_dim1, batch_dice=True, do_bg=False, ddp=False
    )(logits, target)
    expected = ce + 0.01 * class_nll + dice
    torch.testing.assert_close(release(payload, target), expected, atol=1e-6, rtol=1e-6)


def test_release_loss_and_gradients_match_reference_with_gate():
    import pytest

    from lgsr.losses.lgsr_loss import LGSRSelectiveLoss
    from lgsr.models.gmm_head import LGSRHead
    from nnunetv2.nets.UNet_imbalance import GMMPosteriorHead
    try:
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_imbalance import GMMPosteriorCELoss
    except ModuleNotFoundError:
        pytest.skip("project reference trainer is only available in the source workspace")

    features_ref, target = _head_inputs()
    features_ref.requires_grad_(True)
    features_new = features_ref.detach().clone().requires_grad_(True)
    reference_head = GMMPosteriorHead(32, num_classes=3, components_per_class=(1, 3, 3))
    release_head = LGSRHead(32, num_classes=3, components_per_class=(1, 3, 3))
    release_head.load_state_dict(reference_head.state_dict())
    ref_payload = reference_head(features_ref, return_components=True)
    ref_payload["features"] = features_ref
    new_payload = release_head(features_new, return_components=True)
    new_payload["features"] = features_new
    reference = GMMPosteriorCELoss(
        use_gmm_nll=True, gmm_nll_weight=0.01, use_strict_safe_ball=True,
        use_dice=True, dice_weight=1.0, dice_batch_dice=True, feature_dim=32,
        require_safe_prediction_match=False,
    )
    release = LGSRSelectiveLoss(
        use_gmm_nll=True, gmm_nll_weight=0.01, use_strict_safe_ball=True,
        use_dice=True, dice_weight=1.0, dice_batch_dice=True, feature_dim=32,
        require_safe_prediction_match=False,
    )

    ref_loss = reference(ref_payload, target)
    new_loss = release(new_payload, target)
    torch.testing.assert_close(new_loss, ref_loss, atol=1e-6, rtol=1e-6)
    ref_loss.backward()
    new_loss.backward()
    torch.testing.assert_close(features_new.grad, features_ref.grad, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(release_head.mu.grad, reference_head.mu.grad, atol=1e-6, rtol=1e-6)


def test_release_unet_loads_original_network_state_and_matches_outputs():
    from lgsr.models.unet import LGSRUNet
    from lgsr.losses.lgsr_loss import LGSRSelectiveLoss
    from nnunetv2.nets.UNet_imbalance import UNet_imbalance_ema

    common = {
        "n_stages": 2,
        "features_per_stage": [32, 64],
        "conv_op": torch.nn.Conv2d,
        "kernel_sizes": [[3, 3], [3, 3]],
        "strides": [[1, 1], [2, 2]],
        "n_conv_per_stage": [1, 1],
        "n_conv_per_stage_decoder": [1],
        "conv_bias": True,
        "norm_op": torch.nn.InstanceNorm2d,
        "norm_op_kwargs": {"eps": 1e-5, "affine": True},
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": torch.nn.LeakyReLU,
        "nonlin_kwargs": {"inplace": True},
        "deep_supervision": False,
    }
    reference = UNet_imbalance_ema(1, 3, mode="gmm_posterior", gmm_components_per_class=(1, 3, 3), **common)
    release = LGSRUNet(1, 3, gmm_components_per_class=(1, 3, 3), **common)
    release.load_state_dict(reference.state_dict(), strict=True)
    image = torch.randn(1, 1, 16, 16)
    ref_train = reference(image, mode="train")
    new_train = release(image, mode="train")
    for key in ("logits", "component_scores", "features"):
        torch.testing.assert_close(new_train[key], ref_train[key], atol=0.0, rtol=0.0)
    torch.testing.assert_close(release(image).argmax(1), reference(image).argmax(1), atol=0, rtol=0)
    target = torch.randint(0, 3, (1, 16, 16))
    loss = LGSRSelectiveLoss()(new_train, target)
    loss.backward()
    assert release.encoder.stages[0][0].convs[0].conv.weight.grad is not None
