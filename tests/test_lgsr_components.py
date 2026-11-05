from __future__ import annotations

import torch


def _head_inputs():
    torch.manual_seed(13)
    features = torch.randn(1, 32, 2, 3)
    target = torch.tensor([[[0, 1, 2], [2, 1, 0]]], dtype=torch.long)
    return features, target


def test_lgsr_head_returns_class_and_component_scores():
    from lgsr.models.gmm_head import LGSRHead

    head = LGSRHead(32, num_classes=3, components_per_class=(1, 3, 3))
    features, _ = _head_inputs()

    payload = head(features, return_components=True)
    assert payload["logits"].shape == (1, 3, 2, 3)
    assert payload["component_scores"].shape == (1, 3, 3, 2, 3)
    assert payload["valid_component_mask"].tolist() == [
        [True, False, False],
        [True, True, True],
        [True, True, True],
    ]
    assert torch.isfinite(payload["logits"]).all()


def test_lgsr_head_uses_consistent_gmm_parameters():
    from lgsr.models.gmm_head import LGSRHead

    head = LGSRHead(32, num_classes=3, components_per_class=(1, 3, 3))
    torch.manual_seed(17)
    mu = torch.randn_like(head.mu)
    var = torch.rand_like(head.gmm_var).add_(0.25)
    log_pi = head.log_alpha.detach().clone()
    log_pi[1, :3] = torch.log_softmax(torch.tensor([0.2, -0.1, 0.4]), dim=0)
    head.set_consistent_gmm(mu, var, log_pi)
    features, _ = _head_inputs()

    payload = head(features, return_components=True)
    assert bool(payload["gmm_ready"])
    torch.testing.assert_close(head.mu, mu)
    torch.testing.assert_close(head.gmm_var, var)
    torch.testing.assert_close(head.gmm_log_pi[1, :3].exp().sum(), torch.tensor(1.0))
    assert torch.isfinite(payload["logits"]).all()


def test_lgsr_loss_matches_configured_ce_nll_and_dice_objective():
    from lgsr.losses.lgsr_loss import LGSRSelectiveLoss
    from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
    from nnunetv2.utilities.helpers import softmax_helper_dim1

    _, target = _head_inputs()
    torch.manual_seed(23)
    logits = torch.randn(1, 3, 2, 3)
    payload = {"logits": logits}
    loss = LGSRSelectiveLoss(
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
    torch.testing.assert_close(loss(payload, target), expected, atol=1e-6, rtol=1e-6)


def test_lgsr_unet_forward_and_loss_backpropagate():
    from lgsr.losses.lgsr_loss import LGSRSelectiveLoss
    from lgsr.models.unet import LGSRUNet

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
    model = LGSRUNet(1, 3, gmm_components_per_class=(1, 3, 3), **common)
    image = torch.randn(1, 1, 16, 16)
    payload = model(image, mode="train")
    assert payload["logits"].shape == (1, 3, 16, 16)
    assert payload["features"].shape == (1, 32, 16, 16)
    target = torch.randint(0, 3, (1, 16, 16))
    loss = LGSRSelectiveLoss()(payload, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.encoder.stages[0][0].convs[0].conv.weight.grad is not None
    assert model.decoder.classify.mu.grad is not None
