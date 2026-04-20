# ---------------------------------------------------------------------------
# Model loading and encoding
# ---------------------------------------------------------------------------
import numpy as np
import torch

from latent_trainer.features.data_utils import load_and_normalize
from latent_trainer.models.lightning.lit_classifier import LatentClassifier
from latent_trainer.models.lightning.lit_vae import LitVAE

# ---------------------------------------------------------------------------
# Feature names -- 203 per player
# ---------------------------------------------------------------------------
_ECON_FIELDS: list[str] = [
    "foodMade",
    "foodUsed",
    "mineralsCollectionRate",
    "mineralsCurrent",
    "mineralsFriendlyFireArmy",
    "mineralsFriendlyFireEconomy",
    "mineralsFriendlyFireTechnology",
    "mineralsKilledArmy",
    "mineralsKilledEconomy",
    "mineralsKilledTechnology",
    "mineralsLostArmy",
    "mineralsLostEconomy",
    "mineralsLostTechnology",
    "mineralsUsedActiveForces",
    "mineralsUsedCurrentArmy",
    "mineralsUsedCurrentEconomy",
    "mineralsUsedCurrentTechnology",
    "mineralsUsedInProgressArmy",
    "mineralsUsedInProgressEconomy",
    "mineralsUsedInProgressTechnology",
    "vespeneCollectionRate",
    "vespeneCurrent",
    "vespeneFriendlyFireArmy",
    "vespeneFriendlyFireEconomy",
    "vespeneFriendlyFireTechnology",
    "vespeneKilledArmy",
    "vespeneKilledEconomy",
    "vespeneKilledTechnology",
    "vespeneLostArmy",
    "vespeneLostEconomy",
    "vespeneLostTechnology",
    "vespeneUsedActiveForces",
    "vespeneUsedCurrentArmy",
    "vespeneUsedCurrentEconomy",
    "vespeneUsedCurrentTechnology",
    "vespeneUsedInProgressArmy",
    "vespeneUsedInProgressEconomy",
    "vespeneUsedInProgressTechnology",
    "workersActiveCount",
]


def _build_feature_names() -> list[str]:
    prefixes = ["early", "mid", "late", "final", "econDelta"]
    names: list[str] = []
    for prefix in prefixes:
        for field in _ECON_FIELDS:
            names.append(f"{prefix}_{field}")
    names.extend(["APM", "MMR", "SQ", "SupplyCapped%"])
    names.extend(["UnitsBorn", "UnitsKilled"])
    names.extend(["UpgradeCount", "GameDuration"])
    assert len(names) == 203
    return names


FEATURE_NAMES: list[str] = _build_feature_names()


def nearest_winning_target(sample_z, win_latents, k=5) -> torch.Tensor:
    dists = torch.cdist(sample_z.unsqueeze(0), win_latents.unsqueeze(0)).squeeze(0)
    _, indices = dists.topk(k, largest=False)
    return win_latents[indices.squeeze()].mean(dim=0)


@torch.no_grad()
def encode_player(vae, data: torch.Tensor) -> torch.Tensor:
    mus = []
    for i in range(0, len(data), 256):
        mu, _ = vae.encode(data[i : i + 256])
        mus.append(mu)
    return torch.cat(mus, dim=0)


@torch.no_grad()
def decode_features(vae, z, norm_mean, norm_std) -> np.ndarray:
    recon_norm = vae.decode(z)
    return (recon_norm * norm_std + norm_mean).cpu().numpy()


def load_model_and_data(
    model_path: str,
    cache_path: str,
) -> tuple:
    info = torch.load(model_path, weights_only=False)

    vae = LitVAE.load_from_checkpoint(info["vae_ckpt_path"])
    vae.eval()

    classifier = LatentClassifier.load_from_checkpoint(info["cls_ckpt_path"])
    classifier.eval()

    norm_mean = info["normalization"]["mean"]
    norm_std = info["normalization"]["std"]
    data = load_and_normalize(cache_path)

    return vae, classifier, data.val_X, data.val_y, norm_mean, norm_std, vae.latent_dim


# ---------------------------------------------------------------------------
# Opponent-aware score / logit functions
#
# These closures adapt the two-stage (player_z, opponent_z) classifier
# into the generic callable interface expected by path strategies.
# ---------------------------------------------------------------------------
def opponent_aware_score(
    z: torch.Tensor,
    *,
    classifier: LatentClassifier,
    opponent_z: torch.Tensor,
    player_idx: int,
) -> torch.Tensor:
    """P(z wins) for each row in z against a fixed opponent."""
    n = z.shape[0]
    opp = opponent_z.unsqueeze(0).expand(n, -1)
    if player_idx == 0:
        combined = torch.cat([z, opp], dim=1)
        return classifier(combined).squeeze(-1)
    else:
        combined = torch.cat([opp, z], dim=1)
        return 1.0 - classifier(combined).squeeze(-1)


def opponent_aware_logit(
    z: torch.Tensor,
    *,
    classifier: LatentClassifier,
    opponent_z: torch.Tensor,
    player_idx: int,
) -> torch.Tensor:
    """Pre-sigmoid logit for each row in z, with a fixed opponent.

    Strips the final Sigmoid layer from the classifier so we get
    raw logits — better gradients far from the decision boundary.
    """
    n = z.shape[0]
    opp = opponent_z.unsqueeze(0).expand(n, -1)
    if player_idx == 0:
        combined = torch.cat([z, opp], dim=1)
    else:
        combined = torch.cat([opp, z], dim=1)
    # Forward through all layers except the final Sigmoid
    h = combined
    for layer in list(classifier.net.children())[:-1]:
        h = layer(h)
    logit = h.squeeze(-1)
    # Negate for player 1 so the logit sign matches "z is winning".
    return logit if player_idx == 0 else -logit
