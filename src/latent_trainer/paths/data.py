# ---------------------------------------------------------------------------
# Model loading and encoding
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from latent_trainer.features.data_utils import load_and_normalize
from latent_trainer.models.lightning.lit_guided_vae import LitGuidedVAE

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
    model_path: Path,
    cached_dataset_filepath: Path,
) -> tuple[LitGuidedVAE, torch.Tensor, torch.Tensor]:

    # TODO: If there are other models than Guided VAE, this function
    # TODO: will need to take the model class as an argument instead of hardcoding:
    vae_model = LitGuidedVAE.load_from_checkpoint(checkpoint_path=model_path)
    vae_model.eval()

    data = load_and_normalize(cached_dataset_filepath=cached_dataset_filepath)

    return vae_model, data.test_X, data.test_y


# Opponent-aware score / logit functions
#
# These closures adapt the two-stage (player_z, opponent_z) classifier
# into the generic callable interface expected by path strategies.
def opponent_aware_score(
    z: torch.Tensor,
    *,
    guided_vae: LitGuidedVAE,
    opponent_z: torch.Tensor,
    player_idx: int,
) -> torch.Tensor:
    """
    P(z wins) for each row in z against a fixed opponent.

    The classifier operates only on the supervised latent dims
    (first ``supervised_dim`` entries).
    We derive that count from the classifier's
    input layer so callers don't need to pass it explicitly.
    """

    classifier = guided_vae.model.classifier

    sup_dim = classifier[0].in_features // 2
    n = z.shape[0]
    z_cls = z[:, :sup_dim]
    opp_cls = opponent_z[:sup_dim].unsqueeze(0).expand(n, -1)
    if player_idx == 0:
        combined = torch.cat([z_cls, opp_cls], dim=1)
        return classifier(combined).squeeze(-1)

    combined = torch.cat([opp_cls, z_cls], dim=1)
    return 1.0 - classifier(combined).squeeze(-1)


def opponent_aware_logit(
    z: torch.Tensor,
    *,
    guided_vae: LitGuidedVAE,
    opponent_z: torch.Tensor,
    player_idx: int,
) -> torch.Tensor:
    """Pre-sigmoid logit for each row in z, with a fixed opponent.

    Strips the final Sigmoid layer from the classifier so we get
    raw logits — better gradients far from the decision boundary.
    """

    classifier = guided_vae.model.classifier

    sup_dim = classifier[0].in_features // 2
    n = z.shape[0]
    z_cls = z[:, :sup_dim]
    opp_cls = opponent_z[:sup_dim].unsqueeze(0).expand(n, -1)
    if player_idx == 0:
        combined = torch.cat([z_cls, opp_cls], dim=1)
    else:
        combined = torch.cat([opp_cls, z_cls], dim=1)
    # Forward through all layers except the final Sigmoid
    h = combined
    for layer in list(classifier.children())[:-1]:
        h = layer(h)
    logit = h.squeeze(-1)
    # Negate for player 1 so the logit sign matches "z is winning".
    return logit if player_idx == 0 else -logit


# Shared path-charting context
@dataclass
class PathContext:
    guided_vae: LitGuidedVAE
    X: torch.Tensor
    labels: np.ndarray
    labels_tensor: torch.Tensor
    latents_p0: torch.Tensor
    latents_p1: torch.Tensor
    chosen: int
    player_idx: int
    sample_z: torch.Tensor


def compute_win_latents(
    labels_tensor: torch.Tensor,
    latents_p0: torch.Tensor,
    latents_p1: torch.Tensor,
) -> torch.Tensor:
    """Return the winning player's latent for each sample (label=1 → p0 won)."""
    return torch.where(
        (labels_tensor == 1).unsqueeze(1),
        latents_p0,
        latents_p1,
    )


def compute_loss_latents(
    labels_tensor: torch.Tensor,
    latents_p0: torch.Tensor,
    latents_p1: torch.Tensor,
) -> torch.Tensor:
    """Return the losing player's latent for each sample (label=0 → p1 won, p0 lost)."""
    return torch.where(
        (labels_tensor == 0).unsqueeze(1),
        latents_p0,
        latents_p1,
    )


def prepare_path_context(
    model_path: Path,
    dataset_path: Path,
    sample_idx: int | None,
) -> PathContext:
    """Load model + data, encode both players, and select a game to analyse.

    Every game has exactly one loser. ``sample_z`` is always that player's
    latent — the starting point for the improvement path. The winning target
    (centroid, nearest-win, etc.) is determined separately by each strategy.
    """
    print("Loading model and data...")
    guided_vae, X, y = load_model_and_data(
        model_path=model_path,
        cached_dataset_filepath=dataset_path,
    )
    labels = y.numpy()
    labels_tensor = torch.tensor(labels)
    print(f"Validation: {len(X)}")

    print("Encoding into latent space...")
    latents_p0 = encode_player(vae=guided_vae.model, data=X[:, 0, :])
    latents_p1 = encode_player(vae=guided_vae.model, data=X[:, 1, :])

    n = len(labels)
    chosen = int(
        sample_idx
        if (sample_idx is not None and sample_idx < n)
        else torch.randint(n, (1,)).item()
    )
    player_idx = int(labels[chosen])
    sample_z = latents_p0[chosen] if player_idx == 0 else latents_p1[chosen]
    print(
        f"Sample idx: {chosen} (label={int(labels[chosen])}, loser=player {player_idx})"
    )

    return PathContext(
        guided_vae=guided_vae,
        X=X,
        labels=labels,
        labels_tensor=labels_tensor,
        latents_p0=latents_p0,
        latents_p1=latents_p1,
        chosen=chosen,
        player_idx=player_idx,
        sample_z=sample_z,
    )
