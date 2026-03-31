# ---------------------------------------------------------------------------
# Model loading and encoding
# ---------------------------------------------------------------------------
import torch

from latent_trainer.data_utils import load_and_normalize
from latent_trainer.models.lightning.lit_classifier import LatentClassifier

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


@torch.no_grad()
def _encode_player(vae, data: torch.Tensor) -> torch.Tensor:
    mus = []
    for i in range(0, len(data), 256):
        mu, _ = vae.encode(data[i : i + 256])
        mus.append(mu)
    return torch.cat(mus, dim=0)


@torch.no_grad()
def _decode_features(vae, z, norm_mean, norm_std) -> np.ndarray:
    recon_norm = vae.decode(z)
    return (recon_norm * norm_std + norm_mean).cpu().numpy()


def _load_model_and_data(model_path: str, cache_path: str) -> tuple:
    checkpoint = torch.load(model_path, weights_only=False)
    latent_dim = checkpoint["latent_dim"]
    input_dim = checkpoint["input_dim"]

    vae = SimpleVAE(input_dim=input_dim, latent_dim=latent_dim)
    vae.load_state_dict(checkpoint["vae_state"])
    vae.eval()

    classifier = LatentClassifier(latent_dim=latent_dim)
    classifier.load_state_dict(checkpoint["classifier_state"])
    classifier.eval()

    norm_mean = checkpoint["normalization"]["mean"]
    norm_std = checkpoint["normalization"]["std"]
    _, _, val_X, val_y, _, _ = load_and_normalize(cache_path)

    return vae, classifier, val_X, val_y, norm_mean, norm_std, latent_dim


# ---------------------------------------------------------------------------
# Opponent-aware score / logit functions
#
# These closures adapt the two-stage (player_z, opponent_z) classifier
# into the generic callable interface expected by path strategies.
# ---------------------------------------------------------------------------
def _opponent_aware_score(
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


def _opponent_aware_logit(
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
