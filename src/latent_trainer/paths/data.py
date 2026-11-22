"""Model/data loading, latent encoding and opponent-aware scoring for path charting."""

# Model loading and encoding:
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from latent_trainer.features.rich_transform import (
    META_FEATURE_NAMES,
    SORTED_PLAYERSTATS_KEYS,
)
from latent_trainer.features.type import CachedDatasetFileSpec
from latent_trainer.models.lightning.lit_guided_vae import LitGuidedVAE


def _build_feature_names() -> list[str]:
    prefixes = ["early", "mid", "late", "final", "econDelta"]
    names: list[str] = []
    for prefix in prefixes:
        for field in SORTED_PLAYERSTATS_KEYS:
            names.append(f"{prefix}_{field}")
    names.extend(META_FEATURE_NAMES)
    return names


FEATURE_NAMES: list[str] = _build_feature_names()


def build_granular_feature_names(n_bins: int) -> list[str]:
    """
    Return feature names matching the ``granular`` transform layout.

    Parameters
    ----------
    n_bins : int
        Number of temporal bins.

    Returns
    -------
    list[str]
        ``[meta(1), bin00_<stat>(39), ..., bin{n_bins-1}_<stat>(39)]``.
    """
    names: list[str] = list(META_FEATURE_NAMES)
    for i in range(n_bins):
        for field in SORTED_PLAYERSTATS_KEYS:
            names.append(f"bin{i:02d}_{field}")
    return names


def get_supervised_dim(vae: LitGuidedVAE) -> int:
    """
    Return the number of supervised latent dims of a trained model.

    Parameters
    ----------
    vae : LitGuidedVAE
        Trained model.

    Returns
    -------
    int
        ``vae.model.supervised_dim``.
    """
    return vae.model.supervised_dim


def nearest_winning_target(sample_z, win_latents, k=5) -> torch.Tensor:
    """
    Return the mean of the ``k`` winning latents nearest to a sample.

    Parameters
    ----------
    sample_z : torch.Tensor
        Sample latent, ``[D]``.
    win_latents : torch.Tensor
        Winning latents, ``[N, D]``.
    k : int
        Number of neighbours, by default 5.

    Returns
    -------
    torch.Tensor
        Mean of the nearest winners, ``[D]``.
    """
    dists = torch.cdist(sample_z.unsqueeze(0), win_latents.unsqueeze(0)).squeeze(0)
    _, indices = dists.topk(k, largest=False)
    return win_latents[indices.squeeze()].mean(dim=0)


def opponent_filtered_win_target(
    opponent_z: torch.Tensor,
    win_latents: torch.Tensor,
    loss_latents: torch.Tensor,
    k: int = 50,
) -> torch.Tensor:
    """
    Return the centroid of the winners whose beaten opponent is closest to ``opponent_z``.

    Parameters
    ----------
    opponent_z : torch.Tensor
        Current opponent's latent, ``[D]``.
    win_latents : torch.Tensor
        Winning latents, ``[N, D]``.
    loss_latents : torch.Tensor
        Latent of the opponent each winner beat, ``[N, D]``.
    k : int
        Number of winners averaged, by default 50.

    Returns
    -------
    torch.Tensor
        Opponent-aware target latent, ``[D]``.
    """
    dists = torch.cdist(opponent_z.unsqueeze(0), loss_latents).squeeze(0)
    _, indices = dists.topk(k, largest=False)
    return win_latents[indices].mean(dim=0)


@torch.no_grad()
def encode_player(vae, data: torch.Tensor) -> torch.Tensor:
    """
    Encode one player's features to posterior means, in chunks of 256.

    Parameters
    ----------
    vae : suGuidedVAE
        Model exposing ``encode``.
    data : torch.Tensor
        Normalised features of one player, ``[N, F]``.

    Returns
    -------
    torch.Tensor
        Posterior means, ``[N, latent_dim]``.
    """
    mus = []
    for i in range(0, len(data), 256):
        mu, _ = vae.encode(data[i : i + 256])
        mus.append(mu)
    return torch.cat(mus, dim=0)


@torch.no_grad()
def decode_features(vae, z, norm_mean, norm_std) -> np.ndarray:
    """
    Decode latents and undo the feature normalisation.

    Parameters
    ----------
    vae : suGuidedVAE
        Model exposing ``decode``.
    z : torch.Tensor
        Latents, ``[N, latent_dim]``.
    norm_mean : torch.Tensor
        Training feature mean.
    norm_std : torch.Tensor
        Training feature std.

    Returns
    -------
    np.ndarray
        Features in original units, ``[N, F]``.
    """
    recon_norm = vae.decode(z)
    return (recon_norm * norm_std + norm_mean).cpu().numpy()


def load_model_and_data(
    model_path: Path,
    cached_dataset_filepath: Path,
) -> tuple[LitGuidedVAE, torch.Tensor, torch.Tensor]:
    """
    Load a guided VAE checkpoint and the normalised test split of a dense cache.

    Features are truncated to the model's input width and normalised with the
    checkpoint's ``mean``/``std``.

    Parameters
    ----------
    model_path : Path
        Guided VAE checkpoint.
    cached_dataset_filepath : Path
        Dense feature cache.

    Returns
    -------
    tuple[LitGuidedVAE, torch.Tensor, torch.Tensor]
        Model in eval mode, test features ``[N, 2, F]`` and test labels.
    """
    # TODO: If there are other models than Guided VAE, this function
    # TODO: will need to take the model class as an argument instead of hardcoding:
    vae_model = LitGuidedVAE.load_from_checkpoint(checkpoint_path=model_path)
    vae_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae_model = vae_model.to(device)
    print(f"  Model device: {device}")

    cached: dict[str, torch.Tensor] = torch.load(
        str(cached_dataset_filepath), weights_only=True
    )
    spec = CachedDatasetFileSpec(**cached)

    mean = vae_model.mean.cpu()
    std = vae_model.std.cpu()
    test_features = spec.test_features.float()
    input_dim = mean.shape[-1]
    if test_features.shape[-1] != input_dim:
        # The checkpoint was trained on a truncated cache (--max_input_dim
        # during a bin-count/leakage sweep) -- per-feature z-score
        # normalization is slice-invariant, so truncating the wider cache's
        # raw features to the checkpoint's own width first, before
        # normalizing with its stored mean/std, reproduces training exactly.
        test_features = test_features[..., :input_dim]
    shape = test_features.shape
    test_flat = test_features.reshape(-1, shape[-1])
    test_X = ((test_flat - mean) / std).reshape(shape).to(device)
    test_y = spec.test_labels.float().to(device)
    print(f"  Data device: X={test_X.device}, y={test_y.device}")

    return vae_model, test_X, test_y


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
    Return P(z wins) for each row of ``z`` against a fixed opponent.

    The supervised dim count is read from the classifier's input layer.

    Parameters
    ----------
    z : torch.Tensor
        Candidate latents of the analysed player, ``[N, D]``.
    guided_vae : LitGuidedVAE
        Trained model.
    opponent_z : torch.Tensor
        Opponent latent, ``[D]``.
    player_idx : int
        0 if the analysed player is player 1, 1 if player 2; sets the order of
        the classifier input.

    Returns
    -------
    torch.Tensor
        Win probabilities of the analysed player, ``[N]``.
    """
    classifier = guided_vae.model.classifier
    z = z.to(opponent_z.device)

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
    """
    Return the pre-sigmoid win logit for each row of ``z`` against a fixed opponent.

    Runs the classifier without its final Sigmoid, giving better gradients far
    from the decision boundary.

    Parameters
    ----------
    z : torch.Tensor
        Candidate latents of the analysed player, ``[N, D]``.
    guided_vae : LitGuidedVAE
        Trained model.
    opponent_z : torch.Tensor
        Opponent latent, ``[D]``.
    player_idx : int
        0 if the analysed player is player 1, 1 if player 2.

    Returns
    -------
    torch.Tensor
        Win logits of the analysed player, ``[N]``.
    """
    classifier = guided_vae.model.classifier
    z = z.to(opponent_z.device)

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
    """
    Everything a strategy needs to chart a path for one selected game.

    Parameters
    ----------
    guided_vae : LitGuidedVAE
        Trained model.
    X : torch.Tensor
        Normalised features, ``[N, 2, F]``.
    labels : np.ndarray
        Labels as numpy, ``[N]``; 1 = player 1 won.
    labels_tensor : torch.Tensor
        Labels as a tensor, ``[N]``.
    latents_p0 : torch.Tensor
        Player 1 latents, ``[N, D]``.
    latents_p1 : torch.Tensor
        Player 2 latents, ``[N, D]``.
    chosen : int
        Index of the analysed game.
    player_idx : int
        Index of the losing player in that game (0 or 1).
    sample_z : torch.Tensor
        The losing player's latent, the start of the path.
    """

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
    """
    Return the winning player's latent for each game.

    Parameters
    ----------
    labels_tensor : torch.Tensor
        Labels, ``[N]``; 1 = player 1 won.
    latents_p0 : torch.Tensor
        Player 1 latents, ``[N, D]``.
    latents_p1 : torch.Tensor
        Player 2 latents, ``[N, D]``.

    Returns
    -------
    torch.Tensor
        Winner latents, ``[N, D]``.
    """
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
    """
    Return the losing player's latent for each game.

    Parameters
    ----------
    labels_tensor : torch.Tensor
        Labels, ``[N]``; 1 = player 1 won.
    latents_p0 : torch.Tensor
        Player 1 latents, ``[N, D]``.
    latents_p1 : torch.Tensor
        Player 2 latents, ``[N, D]``.

    Returns
    -------
    torch.Tensor
        Loser latents, ``[N, D]``.
    """
    return torch.where(
        (labels_tensor == 0).unsqueeze(1),
        latents_p0,
        latents_p1,
    )


def build_path_context(
    *,
    guided_vae: "LitGuidedVAE",
    X: torch.Tensor,
    labels: "np.ndarray",
    labels_tensor: torch.Tensor,
    latents_p0: torch.Tensor,
    latents_p1: torch.Tensor,
    chosen: int,
) -> "PathContext":
    """
    Build a :class:`PathContext` from pre-encoded latents.

    Lets the orchestrator reuse one encoding pass across many games.

    Parameters
    ----------
    guided_vae : LitGuidedVAE
        Trained model.
    X : torch.Tensor
        Normalised features, ``[N, 2, F]``.
    labels : np.ndarray
        Labels as numpy.
    labels_tensor : torch.Tensor
        Labels as a tensor.
    latents_p0 : torch.Tensor
        Player 1 latents.
    latents_p1 : torch.Tensor
        Player 2 latents.
    chosen : int
        Index of the game to analyse.

    Returns
    -------
    PathContext
        Context whose ``sample_z`` is the losing player's latent.
    """
    player_idx = int(labels[chosen])
    sample_z = latents_p0[chosen] if player_idx == 0 else latents_p1[chosen]
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


def prepare_path_context(
    model_path: Path,
    dataset_path: Path,
    sample_idx: int | None,
) -> PathContext:
    """
    Load model and data, encode both players and select a game to analyse.

    Every game has exactly one loser; ``sample_z`` is that player's latent. Each
    strategy chooses its own winning target.

    Parameters
    ----------
    model_path : Path
        Guided VAE checkpoint.
    dataset_path : Path
        Dense feature cache (its test split is used).
    sample_idx : int | None
        Game index to analyse; None or an out-of-range index picks one at random.

    Returns
    -------
    PathContext
        Context for the selected game.
    """
    print("Loading model and data...")
    guided_vae, X, y = load_model_and_data(
        model_path=model_path,
        cached_dataset_filepath=dataset_path,
    )
    labels = y.cpu().numpy()
    labels_tensor = y
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
    print(
        f"Sample idx: {chosen} (label={int(labels[chosen])}, loser=player {int(labels[chosen])})"
    )

    return build_path_context(
        guided_vae=guided_vae,
        X=X,
        labels=labels,
        labels_tensor=labels_tensor,
        latents_p0=latents_p0,
        latents_p1=latents_p1,
        chosen=chosen,
    )
