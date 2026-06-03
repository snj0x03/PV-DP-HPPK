# Near-duplicate frame filtering using average perceptual hashing.
# Designed to remove redundant adjacent frames extracted from the same video.
# Uses only numpy — no additional dependencies required.

import cv2
import numpy as np


def average_hash(image: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """
    Compute an average hash for an RGB image.

    The image is resized to hash_size×hash_size in grayscale, then each pixel
    is compared to the mean — pixels at or above the mean become True (1),
    below become False (0). The result is a flat boolean array of hash_size²
    bits that acts as a compact visual fingerprint.
    """
    gray    = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (hash_size, hash_size), interpolation=cv2.INTER_AREA)
    return (resized >= resized.mean()).flatten()


def hamming_distance(hash_a: np.ndarray, hash_b: np.ndarray) -> int:
    """Count the number of bit positions where two hashes differ."""
    return int(np.count_nonzero(hash_a != hash_b))


def filter_duplicates(
    frames: list,
    hash_size: int = 8,
    threshold: int = 5,
) -> list:
    """
    Remove near-duplicate frames from a list using average-hash comparison.

    A frame is kept only if its hash differs from every previously kept hash
    by more than threshold bits. threshold=5 means "accept frames that differ
    in at least 6 out of 64 hash bits" — this catches frames that are nearly
    identical due to video compression or very slow camera movement.

    Scoped to a single video's frame list; runs inside each worker process.
    """
    seen_hashes  = []
    unique_frames = []

    for frame in frames:
        h      = average_hash(frame, hash_size)
        is_dup = any(hamming_distance(h, s) <= threshold for s in seen_hashes)
        if not is_dup:
            seen_hashes.append(h)
            unique_frames.append(frame)

    return unique_frames
