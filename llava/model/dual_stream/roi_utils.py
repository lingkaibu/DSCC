"""ROI utilities -- COCO bbox to ViT patch grid, and ROI pooling (paper §3.2.1).

LLaVA-1.5 uses CLIP-ViT-L/14-336, so patch_size=14 and 336/14 = 24: the image
occupies a 24x24 = 576 patch-token grid. `bbox_to_grid_indices` maps a COCO
(x, y, w, h) box onto that grid (always covering at least one patch), and
`roi_pool_features` mean-pools the perception-layer hidden states over those
patches to produce one visual anchor per object.

`find_image_token_span` is a helper for locating the <image> placeholder in the
pre-expansion `input_ids`, since after expansion that single position becomes
`num_image_patches` consecutive tokens.
"""

from typing import List, Sequence, Tuple

import torch


def bbox_to_grid_indices(
    bbox: Sequence[float],
    img_size: Tuple[int, int],
    grid_size: int = 24,
) -> List[int]:
    """
    Map one bbox (x, y, w, h) onto a grid_size x grid_size patch grid and return
    the flat indices of the patches it covers.

    Args:
        bbox: (x, y, w, h) in pixel coordinates
        img_size: (W, H) of the original image
        grid_size: side length of the patch grid (default 24 = 336/14 for LLaVA-1.5)

    Returns:
        Flat patch indices in [0, grid_size*grid_size). Always at least one patch.
    """
    x, y, w, h = bbox
    W, H = img_size
    if W <= 0 or H <= 0:
        return []

    # normalise to [0, 1]
    nx1, ny1 = x / W, y / H
    nx2, ny2 = (x + w) / W, (y + h) / H

    # discretise onto the grid; the ceil-style upper bound guarantees >= 1 patch
    gx1 = max(0, min(grid_size - 1, int(nx1 * grid_size)))
    gy1 = max(0, min(grid_size - 1, int(ny1 * grid_size)))
    gx2 = max(gx1 + 1, min(grid_size, int(nx2 * grid_size + 0.999)))
    gy2 = max(gy1 + 1, min(grid_size, int(ny2 * grid_size + 0.999)))

    indices = []
    for gy in range(gy1, gy2):
        row_off = gy * grid_size
        for gx in range(gx1, gx2):
            indices.append(row_off + gx)
    return indices


def roi_pool_features(
    image_features: torch.Tensor,
    bboxes_per_image: List[List[Sequence[float]]],
    img_sizes: List[Tuple[int, int]],
    grid_size: int = 24,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mean-pool one ROI feature per bbox, for every image in the batch.

    Args:
        image_features: [B, N, D], N = grid_size**2 (576 by default for LLaVA-1.5)
        bboxes_per_image: length-B list; entry b is that image's list of (x, y, w, h) boxes
        img_sizes: length-B list of (W, H)
        grid_size: side length of the patch grid

    Returns:
        flat_feats: [K, D], K = sum_b len(bboxes_per_image[b])
        batch_idx:  [K], which image each ROI came from (used for cross-sample contrast)
        valid_mask: [K] bool, True if the ROI covers at least one patch (always True
                    here, since empty ROIs are dropped above)
    """
    assert image_features.dim() == 3, f"expected [B,N,D], got {image_features.shape}"
    B, N, D = image_features.shape
    assert N == grid_size * grid_size, (
        f"image_features patch count {N} != grid_size^2 {grid_size*grid_size}"
    )
    assert len(bboxes_per_image) == B == len(img_sizes)

    flat_feats: List[torch.Tensor] = []
    batch_ids: List[int] = []

    for b_i in range(B):
        boxes = bboxes_per_image[b_i]
        sz = img_sizes[b_i]
        for box in boxes:
            indices = bbox_to_grid_indices(box, sz, grid_size)
            if not indices:
                continue
            idx_t = torch.tensor(indices, device=image_features.device, dtype=torch.long)
            pooled = image_features[b_i].index_select(0, idx_t).mean(dim=0)  # [D]
            flat_feats.append(pooled)
            batch_ids.append(b_i)

    if not flat_feats:
        empty_feat = image_features.new_zeros((0, D))
        empty_idx = torch.empty((0,), dtype=torch.long, device=image_features.device)
        empty_mask = torch.empty((0,), dtype=torch.bool, device=image_features.device)
        return empty_feat, empty_idx, empty_mask

    flat_feats_t = torch.stack(flat_feats, dim=0)
    batch_idx_t = torch.tensor(batch_ids, device=image_features.device, dtype=torch.long)
    valid_mask_t = torch.ones((flat_feats_t.size(0),), dtype=torch.bool, device=flat_feats_t.device)
    return flat_feats_t, batch_idx_t, valid_mask_t


def find_image_token_span(input_ids: torch.Tensor, image_token_index: int, num_image_patches: int) -> List[Tuple[int, int]]:
    """
    [helper] Locate the <image> placeholder of every sample in the `input_ids` as
    they look *before* `prepare_inputs_labels_for_multimodal`; after expansion that
    single position becomes `num_image_patches` consecutive patch tokens.

    Args:
        input_ids: [B, L_orig], the raw sequence still holding the image_token_index placeholder
        image_token_index: usually -200
        num_image_patches: usually 576

    Returns:
        Length-B list of (start, end): the image span in the expanded hidden_states,
        as a half-open index range [start, end).
    """
    B, _ = input_ids.shape
    spans: List[Tuple[int, int]] = []
    for b in range(B):
        pos = (input_ids[b] == image_token_index).nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            spans.append((0, 0))
            continue
        # after expansion, the span starts where the image_token used to sit
        # and ends num_image_patches later
        start = pos[0].item()
        spans.append((start, start + num_image_patches))
    return spans
