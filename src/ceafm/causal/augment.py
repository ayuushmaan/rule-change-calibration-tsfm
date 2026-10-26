"""Causal event augmentation utilities."""

import torch
import numpy as np


class EventAugmenter:
    def __init__(self, p_mask: float = 0.3, p_swap: float = 0.2):
        self.p_mask = p_mask
        self.p_swap = p_swap

    def augment(self, event_meta: torch.Tensor, text_emb: torch.Tensor):
        """Apply event masking/swapping for causal robustness."""
        B = event_meta.shape[0]
        aug_meta = event_meta.clone()
        aug_emb = text_emb.clone()

        for i in range(B):
            if np.random.rand() < self.p_mask:
                aug_meta[i] = torch.zeros_like(aug_meta[i])
                aug_emb[i] = torch.zeros_like(aug_emb[i])
            elif np.random.rand() < self.p_swap and B > 1:
                j = np.random.randint(0, B)
                aug_meta[i] = event_meta[j]
                aug_emb[i] = text_emb[j]

        return aug_meta, aug_emb
