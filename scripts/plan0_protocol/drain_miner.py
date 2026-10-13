"""
Drain3 Mining & Unseen Template Embedding Similarity Mapping for Plan 0 Protocol.
Supports:
1. Train-fit Drain3 (leakage-free, fits strictly on training split).
2. Embedding-similarity mapper for unseen test templates.
3. Full-fit comparison (reproducing LogLLM unconstrained fitting).
4. Drift diagnostics: tracking share of test windows with unseen templates.
"""

import os
import math
import json
import logging
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


class TrainFitDrainMiner:
    """
    Leakage-controlled Drain3 miner.
    Fitted strictly on training period messages. Unseen templates in test are mapped
    to the closest training template via embedding cosine similarity.
    """

    def __init__(self, depth: int = 4, sim_th: float = 0.5, max_children: int = 100):
        self.depth = depth
        self.sim_th = sim_th
        self.max_children = max_children
        
        config = TemplateMinerConfig()
        config.drain_depth = depth
        config.drain_sim_th = sim_th
        config.drain_max_children = max_children
        # Disable noisy drain logger
        logging.getLogger('drain3').setLevel(logging.WARNING)

        self.miner = TemplateMiner(config=config)
        self.is_frozen = False
        self.train_cluster_ids: Set[int] = set()
        self.train_templates: Dict[int, str] = {}
        
        # Unseen mapping cache
        self.unseen_mapping_cache: Dict[str, Tuple[int, float]] = {}  # msg/template -> (nearest_cluster_id, sim)
        self.train_embeddings: Optional[torch.Tensor] = None
        self.train_cluster_list: List[int] = []

    def fit_message(self, message: str) -> int:
        """Add training message to Drain3 and record cluster."""
        if self.is_frozen:
            raise RuntimeError("Cannot fit message: miner is frozen to prevent test leakage.")
        res = self.miner.add_log_message(message)
        cluster_id = res['cluster_id']
        template = res['template_mined']
        self.train_cluster_ids.add(cluster_id)
        self.train_templates[cluster_id] = template
        return cluster_id

    def fit_training_corpus(self, messages: List[str]):
        """Fits Drain3 on all messages in the training partition."""
        for msg in messages:
            self.fit_message(msg)
        self.freeze_and_index()

    def freeze_and_index(self):
        """Freezes miner against new cluster creations and builds embedding search index."""
        self.is_frozen = True
        self.train_cluster_list = sorted(list(self.train_cluster_ids))
        self._build_template_embeddings()

    def _build_template_embeddings(self):
        """Constructs normalized n-gram / token frequency representations for cosine search."""
        templates = [self.train_templates.get(cid, "") for cid in self.train_cluster_list]
        self.train_embeddings = self._embed_texts(templates)

    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Fast token hashing bag-of-words / character 3-gram embedding in PyTorch.
        Dimension 512, normalized for instant cosine similarity search without GPU VRAM exhaustion.
        """
        dim = 512
        embeddings = torch.zeros((len(texts), dim), dtype=torch.float32)
        for i, text in enumerate(texts):
            tokens = text.lower().split()
            if not tokens:
                continue
            for tok in tokens:
                # token hash into dimension
                h = abs(hash(tok)) % dim
                embeddings[i, h] += 1.0
                # character n-grams
                for k in range(len(tok) - 2):
                    sub = tok[k:k+3]
                    embeddings[i, abs(hash(sub)) % dim] += 0.5
        norm = torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp(min=1e-8)
        return embeddings / norm

    def map_message(self, message: str) -> Tuple[int, bool, float]:
        """
        Maps a test message to a template ID.
        Returns:
            cluster_id: mapped train template ID
            is_unseen: True if message did not match any training template exactly
            similarity: cosine similarity to mapped template (1.0 if exact Drain match)
        """
        if not self.is_frozen:
            self.freeze_and_index()

        # 1. Check exact Drain3 match against training clusters
        cluster = self.miner.match(message)
        if cluster is not None and cluster.cluster_id in self.train_cluster_ids:
            return cluster.cluster_id, False, 1.0

        # 2. Unseen template: look up in cache
        if message in self.unseen_mapping_cache:
            cid, sim = self.unseen_mapping_cache[message]
            return cid, True, sim

        # 3. Find closest train template by embedding similarity
        if self.train_embeddings is None or len(self.train_cluster_list) == 0:
            return 0, True, 0.0

        msg_emb = self._embed_texts([message])
        sims = torch.matmul(msg_emb, self.train_embeddings.T).squeeze(0)  # (N_train,)
        best_idx = int(torch.argmax(sims).item())
        best_sim = float(sims[best_idx].item())
        best_cluster_id = self.train_cluster_list[best_idx]

        self.unseen_mapping_cache[message] = (best_cluster_id, best_sim)
        return best_cluster_id, True, best_sim

    def get_template(self, cluster_id: int) -> str:
        return self.train_templates.get(cluster_id, "<UNKNOWN_TEMPLATE>")


class DrainFitComparison:
    """
    Computes comparative statistics between Train-Fit and Full-Fit Drain mining,
    and calculates drift diagnostics on test windows.
    """

    @staticmethod
    def compare(
        train_miner: TrainFitDrainMiner,
        full_miner: TemplateMiner,
        test_messages: List[str],
        test_window_ranges: List[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """
        Computes Drain-fit comparison and unseen template drift diagnostic on test windows.
        """
        n_train_templates = len(train_miner.train_cluster_ids)
        n_full_templates = len(full_miner.drain.id_to_cluster)

        unseen_test_msgs = 0
        window_has_unseen = [False] * len(test_window_ranges)
        mapped_similarities = []

        # Map each test message
        test_msg_is_unseen = []
        for msg in test_messages:
            _, is_unseen, sim = train_miner.map_message(msg)
            test_msg_is_unseen.append(is_unseen)
            if is_unseen:
                unseen_test_msgs += 1
                mapped_similarities.append(sim)

        # Check per-window unseen rate
        for w_idx, (start_idx, end_idx) in enumerate(test_window_ranges):
            if any(test_msg_is_unseen[start_idx:end_idx]):
                window_has_unseen[w_idx] = True

        share_unseen_windows = (
            sum(window_has_unseen) / max(1, len(test_window_ranges))
        )
        avg_mapped_similarity = (
            float(np.mean(mapped_similarities)) if mapped_similarities else 1.0
        )

        return {
            "train_fit_templates_count": n_train_templates,
            "full_fit_templates_count": n_full_templates,
            "test_messages_count": len(test_messages),
            "unseen_test_messages_count": unseen_test_msgs,
            "unseen_message_rate": unseen_test_msgs / max(1, len(test_messages)),
            "share_test_windows_with_unseen_templates": share_unseen_windows,
            "avg_unseen_mapped_cosine_similarity": avg_mapped_similarity,
        }
