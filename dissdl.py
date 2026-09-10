import torch
import torch.nn.functional as F
import random
import math
import copy

from dataclasses import dataclass, field

class DissDLNode:
    def __init__(self, node_id, model, neighbors, beta=1.0, k=5):
        self.id = node_id
        self.model = model

        # known peers Pi
        self.known_peers = set(neighbors)

        # wanted_senders
        self.wanted_senders = set(neighbors)

        self.beta = beta
        self.k = k

        # store similarity history: peer_id -> list of scores
        self.similarity_history = {}

        # store received models: peer_id -> model object
        self.received_models = {}

    # ------------------------------------
    # Cosine similarity
    # ------------------------------------
    def cosine_sim(self, model_j):
        with torch.no_grad():
            wi = torch.cat([p.data.view(-1) for p in self.model.parameters()])
            wj = torch.cat([p.data.view(-1) for p in model_j.parameters()])
            return F.cosine_similarity(wi, wj, dim=0).item()

    def softmax_sample(self, score_dict):
        keys = list(score_dict.keys())
        scores = torch.tensor([score_dict[k] for k in keys], dtype=torch.float32)

        probs = torch.softmax(self.beta * scores, dim=0)
        idx = torch.multinomial(probs, 1).item()
        return keys[idx]

    # ------------------------------------
    # Algorithm 2: UpdateWantedSenders()
    # ------------------------------------
    def update_wanted_senders(self):
        S = set(self.wanted_senders)
        A = list(self.known_peers - S)
        R = list(S)

        if len(A) == 0 or len(R) <= 1:
            return

        all_sims = []
        for v in self.similarity_history.values():
            all_sims.extend(v)
        fdefault = sum(all_sims)/len(all_sims) if len(all_sims) > 0 else 0.0

        add_scores = {}
        for j in A:
            sim = self.cosine_sim(self.received_models[j]) if j in self.received_models else fdefault
            add_scores[j] = -sim
        j_plus = self.softmax_sample(add_scores)

        remove_scores = {}
        for j in R:
            sim = self.cosine_sim(self.received_models[j]) if j in self.received_models else fdefault
            remove_scores[j] = sim
        j_minus = self.softmax_sample(remove_scores)

        self.wanted_senders.remove(j_minus)
        self.wanted_senders.add(j_plus)

    # ------------------------------------
    # Aggregate received models
    # ------------------------------------
    def aggregate(self):
        if len(self.received_models) == 0:
            return

        current_device = next(self.model.parameters()).device

        with torch.no_grad():
            my_weight = copy.deepcopy(self.model.state_dict())

            for j, m_j in self.received_models.items():
                sim = self.cosine_sim(m_j)
                if j not in self.similarity_history:
                    self.similarity_history[j] = []
                self.similarity_history[j].append(sim)
                if len(self.similarity_history[j]) > self.k:
                    self.similarity_history[j].pop(0)

            for param in self.model.parameters():
                param.data.zero_()

            num_total = len(self.received_models) + 1

            for m in self.received_models.values():
                for p_self, p_m in zip(self.model.parameters(), m.parameters()):
                    p_self.data += p_m.data / num_total

            for name, param in self.model.named_parameters():
                param.data += my_weight[name].to(current_device) / num_total

        self.received_models = {}

@dataclass
class DissDLState:
    wanted_senders: set
    known_peers: set
    similarity_history: dict = field(default_factory=dict)

    def checkpoint(self):
        return {"wanted_senders": sorted(self.wanted_senders),
                "known_peers": sorted(self.known_peers),
                "similarity_history": self.similarity_history}

    @classmethod
    def restore(cls, value):
        return cls(set(value["wanted_senders"]), set(value["known_peers"]),
                   value.get("similarity_history", {}))
