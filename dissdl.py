"""State container and canonical DissDL peer-selection helpers."""
from dataclasses import dataclass, field

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
