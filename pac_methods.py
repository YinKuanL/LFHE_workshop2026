"""PAC workshop method registry independent of the training runtime."""

PAC_METHODS = frozenset({"random_fof", "lfhe", "lfhe_expand", "lfhe_pac"})
FIXED_EDGE_SWAP_METHODS = frozenset({"random_fof", "lfhe"})
PAC_PROTOCOL_METHOD = {
    "random_fof": "random_fof_swap",
    "lfhe": "lfhe_representation_swap",
    "lfhe_expand": "lfhe_pac_v2",
    "lfhe_pac": "lfhe_pac_strict",
}
