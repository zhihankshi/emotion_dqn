"""Episode path classification for maze diagnostics."""
from typing import Any, Dict


def classify_episode_path(env, info: Dict[str, Any], success: bool) -> str:
    """
    Classify how the agent completed (or failed) an episode.

    Returns one of:
        shield_route, trap_rush, key_route, direct, timeout, other
    """
    if not success:
        return "timeout"

    shield_step = info.get("shield_pickup_step", -1)
    trap_step = info.get("trap_hit_step", -1)
    key_step = info.get("key_pickup_step", -1)
    door_open = info.get("door_open", False)

    if getattr(env, "has_shield_mechanic", False):
        if shield_step > 0 and trap_step > 0 and shield_step < trap_step:
            return "shield_route"
        if trap_step > 0 and shield_step <= 0:
            return "trap_rush"
        if shield_step > 0:
            return "other"
        return "direct"

    if getattr(env, "has_key_mechanic", False) and getattr(env, "key_required", False):
        if key_step > 0 and door_open:
            return "key_route"
        return "other"

    return "direct"
