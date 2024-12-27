from .glhf import GlhfChat

NODE_CLASS_MAPPINGS = {
    "glhf_chat": GlhfChat
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "glhf_chat": "GLHF Chat",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]