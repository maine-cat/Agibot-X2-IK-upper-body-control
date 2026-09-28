"""X2 robot module: MDI and MoveJ are the supported public interfaces."""
__all__ = ["Robot", "HOME"]
__version__ = "2.1.1"


def __getattr__(name):
    if name in __all__:
        from .x2_movej import Robot, HOME
        return {"Robot": Robot, "HOME": HOME}[name]
    raise AttributeError(name)
