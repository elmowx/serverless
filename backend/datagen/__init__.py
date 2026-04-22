__all__ = ["PoissonGenerator", "FlowGenerator", "parse_user_csv"]


def __getattr__(name: str):
    if name == "PoissonGenerator":
        from .poisson import PoissonGenerator
        return PoissonGenerator
    if name == "FlowGenerator":
        from .flow import FlowGenerator
        return FlowGenerator
    if name == "parse_user_csv":
        from .upload import parse_user_csv
        return parse_user_csv
    raise AttributeError(name)
