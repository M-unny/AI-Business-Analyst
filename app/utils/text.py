"""Column-name helpers shared by the profiler and semantic detector."""
import re


def tokenize_name(name: str) -> list[str]:
    """'CustomerID' -> ['customer','id']; 'unit_price ($)' -> ['unit','price']."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).lower()
    return s.split()


def stem(token: str) -> str:
    """Tiny plural stripper: 'sales' -> 'sale', 'orders' -> 'order'."""
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token
