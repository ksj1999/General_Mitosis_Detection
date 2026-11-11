"""fmbench: a reproduction of the Virchow2 tile-level benchmark suite.

The suite embeds every benchmark tile with a frozen encoder, caches the embeddings,
trains the Virchow2 linear probe on them, and records the test metrics so that any
number of models can be compared with each other and with the published Virchow2 tables.
"""

__version__ = "0.1.0"
PROTOCOL_VERSION = "virchow2-linear-v1"
