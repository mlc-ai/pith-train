"""
Process-global runtime state, one module per concern.

These values are set once at initialization, constant thereafter, and accessed frequently across
various parts of the framework. Keeping them here as module state lets code read them in-line
instead of threading them through every constructor and call. See each module for its exports.
"""
