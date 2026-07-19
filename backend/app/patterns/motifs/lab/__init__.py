"""Motif LAB — candidate centerpiece silhouettes for the remaining box faces.

Each module exposes a ``<name>_silhouette(extent_um, n_grid=256) -> np.ndarray``
following the same conventions as ``motifs/colibri.py`` and ``motifs/globe.py``:
a binary bool grid (True = gold), authored from geometric primitives / cubic
Béziers in a normalized 0..1 art box (Pillow y-down), scale-free (the caller
picks ``cell_um`` to hit the physical extent).

These are prototype candidates for the orchestrator to present to the user; not
yet wired into the pattern registry.
"""
