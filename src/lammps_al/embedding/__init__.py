"""Amorphous-matrix embedding sampler for this project.

Everything here needs the ``[run]`` extra (``remote_handler`` + ``ase`` +
``numpy``); extraction in ``"surface"``/``"interface"`` topology mode
additionally needs the ``embedding`` extra (``ovito``) on the *remote*
extraction machine. Import the sampler as
``lammps_al.embedding.sampler.AmorphousEmbeddingSampler``.
"""
