"""Amorphous-matrix embedding actors for this project.

Two peer actors, each its own AL-loop stage: ``ROIExtractor``, a
``Selector`` that finds regions of interest in an uncertainty-tagged
trajectory and carves/orients/cleans a box around each, and
``EmbeddingAnnealer``, a ``Sampler`` that anneals every carved box around a
frozen ROI core. Wire ``ROIExtractor``'s output as ``EmbeddingAnnealer``'s
input -- e.g. as two consecutive ``Stage``s in a
``remote_handler.workflows.staged_active_learning.StagedActiveLearningHandler``.

Everything here needs the ``[run]`` extra (``remote_handler`` + ``ase`` +
``numpy``); extraction in ``"surface"``/``"interface"`` topology mode
additionally needs the ``embedding`` extra (``ovito``) on the *remote*
extraction machine. Import the actors as
``lammps_al.embedding.extractor.ROIExtractor`` and
``lammps_al.embedding.annealer.EmbeddingAnnealer``.
"""
