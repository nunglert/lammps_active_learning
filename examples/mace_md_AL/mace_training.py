import contextlib
import json
import logging
import os
import pathlib
import subprocess
import sys
from pathlib import Path
from typing import Generator, Optional

import ase.io
import jinja2
import numpy as np
from ase import Atoms

calcpath = Path(sys.argv[1])
INPUTFILE = calcpath / "input_train.json"
input_args = json.load(open(INPUTFILE, "r"))

rootpath = calcpath.parents[1]

if len(input_args["database_files"]) > 1:
    all_ats = []
    for p in input_args["database_files"]:
        ats = ase.io.read(rootpath / p, index=":")
        if ats is Atoms:
            all_ats.append(ats)
        else:
            all_ats.extend(ats)
    combined_path = calcpath / "combined_db.extxyz"
    ase.io.write(combined_path, all_ats)
    database_path = combined_path

else:
    database_path = rootpath / input_args["database_files"][0]


template_file_name = input_args["template_file"]
model_name = input_args["model_name"]
seed = input_args["seed"]
valid_fraction = input_args["validation_fraction"]
train_runner = Path(input_args["train_runner"])

config_path = calcpath / "mace_config.yaml"


def make_dir(path: pathlib.Path, **kwargs) -> None:
    """Create a directory if it does not exist yet

    Args:
        path: Directory to create
    """
    try:
        path.mkdir(**kwargs)
    except FileExistsError:
        pass


def split_db(
    database: pathlib.Path,
    out_file: pathlib.Path,
    validation_fraction: float = 0.1,
    seed: Optional[int] = None,
    do_return_path: bool = False,
) -> Optional[tuple[pathlib.Path, pathlib.Path]]:
    """Split database into training and validation sets

    Args:
        database: Path to full database
        out_file: Path to files to write the split database to. Training and
            validation splits are marked by '_train' and '_val' respectively
        validation_fraction: Fraction of the database used for validation
        seed: Seed for setting up the NumPy random number generator
        do_return_path: Return a tuple of paths to train and validation sets
    """
    all_atoms = ase.io.read(database.resolve(), index=":")

    rng = np.random.default_rng(seed)
    n_validate = int(len(all_atoms) * validation_fraction)
    n_train = len(all_atoms) - n_validate
    indices = np.arange(len(all_atoms), dtype=int)
    rng.shuffle(indices)
    training_set = [all_atoms[i] for i in indices[:n_train]]
    validation_set = [all_atoms[i] for i in indices[n_train:]]

    out_stem = out_file.stem
    out_suffix = out_file.suffix
    train_file = out_file.parent / (out_stem + "_train" + out_suffix)
    valid_file = out_file.parent / (out_stem + "_val" + out_suffix)
    ase.io.write(train_file.resolve(), training_set)
    ase.io.write(valid_file.resolve(), validation_set)
    return (train_file, valid_file) if do_return_path else None


def parse_jinja_template(
    output_file: pathlib.Path,
    template_dir: pathlib.Path,
    template_file_name: str,
    **kwargs,
) -> None:
    """Parse a Jinja2 template and write to file

    Args:
        output_file: Path to file to store the parsed template to
        template_dir: Path to directory containing Jinja2 templates
        template_file_name: File name of the template to use
        **kwargs: Keyword arguments setting template parameters to fill
    """
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
    )
    template = jinja_env.get_template(template_file_name)
    with open(output_file.resolve(), "w") as file:
        file.write(template.render(**kwargs))


def run_training_from_config_template(
    config_file: pathlib.Path,
    model_name: str,
    seed: int,
    train_file: pathlib.Path,
    valid_file: pathlib.Path,
    train_runner: str,
    storage_dir: pathlib.Path,
    template_dir: pathlib.Path,
    template_file_name: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Prepare a yaml configuration file for MACE training from a template and
    train a model based on that template and the supplied data

    Args:
        config_file: Path to yaml file to store the MACE configuration to
        model_name: Name tag of the model to be trained
        seed: Seed used for model training
        train_file: Path to file to read the training configurations from
        valid_file: Path to file to read the validation configurations from
        train_runner: Path to MACE training runner script.
            (Usually located at $MACE_ROOT_DIR/mace/cli/run_train.py)
        storage_dir: Directory to store the MACE training outputs in.
            Trained models are serialized in storage_dir/models
        template_dir: Path to directory containing templates
        template_file_name: File name of the template to use
        logger: Logger used to write additional output to
    """
    # Create model directory beforehand if it does not exist
    # (Only output directory MACE does not create itself)
    model_dir = storage_dir / "models"
    make_dir(model_dir)

    parse_jinja_template(
        config_file,
        template_dir,
        template_file_name,
        name=model_name,
        seed=seed,
        train_file=str(train_file.resolve()),
        valid_file=str(valid_file.resolve()),
        checkpoints_dir=str((storage_dir / "checkpoints").resolve()),
        log_dir=str((storage_dir / "logs").resolve()),
        model_dir=str(model_dir.resolve()),
        results_dir=str((storage_dir / "results").resolve()),
    )
    try:
        subprocess.run(
            ["python", train_runner, "--config", str(config_file.resolve())],
            check=True,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as run_error:
        if logger is not None:
            logger.error(
                f"MACE training returned with non-zero exit code "
                f"{run_error.returncode}"
            )
            logger.error(run_error.stderr.decode("utf-8"))
        # TODO: Check how to best handle the error here
        raise RuntimeError(run_error)


split = split_db(
    database_path,
    calcpath / "db.extxyz",
    do_return_path=True,
    validation_fraction=valid_fraction,
)
train_path = split[0]
valid_path = split[1]

run_training_from_config_template(
    config_file=config_path,
    model_name=model_name,
    seed=seed,
    train_file=train_path,
    valid_file=valid_path,
    train_runner=str(train_runner),
    storage_dir=calcpath,
    template_dir=rootpath,
    template_file_name=template_file_name,
    logger=logging.getLogger(),
)
