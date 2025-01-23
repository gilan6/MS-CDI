import pickle
import json
from typing import Tuple
from dataclasses import asdict, fields

import numpy as np
import h5py

from Param import Param
from utils import cpu2gpu, gpu2cpu


def save_outputs(
    meas_errs,
    rec_errs,
    objs: np.ndarray = np.asarray([0]),
    recs: np.ndarray = np.asarray([0]),
    file_path: str = "outputs.h5",
) -> None:
    save_list = [meas_errs, rec_errs, objs, recs]
    good_types = all(isinstance(var, np.ndarray) for var in save_list)

    if good_types:
        with h5py.File(file_path, "w") as h5file:
            # Save each array to the HDF5 file
            h5file.create_dataset("meas_errs", data=meas_errs)
            h5file.create_dataset("rec_errs", data=rec_errs)
            h5file.create_dataset("objs", data=objs)
            h5file.create_dataset("recs", data=recs)
    else:
        print(
            "Sort out your types! (you should probably use gpu2cpu before saving to HDF5)"
        )
    return

def load_outputs(file_path: str = "outputs.h5"):
    with h5py.File(file_path, "r") as h5file:
        # Load each dataset
        meas_errs = h5file["meas_errs"][:]
        rec_errs = h5file["rec_errs"][:]
        objs = h5file["objs"][:]
        recs = h5file["recs"][:]
    return meas_errs, rec_errs, objs, recs

def save_inputs(
    p: "Param",
    frames: np.ndarray,
    masks: np.ndarray,
    rec0: np.ndarray,
    file_path: str = "inputs.h5",
) -> None:
    p, frames, masks, rec0 = gpu2cpu((p, frames, masks, rec0))
    # Serialize Param dataclass to a dictionary, then to JSON
    param_dict = asdict(p)  # Convert the nested dataclass to a dictionary
    param_json = json.dumps(param_dict)  # Convert dictionary to JSON string

    with h5py.File(file_path, "w") as h5file:
        # Save the serialized Param object
        h5file.create_dataset("param", data=np.bytes_(param_json))  # Use np.bytes_
        # Save NumPy arrays
        h5file.create_dataset("frames", data=frames)
        h5file.create_dataset("masks", data=masks)
        h5file.create_dataset("rec0", data=rec0)
    return


def load_inputs(file_path: str) -> tuple:
    with h5py.File(file_path, "r") as h5file:
        # Load the serialized Param object
        param_json = h5file["param"][()].decode("utf-8")
        param_dict = json.loads(param_json)  # Convert JSON string back to dictionary
        # Reconstruct the Param dataclass
        p = dict_to_dataclass(Param, param_dict)
        # Load NumPy arrays
        frames = h5file["frames"][:]
        masks = h5file["masks"][:]
        rec0 = h5file["rec0"][:]

    return cpu2gpu((p, frames, masks, rec0))


def dict_to_dataclass(cls: type, data: dict):
    """
    Recursively converts a dictionary to a dataclass instance.
    Handles nested dataclasses.
    """
    if not hasattr(cls, "__dataclass_fields__"):
        return data  # Base case: not a dataclass

    field_types = {f.name: f.type for f in fields(cls)}
    return cls(
        **{
            key: dict_to_dataclass(field_types[key], value)
            for key, value in data.items()
        }
    )

if __name__ == "__main__":
    pass
