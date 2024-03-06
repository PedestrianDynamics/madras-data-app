"""Plot profiles loaded ffrom precalcalculated file."""

import logging
import pickle
import sys
from pathlib import Path
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pedpy as pp
from log_config import setup_logging
from profile_config_data import Config


def init_parameters(config: Config) -> Tuple[List[Any], pp.WalkableArea, List[str]]:
    """Init all necessary parameters."""
    rmax = config.rmax
    vmax = config.vmax
    jmax = config.jmax
    walkable_area = config.walkable_area
    density_profiles = []
    speed_profiles = []
    result_file = config.result_file

    if Path(result_file).exists():
        with open(result_file, "rb") as f:
            density_profiles, speed_profiles = pickle.load(f)
    else:
        sys.exit(f"File {result_file} does not exist!")

    files = config.files
    logging.info("Plotting ... ")
    # Define a list of dictionaries to hold the parameters that change in each iteration
    plot_params = [
        {
            "title": "Density profile",
            "ylabel": r"$\rho\, /\, 1/m^2$",
            "vmax": rmax,
            "data_func": lambda file_: density_profiles[file_],
        },
        {
            "title": "Speed profile",
            "ylabel": r"$V\, /\, m/s$",
            "vmax": vmax,
            "data_func": lambda file_: speed_profiles[file_],
        },
        {
            "title": "Flow profile",
            "ylabel": r"$J\, /\, 1/ms$",
            "vmax": jmax,
            "data_func": lambda file_: np.array(density_profiles[file_]) * np.array(speed_profiles[file_]),
        },
    ]
    return plot_params, walkable_area, files


def main(config: Config) -> None:
    """Run main plotting."""
    setup_logging()
    plot_params, walkable_area, files = init_parameters(config)
    num_figures = 3
    figures, axes = zip(*[plt.subplots(nrows=1, ncols=1) for _ in range(num_figures)])
    # Iterate over each set of axes, figures, and their corresponding parameters
    from pathlib import Path

    # Define the directory path for pdf files
    dir_path = Path("pdfs")
    # Check if the directory exists and create it if it does not
    dir_path.mkdir(parents=True, exist_ok=True)
    for ax, fig, params in zip(axes, figures, plot_params):
        fig.suptitle(params["title"])
        logging.info(f"{params['title']}")
        for file_ in files:
            profile_data = params["data_func"](file_)
            pp.plot_profiles(
                walkable_area=walkable_area,
                profiles=profile_data,
                axes=ax,
                vmin=0,
                vmax=params["vmax"],
            )
            # Setting colorbar properties (assuming last axes is always the colorbar)
            colorbar_ax = fig.axes[-1]
            colorbar_ax.set_ylabel(params["ylabel"], size=18)
            colorbar_ax.tick_params(labelsize=18)

            ax.tick_params(axis="x", length=0)
            ax.tick_params(axis="y", length=0)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            logging.info("Saving fig..")
            figname = dir_path / f"profiles_{params['title']}_{Path(file_).stem}.pdf"
            fig.savefig(figname, bbox_inches="tight", pad_inches=0.1)
            logging.info(figname)
