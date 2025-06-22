import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Dict
import os


class DataConfig:
    """Class konfigurasi untuk data path dan parameter processing"""

    def __init__(self):
        # My DPI Configuration, see https://www.infobyip.com/detectmonitordpi.php
        self.MY_DPI = 120
        self.wound_types = ["hitam", "kuning", "merah"]
        self.processing_methods = ["integer", "interpolation"]

    def get_data_paths(self) -> Dict[str, str]:
        """Mengambil data path untuk semua tipe luka"""
        return {
            wound_type: f"./data/luka_{wound_type}/images"
            for wound_type in self.wound_types
        }

    def get_output_paths(self) -> Dict[str, Dict[str, str]]:
        """Mengambil path output untuk semua tipe luka dan metode"""
        output_paths = {}
        for wound_type in self.wound_types:
            output_paths[wound_type] = {
                method: f"./data/luka_{wound_type}/output/{method}"
                for method in self.processing_methods
            }
        return output_paths


def load_dataframes() -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load semua file CSV menjadi dataFrame"""
    wound_types = ["hitam", "kuning", "merah"]
    dataframes = {}

    for wound_type in wound_types:
        dataframes[wound_type] = {
            "integer": pd.read_csv(
                f"./data/luka_{wound_type}/integer.csv", skipinitialspace=True
            ),
            "interpolation": pd.read_csv(
                f"./data/luka_{wound_type}/interpolation.csv", skipinitialspace=True
            ),
        }

    return dataframes


def extract_row_parameters(
    row: pd.Series,
) -> Tuple[str, str, float, float, float, float, int, float, float, float, int]:
    """Ekstrak parameter dari dataframe"""
    filename = str(row["filename"])
    file_extension = str(row["extension"])
    cr, cc, rad = row["center_r"], row["center_c"], row["radius"]
    sigma, sample = row["sigma"], row["sample"]
    alpha, beta, gamma = row["alpha"], row["beta"], row["gamma"]
    max_iter = row["max_iter"]

    return (
        filename,
        file_extension,
        cr,
        cc,
        rad,
        sigma,
        sample,
        alpha,
        beta,
        gamma,
        max_iter,
    )


def create_and_save_result_image(
    img,
    initial_snake,
    final_snake,
    filename: str,
    file_extension: str,
    output_path: str,
    dpi: int = 120,
) -> None:
    """Membuat dan menyimpan gambar hasil dengan kontur snake (awal & akhir)"""
    # Memastikan output directory ada
    os.makedirs(output_path, exist_ok=True)

    snake_final_fig = plt.figure(
        frameon=False,
        figsize=(img.shape[1] / dpi, img.shape[0] / dpi),
        dpi=dpi,
    )

    snake_final_ax = snake_final_fig.add_axes([0, 0, 1, 1])
    snake_final_ax.imshow(img, cmap=plt.cm.gray)
    snake_final_ax.plot(initial_snake[:, 0], initial_snake[:, 1], "-r", lw=2)
    snake_final_ax.plot(final_snake[:, 0], final_snake[:, 1], "-b", lw=2)
    snake_final_ax.set_xticks([])
    snake_final_ax.set_yticks([])
    snake_final_ax.axis("off")

    output_name = f"{filename}.{file_extension}"
    plt.savefig(f"{output_path}/{output_name}", dpi=dpi)
    plt.close(snake_final_fig)


def process_wound_batch(
    processor,
    dataframe: pd.DataFrame,
    data_path: str,
    output_path: str,
    method: str,
    dpi: int = 120,
) -> None:
    """Batch proses menggunakan metode-metode yg ditentukan"""
    for index, row in dataframe.iterrows():
        (
            filename,
            file_extension,
            cr,
            cc,
            rad,
            sigma,
            sample,
            alpha,
            beta,
            gamma,
            max_iter,
        ) = extract_row_parameters(row)

        image_path = f"{data_path}/{filename}.{file_extension}"

        if method == "integer":
            img, img_gray, initial_snake, final_snake, external_energy = (
                processor.process_with_integer(
                    image_path=image_path,
                    center_row=cr,
                    center_col=cc,
                    radius=rad,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    sigma=sigma,
                    max_iterations=max_iter,
                    sample_points=sample,
                )
            )
        elif method == "interpolation":
            img, img_gray, initial_snake, final_snake, external_energy = (
                processor.process_with_interpolation(
                    image_path=image_path,
                    center_row=cr,
                    center_col=cc,
                    radius=rad,
                    alpha=alpha,
                    beta=beta,
                    gamma=gamma,
                    sigma=sigma,
                    max_iterations=max_iter,
                    sample_points=sample,
                    max_px_move=1.0,
                )
            )
        else:
            raise ValueError(f"Unknown processing method: {method}")

        create_and_save_result_image(
            img, initial_snake, final_snake, filename, file_extension, output_path, dpi
        )


def create_output_directories(output_paths: Dict[str, Dict[str, str]]) -> None:
    """Membuat direktori output yang diperlukan"""
    for wound_type, paths in output_paths.items():
        for method, path in paths.items():
            os.makedirs(path, exist_ok=True)
