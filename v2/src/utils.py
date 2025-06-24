import matplotlib.pyplot as plt
import pandas as pd
from typing import Tuple, Dict
import os
import numpy as np
from skimage.io import imread
from skimage.color import rgb2gray


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

    def get_validation_paths(self) -> Dict[str, Dict[str, str]]:
        """Mengambil path validation untuk semua tipe luka dan metode"""
        val_path = {}
        for wound_type in self.wound_types:
            val_path[wound_type] = {
                method: f"./data/luka_{wound_type}/validation/{method}"
                for method in self.processing_methods
            }
        return val_path


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


def create_and_save_final_snake_region_image(
    img,
    final_snake,
    filename: str,
    file_extension: str,
    output_path: str,
    dpi: int = 120,
) -> None:
    """Membuat dan menyimpan gambar region kontur akhir snake"""
    # Memastikan output directory ada
    os.makedirs(output_path, exist_ok=True)

    snake_final_region_fig = plt.figure(
        frameon=False, figsize=(img.shape[1] / dpi, img.shape[0] / dpi), dpi=dpi
    )
    snake_final_region_ax = snake_final_region_fig.add_axes([0, 0, 1, 1])
    snake_final_region_ax.imshow(np.ones(img.shape))
    snake_final_region_ax.fill(final_snake[:, 0], final_snake[:, 1], color="black")
    snake_final_region_ax.set_xticks([]), snake_final_region_ax.set_yticks([])
    snake_final_region_ax.axis("off")

    output_name = f"{filename}_region_result.{file_extension}"
    plt.savefig(f"{output_path}/{output_name}", dpi=dpi)
    plt.close(snake_final_region_fig)


def process_wound_batch(
    processor,
    dataframe: pd.DataFrame,
    data_path: str,
    output_path: str,
    validation_path: str,
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
            img=img,
            initial_snake=initial_snake,
            final_snake=final_snake,
            filename=filename,
            file_extension=file_extension,
            output_path=output_path,
            dpi=dpi,
        )

        create_and_save_final_snake_region_image(
            img=img,
            final_snake=final_snake,
            filename=filename,
            file_extension=file_extension,
            output_path=validation_path,
            dpi=dpi,
        )


def validate_results(
    dataframe: pd.DataFrame,
    output_path: str,
    validation_path: str,
    dpi: int = 120,
) -> None:
    """Batch validasi hasil deteksi"""
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

        image_path = f"{output_path}/{filename}.{file_extension}"
        result_path = f"{validation_path}/{filename}_region_result.{file_extension}"
        groundtruth_path = (
            f"{validation_path}/{filename}_region_groundtruth.{file_extension}"
        )

        img_result = imread(image_path)
        img_groundtruth = rgb2gray(imread(groundtruth_path))
        img_result_region = rgb2gray(imread(result_path))

        sum_gt = 0
        sum_int = 0

        for x in img_groundtruth:
            for y in x:
                if y != 1.0:
                    sum_gt += 1

        for p in img_result_region:
            for q in p:
                if q != 1.0:
                    sum_int += 1

        accuracy = 100 - abs(sum_gt - sum_int) / sum_gt * 100

        # Adaptive padding berdasarkan ukuran gambar
        img_height, img_width = img_result.shape[:2]

        # Hitung padding adaptif (5-15% dari tinggi gambar)
        bottom_padding = max(30, min(int(img_height * 0.1), 100))

        # Padding samping untuk gambar yang sangat lebar atau sempit
        side_padding = 0
        aspect_ratio = img_width / img_height
        if aspect_ratio > 3:  # gambar sangat lebar
            side_padding = int(img_width * 0.05)
        elif aspect_ratio < 0.5:  # gambar sangat tinggi
            side_padding = int(img_width * 0.1)

        # Padding atas untuk memberikan ruang ekstra jika diperlukan
        top_padding = max(10, int(img_height * 0.02))

        pad_color = 0  # hitam

        # Apply padding: top, bottom, left, right
        if len(img_result.shape) == 3:  # Color image
            img_result_padded = np.pad(
                img_result,
                ((top_padding, bottom_padding), (side_padding, side_padding), (0, 0)),
                mode="constant",
                constant_values=pad_color,
            )
        else:
            img_result_padded = np.pad(
                img_result,
                ((top_padding, bottom_padding), (side_padding, side_padding)),
                mode="constant",
                constant_values=pad_color,
            )

        # Hitung ukuran font adaptif berdasarkan ukuran gambar total
        total_pixels = img_result_padded.shape[0] * img_result_padded.shape[1]

        if total_pixels < 100000:  # gambar kecil
            fontsize = max(8, int(min(img_width, img_height) * 0.03))
        elif total_pixels < 500000:  # gambar medium
            fontsize = max(10, int(min(img_width, img_height) * 0.025))
        else:  # gambar besar
            fontsize = max(12, int(min(img_width, img_height) * 0.02))

        # Batasi ukuran font
        fontsize = min(fontsize, 24)

        accuracy_fig = plt.figure(
            frameon=False,
            figsize=(
                img_result_padded.shape[1] / dpi,
                img_result_padded.shape[0] / dpi,
            ),
            dpi=dpi,
        )

        accuracy_ax = accuracy_fig.add_axes([0, 0, 1, 1])
        accuracy_ax.imshow(img_result_padded, cmap=plt.cm.gray)

        # Posisi teks di area bottom padding
        text_x = img_result_padded.shape[1] // 2
        text_y = img_result_padded.shape[0] - bottom_padding // 2

        accuracy_ax.text(
            text_x,
            text_y,
            f"Accuracy: {accuracy:.2f}%",
            color="white",
            fontsize=fontsize,
            ha="center",
            va="center",
            weight="bold",  # buat teks lebih tebal
            bbox=dict(
                facecolor="black",
                alpha=0.8,
                pad=fontsize // 3,  # padding bbox adaptif
                boxstyle="round,pad=0.3",
            ),
        )

        accuracy_ax.set_xticks([]), accuracy_ax.set_yticks([])
        accuracy_ax.axis("off")
        output_name = f"{filename}_val.{file_extension}"
        plt.savefig(
            f"{validation_path}/{output_name}",
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(accuracy_fig)


def create_output_directories(output_paths: Dict[str, Dict[str, str]]) -> None:
    """Membuat direktori output yang diperlukan"""
    for wound_type, paths in output_paths.items():
        for method, path in paths.items():
            os.makedirs(path, exist_ok=True)


def create_validation_directories(output_paths: Dict[str, Dict[str, str]]) -> None:
    """Membuat direktori output yang diperlukan"""
    for wound_type, paths in output_paths.items():
        for method, path in paths.items():
            os.makedirs(path, exist_ok=True)
