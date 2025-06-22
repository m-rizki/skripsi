from src.processing import SnakeProcessor
from src.utils import (
    DataConfig,
    load_dataframes,
    process_wound_batch,
    create_output_directories,
)


def main():
    """Entry utama untuk proses deteksi semua gambar luka yg ditentukan"""
    config = DataConfig()

    print("Loading dataframes...")
    dataframes = load_dataframes()

    data_paths = config.get_data_paths()
    output_paths = config.get_output_paths()

    print("Creating output directories...")
    create_output_directories(output_paths)

    print("Initializing snake processor...")
    processor = SnakeProcessor()

    # Memproses semua kombinasi jenis deteksi
    for wound_type in config.wound_types:
        print(f"\nProcessing {wound_type} wounds...")

        for method in config.processing_methods:
            print(f"  Using {method} method...")

            df = dataframes[wound_type][method]
            data_path = data_paths[wound_type]
            output_path = output_paths[wound_type][method]

            process_wound_batch(
                processor=processor,
                dataframe=df,
                data_path=data_path,
                output_path=output_path,
                method=method,
                dpi=config.MY_DPI,
            )

            print(f"    Processed {len(df)} images for {wound_type}-{method}")

    print("\nAll processing completed!")


if __name__ == "__main__":
    main()
