from src.processing import SnakeProcessor
from src.utils import (
    DataConfig,
    load_dataframes,
    process_wound_batch,
    create_output_directories,
    create_validation_directories,
    validate_results,
)


def main():
    """Entry utama untuk proses deteksi semua gambar luka yg ditentukan"""
    config = DataConfig()

    print("Loading dataframes...")
    dataframes = load_dataframes()

    data_paths = config.get_data_paths()
    output_paths = config.get_output_paths()
    validation_paths = config.get_validation_paths()

    print("Creating output directories...")
    create_output_directories(output_paths)

    print("Creating validation directories...")
    create_validation_directories(validation_paths)

    print("Initializing snake processor...")
    processor = SnakeProcessor()

    # Memproses semua kombinasi jenis deteksi
    for wound_type in config.wound_types:
        print(f"\nProcessing Luka {wound_type}")

        for method in config.processing_methods:
            print(f"  Using {method} method...")

            df = dataframes[wound_type][method]
            data_path = data_paths[wound_type]
            output_path = output_paths[wound_type][method]
            validation_path = validation_paths[wound_type][method]

            process_wound_batch(
                processor=processor,
                dataframe=df,
                data_path=data_path,
                output_path=output_path,
                validation_path=validation_path,
                method=method,
                dpi=config.MY_DPI,
            )

            print(f"    Processed {len(df)} images for {wound_type}-{method}")

    print("\nAll processing completed!")

    # Validasi hasil deteksi
    for wound_type in config.wound_types:
        print(f"\nValidating Luka {wound_type}...")

        for method in config.processing_methods:
            print(f"  Using {method} method...")

            df = dataframes[wound_type][method]
            output_path = output_paths[wound_type][method]
            validation_path = validation_paths[wound_type][method]

            validate_results(
                dataframe=df,
                output_path=output_path,
                validation_path=validation_path,
                dpi=config.MY_DPI,
            )

            print(f"    Processed {len(df)} images for {wound_type}-{method}")

    print("\nAll processing completed!")


if __name__ == "__main__":
    main()
