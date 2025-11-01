from app.core import Core  # Import Core dari app


def main():
    print("Welcome to My Python App!")

    core = Core()
    try:
        # Ambil data 1 tahun untuk saham Apple
        data = core.fetch_stock_data("AAPL", "1y")
        print("Data Asli:")
        print(data.head())

        # Preprocessing
        train_data, test_data, scaler = core.preprocess_stock_data(data)
        print(f"Train data: {len(train_data)} baris, Test data: {len(test_data)} baris")
        print("Contoh train data (skala 0-1):", train_data[:5].flatten())

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
