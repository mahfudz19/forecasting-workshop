from app.core import Core  # Import Core dari app


def main():
    print("Welcome to My Python App!")

    core = Core()
    try:
        # Ambil data 1 tahun untuk saham Apple
        data = core.fetch_stock_data("AAPL", "1y")

        # Preprocessing
        train_data, test_data, scaler = core.preprocess_stock_data(data)
        print(f"Train data: {len(train_data)} baris, Test data: {len(test_data)} baris")

        # Forecasting dengan GPT
        predictions = core.forecast_with_gpt(train_data, test_data, scaler)
        print(f"Prediksi harga (skala asli): {predictions[:5]}")  # 5 prediksi pertama

    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
