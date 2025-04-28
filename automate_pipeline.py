import os
import subprocess

def run_prepare_metadata():
    """Запускает prepare_metadata.py для подготовки данных."""
    print("[INFO] Запуск подготовки метаданных...")
    subprocess.run([
        "python", "prepare_metadata.py",
        "--trans_dir", "transcriptions",
        "--audio_dir", "processed_audio",
        "--dataset_dir", "dataset",
        "--metadata_file_all", "dataset/metadata.csv",
        "--train_file", "dataset/train.csv",
        "--validation_file", "dataset/validation.csv",
        "--validation_split", "0.1"
    ], check=True)

def check_and_continue_training(model_type, train_script, checkpoint_dir, additional_args=None):
    """Проверяет наличие чекпоинтов и продолжает обучение или начинает с нуля."""
    print(f"[INFO] Проверка чекпоинтов для {model_type}...")
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint") or f.startswith("g_")]

    if checkpoint_files:
        print(f"[INFO] Найдены чекпоинты для {model_type}. Продолжаем обучение...")
        latest_checkpoint = max(checkpoint_files, key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
        command = ["python", train_script, "-c", checkpoint_path]
    else:
        print(f"[INFO] Чекпоинты для {model_type} не найдены. Начинаем обучение с нуля...")
        command = ["python", train_script]

    if additional_args:
        command.extend(additional_args)

    subprocess.run(command, check=True)

def run_hifigan_training():
    """Запускает обучение HiFi-GAN с проверкой чекпоинтов."""
    print("[INFO] Запуск обучения HiFi-GAN...")
    hifigan_checkpoint_dir = "hifigan/checkpoints"
    os.makedirs(hifigan_checkpoint_dir, exist_ok=True)
    check_and_continue_training(
        model_type="HiFi-GAN",
        train_script="hifigan/train.py",
        checkpoint_dir=hifigan_checkpoint_dir,
        additional_args=["--input_wavs_dir", "dataset/wavs", "--input_training_file", "LJSpeech-1.1/training.txt", "--input_validation_file", "LJSpeech-1.1/validation.txt"]
    )

def run_tacotron_training():
    """Запускает обучение Tacotron2 с проверкой чекпоинтов."""
    print("[INFO] Запуск обучения Tacotron2...")
    tacotron_checkpoint_dir = "checkpoints"
    os.makedirs(tacotron_checkpoint_dir, exist_ok=True)
    check_and_continue_training(
        model_type="Tacotron2",
        train_script="train.py",
        checkpoint_dir=tacotron_checkpoint_dir,
        additional_args=["-o", "outdir", "-l", "logdir", "--n_gpus", "1"]
    )

def manage_checkpoints():
    """Управляет чекпоинтами с помощью checkpoint_manager.py."""
    print("[INFO] Управление чекпоинтами...")
    subprocess.run([
        "python", "checkpoint_manager.py",
        "--model", "tacotron",
        "--action", "status"
    ], check=True)

def main():
    """Основная функция для автоматизации всех шагов."""
    try:
        run_prepare_metadata()
        run_tacotron_training()
        run_hifigan_training()
        manage_checkpoints()
        print("[SUCCESS] Все шаги успешно выполнены.")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Произошла ошибка: {e}")

if __name__ == "__main__":
    main()