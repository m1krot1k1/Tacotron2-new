import os
import argparse
from pydub import AudioSegment
import sys # Добавлено для sys.exit
import glob # Добавлено для поиска файлов

def process_and_split_audio_files(audio_dir, processed_dir, chunk_length_sec=20, overlap_sec=0):
    """
    Обрабатывает MP3-файлы, конвертирует их в WAV, нормализует громкость, нарезает длинные на чанки.
    
    Args:
        audio_dir (str): Директория с исходными MP3-файлами
        processed_dir (str): Директория для сохранения обработанных WAV-файлов
        chunk_length_sec (int): Длина чанка в секундах
        overlap_sec (float): Перекрытие между чанками в секундах (0 = без перекрытия)
    
    Returns:
        bool: True при успешной обработке, False в случае ошибок
    """
    os.makedirs(processed_dir, exist_ok=True)
    # Ищем файлы по шаблону *.mp3 в указанной директории (без рекурсии)
    mp3_files_paths = sorted(glob.glob(os.path.join(audio_dir, '*.mp3')))
    if not mp3_files_paths:
        print(f"[ERROR] В директории {audio_dir} не найдено MP3-файлов.")
        return False
    total_chunks = 0
    processed_files_count = 0

    for input_path in mp3_files_paths:
        file = os.path.basename(input_path)
        base_name = os.path.splitext(file)[0]
        print(f"[INFO] Обработка файла: {file}")
        try:
            # Добавлена проверка, что файл существует (хотя glob.glob должен это гарантировать)
            if not os.path.exists(input_path):
                print(f"[WARNING] Файл {file} не найден. Пропуск.")
                continue

            # Загрузка аудио
            audio = AudioSegment.from_file(input_path)
            
            # Нормализация громкости (добавлено)
            target_dBFS = -20.0
            change_in_dBFS = target_dBFS - audio.dBFS
            audio = audio.apply_gain(change_in_dBFS)
            
            # Убедимся в частоте (22050 Гц) и каналах (1, моно) - стандарт для Tacotron 2
            audio = audio.set_frame_rate(22050).set_channels(1)
            duration_ms = len(audio)
            chunk_length_ms = chunk_length_sec * 1000
            overlap_ms = int(overlap_sec * 1000)  # Перекрытие в миллисекундах

            if duration_ms == 0:
                print(f"[WARNING] Файл {file} имеет нулевую длительность. Пропуск.")
                continue

            if duration_ms <= chunk_length_ms:
                # Если файл короче чанка, сохраняем как есть
                output_filename = f"{base_name}.wav"
                output_path = os.path.join(processed_dir, output_filename)
                audio.export(output_path, format="wav")
                print(f"[INFO] Сохранен файл: {output_filename} (длительность: {duration_ms/1000:.2f} сек)")
                total_chunks += 1
                processed_files_count += 1
            else:
                # Для длинных файлов - разделяем на чанки с учетом перекрытия
                chunk_count_for_file = 0
                start_pos = 0
                
                # Пока не достигнем конца файла
                while start_pos < duration_ms:
                    end_pos = min(start_pos + chunk_length_ms, duration_ms)
                    
                    # Если остаток слишком маленький (менее 3 секунд), не создаем чанк
                    if (duration_ms - start_pos) < 3000 and chunk_count_for_file > 0:
                        break
                    
                    chunk = audio[start_pos:end_pos]
                    chunk_filename = f"{base_name}_chunk_{chunk_count_for_file+1:03d}.wav"
                    chunk_path = os.path.join(processed_dir, chunk_filename)
                    chunk.export(chunk_path, format="wav")
                    
                    print(f"[INFO] Создан файл: {chunk_filename} (длительность: {(end_pos-start_pos)/1000:.2f} сек)")
                    total_chunks += 1
                    chunk_count_for_file += 1
                    
                    # Сдвигаем с учетом перекрытия
                    start_pos = end_pos - overlap_ms
                
                if chunk_count_for_file > 0:
                    processed_files_count += 1 # Считаем исходный файл обработанным, если из него создан хотя бы 1 чанк

        except Exception as e:
            print(f"[ERROR] Ошибка при обработке файла {file}: {e}")
            # Продолжаем работу, даже если один файл вызвал ошибку

    if total_chunks > 0:
        print(f"[SUCCESS] Обработка завершена. Обработано исходных файлов: {processed_files_count}. Создано {total_chunks} WAV-файлов.")
        print(f"[INFO] Файлы сохранены в директории: {processed_dir}")
        return True
    else:
        print(f"[WARNING] Обработка завершена, но не создано ни одного WAV-файла. Проверьте логи ошибок.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and split audio files")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with MP3 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed WAV files")
    parser.add_argument("--chunk_length", type=int, default=20, help="Length of audio chunks in seconds")
    parser.add_argument("--overlap", type=float, default=0, help="Overlap between chunks in seconds")
    args = parser.parse_args()

    # Передаем все необходимые аргументы в функцию process_and_split_audio_files
    if not process_and_split_audio_files(args.input_dir, args.output_dir, args.chunk_length, args.overlap):
         sys.exit(1) # Выходим с ошибкой, если не создано ни одного файла
