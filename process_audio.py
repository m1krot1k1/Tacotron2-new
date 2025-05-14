import os
import argparse
from pydub import AudioSegment
import sys
import glob
import subprocess # Импортируем модуль для запуска внешних команд
import math # Импортируем math для математических операций

# Определяем порог размера файла для предварительного разбиения (например, 0.1 GB)
# Понижено для обхода возможного некорректного определения размера больших файлов pydub
LARGE_FILE_THRESHOLD_GB = 0.1
LARGE_FILE_THRESHOLD_BYTES = LARGE_FILE_THRESHOLD_GB * 1024 * 1024 * 1024

# Определяем примерную длительность временных чанков при разбиении больших файлов (в секундах)
# Использование фиксированной длительности более надежно, чем попытка рассчитать по битрейту.
TEMP_CHUNK_DURATION_SECONDS = 1800 # 30 минут

def split_large_mp3_with_ffmpeg(input_path, temp_dir, chunk_duration_sec):
    """
    Разбивает большой MP3-файл на более мелкие MP3-чанки с использованием ffmpeg.

    Args:
        input_path (str): Путь к большому входному MP3-файлу.
        temp_dir (str): Директория для сохранения временных более мелких MP3-чанков.
        chunk_duration_sec (int): Желаемая длительность каждого чанка в секундах.

    Returns:
        list: Список путей к созданным временным MP3-чанкам.
        None: Если произошла ошибка при разбиении.
    """
    os.makedirs(temp_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    temp_chunks = []

    # Получаем длительность входного файла с помощью ffprobe
    try:
        duration_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            input_path
        ]
        # Используем shell=True только если уверены в источнике команды и входных данных
        # Для безопасности лучше избегать shell=True, но для простых команд может быть удобно
        # Здесь мы строим команду из списка, поэтому shell=False безопаснее
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
        duration_sec = float(duration_result.stdout.strip())
        print(f"[INFO] Длительность файла {os.path.basename(input_path)}: {duration_sec:.2f} секунд")
    except FileNotFoundError:
        print(f"[ERROR] Утилита ffprobe не найдена. Убедитесь, что ffmpeg установлен и доступен в PATH.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Ошибка при получении длительности файла {os.path.basename(input_path)} с помощью ffprobe:")
        print(f"        Команда: {' '.join(e.cmd)}")
        print(f"        Код возврата: {e.returncode}")
        print(f"        Stdout: {e.stdout}")
        print(f"        Stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"[ERROR] Не удалось получить длительность файла {os.path.basename(input_path)}: {e}")
        return None

    # Рассчитываем количество чанков
    num_chunks = math.ceil(duration_sec / chunk_duration_sec)
    print(f"[INFO] Разбиение на {num_chunks} чанков длительностью примерно {chunk_duration_sec} секунд.")

    for i in range(num_chunks):
        start_time = i * chunk_duration_sec
        temp_chunk_path = os.path.join(temp_dir, f"{base_name}_temp_chunk_{i+1:03d}.mp3")
        temp_chunks.append(temp_chunk_path)

        # Команда ffmpeg для разбиения
        ffmpeg_cmd = [
            'ffmpeg',
            '-i', input_path,
            '-ss', str(start_time), # Время начала
            '-t', str(chunk_duration_sec), # Длительность чанка
            '-c', 'copy', # Копировать потоки без перекодирования (быстрее, но может быть не совсем точная длительность)
            temp_chunk_path
        ]

        try:
            print(f"[INFO] Запуск команды ffmpeg: {' '.join(ffmpeg_cmd)}")
            # Используем shell=False для безопасности
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
            print(f"[SUCCESS] Создан временный чанк: {os.path.basename(temp_chunk_path)}")
        except FileNotFoundError:
            print(f"[ERROR] Утилита ffmpeg не найдена. Убедитесь, что ffmpeg установлен и доступен в PATH.")
            # Очистка временных файлов, если произошла ошибка
            for f in temp_chunks:
                if os.path.exists(f):
                    os.remove(f)
            return None
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Ошибка при разбиении файла {os.path.basename(input_path)} с помощью ffmpeg:")
            print(f"        Команда: {' '.join(e.cmd)}")
            print(f"        Код возврата: {e.returncode}")
            print(f"        Stdout: {e.stdout}")
            print(f"        Stderr: {e.stderr}")
            # Очистка временных файлов, если произошла ошибка
            for f in temp_chunks:
                if os.path.exists(f):
                    os.remove(f)
            return None
        except Exception as e:
            print(f"[ERROR] Произошла непредвиденная ошибка при разбиении ffmpeg: {e}")
            # Очистка временных файлов, если произошла ошибка
            for f in temp_chunks:
                if os.path.exists(f):
                    os.remove(f)
            return None

    return temp_chunks

def process_and_split_audio_files(audio_dir, processed_dir, chunk_length_sec=20, overlap_sec=0):
    """
    Обрабатывает MP3-файлы, конвертирует их в WAV, нормализует громкость, нарезает длинные на чанки.
    Обрабатывает большие файлы, сначала разбивая их на временные чанки с помощью ffmpeg.

    Args:
        audio_dir (str): Директория с исходными MP3-файлами
        processed_dir (str): Директория для сохранения обработанных WAV-файлов
        chunk_length_sec (int): Длина финального чанка в секундах
        overlap_sec (float): Перекрытие между финальными чанками в секундах (0 = без перекрытия)

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
    processed_original_files_count = 0 # Счетчик оригинальных файлов, из которых успешно созданы чанки
    temp_dir = os.path.join(processed_dir, "temp_ffmpeg_chunks") # Директория для временных чанков ffmpeg

    for input_path in mp3_files_paths:
        file = os.path.basename(input_path)
        base_name = os.path.splitext(file)[0]
        print(f"[INFO] Обработка файла: {file}")

        try:
            if not os.path.exists(input_path):
                print(f"[WARNING] Файл {file} не найден. Пропуск.")
                continue

            file_size = os.path.getsize(input_path)
            print(f"[DEBUG] Размер файла {file}: {file_size} байт ({file_size / (1024*1024*1024):.2f} GB). Порог: {LARGE_FILE_THRESHOLD_GB} GB.") # Отладочное сообщение
            files_to_process_with_pydub = [] # Список файлов (оригинальный или временные чанки) для обработки pydub

            # Изменено условие для принудительного использования ffmpeg для большинства файлов
            if file_size > LARGE_FILE_THRESHOLD_BYTES:
                print(f"[INFO] Файл {file} ({file_size / (1024*1024*1024):.2f} GB) превышает порог {LARGE_FILE_THRESHOLD_GB} GB. Разбиение с помощью ffmpeg.")
                # Разбиваем на временные чанки TEMP_CHUNK_DURATION_SECONDS
                temp_chunks = split_large_mp3_with_ffmpeg(input_path, temp_dir, TEMP_CHUNK_DURATION_SECONDS)
                print(f"[DEBUG] Результат split_large_mp3_with_ffmpeg для {file}: {temp_chunks}") # Отладочное сообщение
                if temp_chunks is None:
                    print(f"[ERROR] Не удалось разбить большой файл {file} на временные чанки. Пропуск оригинального файла.")
                    continue # Пропускаем этот оригинальный файл, если разбиение не удалось
                files_to_process_with_pydub.extend(temp_chunks)
            else:
                # Если файл меньше порога (или если порог очень низкий, как сейчас),
                # мы все равно можем попробовать разбить его через ffmpeg,
                # или передать напрямую в pydub, если уверены, что pydub справится.
                # В данном случае, чтобы обойти ошибку pydub, мы принудительно
                # используем ffmpeg, если размер больше 0.
                if file_size > 0: # Обрабатываем ненулевые файлы через ffmpeg
                     print(f"[INFO] Файл {file} ({file_size / (1024*1024*1024):.2f} GB) меньше порога {LARGE_FILE_THRESHOLD_GB} GB, но будет обработан через ffmpeg для обхода ошибки pydub.")
                     temp_chunks = split_large_mp3_with_ffmpeg(input_path, temp_dir, TEMP_CHUNK_DURATION_SECONDS) # Используем ту же длительность чанка
                     print(f"[DEBUG] Результат split_large_mp3_with_ffmpeg для {file}: {temp_chunks}") # Отладочное сообщение
                     if temp_chunks is None:
                         print(f"[ERROR] Не удалось разбить файл {file} на временные чанки с помощью ffmpeg. Пропуск оригинального файла.")
                         continue # Пропускаем этот оригинальный файл, если разбиение не удалось
                     files_to_process_with_pydub.extend(temp_chunks)
                else:
                    print(f"[WARNING] Файл {file} имеет нулевой размер. Пропуск.")
                    continue


            original_file_produced_chunks = False # Флаг, указывающий, был ли этот оригинальный файл успешно обработан (созданы чанки)

            # Обрабатываем каждый файл/временный чанк с помощью pydub
            for current_input_path in files_to_process_with_pydub:
                current_file_name = os.path.basename(current_input_path)
                # current_base_name = os.path.splitext(current_file_name)[0] # Не используется для именования финальных чанков

                try:
                    # Загрузка аудио с помощью pydub (теперь для меньших файлов или чанков)
                    # Эта часть выполняется после разбиения ffmpeg, поэтому должна работать
                    audio = AudioSegment.from_file(current_input_path)

                    # Нормализация громкости
                    target_dBFS = -20.0
                    change_in_dBFS = target_dBFS - audio.dBFS
                    audio = audio.apply_gain(change_in_dBFS)

                    # Убедимся в частоте (22050 Гц) и каналах (1, моно)
                    audio = audio.set_frame_rate(22050).set_channels(1)
                    duration_ms = len(audio)
                    chunk_length_ms = chunk_length_sec * 1000
                    overlap_ms = int(overlap_sec * 1000)

                    if duration_ms == 0:
                        print(f"[WARNING] Файл {current_file_name} имеет нулевую длительность после загрузки. Пропуск.")
                        continue

                    if duration_ms <= chunk_length_ms:
                        # Если файл/чанк короче финального чанка, сохраняем как есть
                        # Используем базовое имя оригинального файла для именования
                        output_filename = f"{base_name}_chunk_{total_chunks + 1:03d}.wav"
                        output_path = os.path.join(processed_dir, output_filename)
                        audio.export(output_path, format="wav")
                        print(f"[INFO] Сохранен файл: {output_filename} (длительность: {duration_ms/1000:.2f} сек)")
                        total_chunks += 1
                        original_file_produced_chunks = True
                    else:
                        # Для длинных файлов/чанков - разделяем на финальные чанки с учетом перекрытия
                        chunk_count_for_current_item = 0
                        start_pos = 0

                        while start_pos < duration_ms:
                            end_pos = min(start_pos + chunk_length_ms, duration_ms)

                            # Если остаток слишком маленький (менее 3 секунд), не создаем чанк
                            # Проверяем, был ли уже создан хотя бы один чанк из текущего элемента
                            if (duration_ms - start_pos) < 3000 and chunk_count_for_current_item > 0:
                                break

                            chunk = audio[start_pos:end_pos]
                            # Используем базовое имя оригинального файла для именования финальных чанков
                            chunk_filename = f"{base_name}_chunk_{total_chunks + chunk_count_for_current_item + 1:03d}.wav"
                            output_path = os.path.join(processed_dir, chunk_filename)
                            chunk.export(output_path, format="wav")

                            print(f"[INFO] Создан файл: {chunk_filename} (длительность: {(end_pos-start_pos)/1000:.2f} сек)")
                            chunk_count_for_current_item += 1
                            original_file_produced_chunks = True

                            # Сдвигаем с учетом перекрытия
                            start_pos = end_pos - overlap_ms

                        total_chunks += chunk_count_for_current_item

                except Exception as e:
                    print(f"[ERROR] Ошибка при обработке файла/чанка {current_file_name} с помощью pydub: {e}")
                    # Продолжаем работу, даже если один файл/чанк вызвал ошибку

            if original_file_produced_chunks:
                 processed_original_files_count += 1 # Считаем оригинальный файл обработанным, если из него создан хотя бы 1 финальный чанк

        except Exception as e:
            print(f"[ERROR] Непредвиденная ошибка при обработке исходного файла {file}: {e}")
            # Продолжаем работу, даже если исходный файл вызвал ошибку

        finally:
            # Очистка временных чанков ffmpeg для этого оригинального файла
            # Проверяем, были ли созданы временные чанки для этого файла
            temp_chunks_to_clean = glob.glob(os.path.join(temp_dir, f"{base_name}_temp_chunk_*.mp3"))
            if temp_chunks_to_clean:
                 print(f"[INFO] Очистка временных чанков для {file} из {temp_dir}")
                 for temp_chunk in temp_chunks_to_clean:
                     try:
                         if os.path.exists(temp_chunk):
                              os.remove(temp_chunk)
                     except Exception as e:
                         print(f"[WARNING] Не удалось удалить временный файл {temp_chunk}: {e}")


    # Очистка временной директории, если она пуста после обработки всех файлов
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
         print(f"[INFO] Временная директория {temp_dir} пуста. Удаление.")
         try:
             os.rmdir(temp_dir)
         except Exception as e:
             print(f"[WARNING] Не удалось удалить временную директорию {temp_dir}: {e}")


    if total_chunks > 0:
        print(f"[SUCCESS] Обработка завершена. Обработано исходных файлов (созданы чанки): {processed_original_files_count}. Создано всего {total_chunks} WAV-файлов.")
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
