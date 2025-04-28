#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для синтеза речи с использованием моделей Tacotron2 и HiFi-GAN
Автор: GitHub Copilot
"""

import os
import sys
import argparse
import numpy as np
import torch
from datetime import datetime

# Установка путей к Tacotron2 и HiFi-GAN
script_dir = os.path.dirname(os.path.abspath(__file__))
tacotron_dir = os.path.join(script_dir, "Tacotron2-main")
hifigan_dir = os.path.join(tacotron_dir, "hifigan")

# Добавляем пути к модулям в sys.path для корректного импорта
sys.path.append(tacotron_dir)
sys.path.append(hifigan_dir)

# Импортируем soundfile после установки путей к модулям
try:
    import soundfile as sf
except ImportError:
    print("Библиотека soundfile не установлена. Установка...")
    import subprocess
    subprocess.call([sys.executable, "-m", "pip", "install", "soundfile"])
    import soundfile as sf

def load_tacotron2_model(checkpoint_path):
    """
    Загружает модель Tacotron2 из чекпоинта
    """
    try:
        # Импортируем модули из Tacotron2
        from model import Tacotron2
        from hparams import create_hparams
        from text import text_to_sequence
        
        # Создаем гиперпараметры и модель
        hparams = create_hparams()
        model = Tacotron2(hparams)
        
        # Загружаем чекпоинт
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint_dict['state_dict'])
        
        # Переводим модель в режим оценки
        model.eval()
        return model, hparams
    except Exception as e:
        print(f"Ошибка загрузки модели Tacotron2: {e}")
        return None, None

def load_hifigan_model(checkpoint_path):
    """
    Загружает модель HiFi-GAN из чекпоинта
    """
    try:
        # Импортируем модули из HiFi-GAN
        from models import Generator
        from util import AttrDict
        import json
        
        # Определяем директорию с конфигом HiFi-GAN
        h = os.path.dirname(checkpoint_path)
        config_file = os.path.join(h, 'config.json')
        if not os.path.exists(config_file):
            # Проверяем наличие файла конфигурации в разных местах
            possible_config_paths = [
                os.path.join(hifigan_dir, "config.json"),
                os.path.join(hifigan_dir, "UNIVERSAL_V1", "config.json")
            ]
            
            for config_path in possible_config_paths:
                if os.path.exists(config_path):
                    config_file = config_path
                    break
        
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Не удалось найти config.json для HiFi-GAN")
        
        # Загружаем конфигурацию
        with open(config_file) as f:
            data = f.read()
        config = json.loads(data)
        h = AttrDict(config)
        
        # Создаем генератор
        generator = Generator(h)
        
        # Загружаем чекпоинт
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
        generator.load_state_dict(checkpoint_dict['generator'])
        generator.eval()
        generator.remove_weight_norm()
        
        return generator, h
    except Exception as e:
        print(f"Ошибка загрузки модели HiFi-GAN: {e}")
        return None, None

def generate_mel_spectrogram(model, hparams, text):
    """
    Генерирует мел-спектрограмму из текста с помощью Tacotron2
    """
    try:
        # Импортируем модуль для обработки текста
        from text import text_to_sequence
        
        # Преобразуем текст в последовательность
        sequence = text_to_sequence(text, ['russian_cleaners'])
        sequence = torch.IntTensor(sequence)[None, :].long()
        
        # Генерируем мел-спектрограмму
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        
        return mel_outputs_postnet.float()
    except Exception as e:
        print(f"Ошибка генерации мел-спектрограммы: {e}")
        return None

def generate_audio_from_mel(generator, h, mel):
    """
    Генерирует аудио из мел-спектрограммы с помощью HiFi-GAN
    """
    try:
        # Преобразуем мел-спектрограмму для HiFi-GAN
        mel = torch.tensor(mel.T[None]).float()
        
        # Генерируем аудио
        with torch.no_grad():
            audio = generator(mel)
            audio = audio.squeeze()
            audio = audio.cpu().numpy()
        
        return audio
    except Exception as e:
        print(f"Ошибка генерации аудио из мел-спектрограммы: {e}")
        return None

def griffin_lim_audio_from_mel(mel, hparams):
    """
    Генерирует аудио из мел-спектрограммы с помощью алгоритма Griffin-Lim
    (используется, если HiFi-GAN не доступен)
    """
    try:
        # Импортируем необходимые модули
        from audio_processing import griffin_lim, mel_to_magnitude
        
        # Преобразуем мел-спектрограмму в магнитуду
        mag = mel_to_magnitude(mel.cpu().numpy()[0], hparams)
        
        # Применяем алгоритм Griffin-Lim
        audio = griffin_lim(mag, hparams)
        
        return audio
    except Exception as e:
        print(f"Ошибка генерации аудио с помощью Griffin-Lim: {e}")
        return None

def synthesize_speech(text, output_file=None, tacotron_model_path=None, hifigan_model_path=None):
    """
    Синтезирует речь из текста с использованием моделей Tacotron2 и HiFi-GAN
    """
    # Получаем пути к моделям
    if tacotron_model_path is None:
        tacotron_model_path = os.path.join(script_dir, "model", "tacotron2_model.pt")
    
    if hifigan_model_path is None:
        hifigan_model_path = os.path.join(script_dir, "model", "hifigan_model")
    
    # Определяем выходной файл, если не указан
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(script_dir, "synthesized_audio")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"audio_{timestamp}.wav")
    
    # Проверка существования модели Tacotron2
    if not os.path.exists(tacotron_model_path):
        print(f"Ошибка: Модель Tacotron2 не найдена по пути {tacotron_model_path}")
        return False
    
    # Загружаем модель Tacotron2
    print(f"Загрузка модели Tacotron2 из {tacotron_model_path}...")
    tacotron_model, hparams = load_tacotron2_model(tacotron_model_path)
    if tacotron_model is None:
        return False
    
    # Генерируем мел-спектрограмму
    print("Генерация мел-спектрограммы из текста...")
    mel_spectrogram = generate_mel_spectrogram(tacotron_model, hparams, text)
    if mel_spectrogram is None:
        return False
    
    # Пробуем загрузить модель HiFi-GAN
    use_hifigan = os.path.exists(hifigan_model_path)
    if use_hifigan:
        print(f"Загрузка модели HiFi-GAN из {hifigan_model_path}...")
        hifigan_model, hifigan_config = load_hifigan_model(hifigan_model_path)
        if hifigan_model is not None:
            # Генерируем аудио с использованием HiFi-GAN
            print("Генерация аудио с помощью HiFi-GAN...")
            audio = generate_audio_from_mel(hifigan_model, hifigan_config, mel_spectrogram)
            if audio is None:
                use_hifigan = False
        else:
            use_hifigan = False
    
    # Если HiFi-GAN не доступен или произошла ошибка, используем Griffin-Lim
    if not use_hifigan:
        print("HiFi-GAN не доступен или произошла ошибка. Использование Griffin-Lim...")
        audio = griffin_lim_audio_from_mel(mel_spectrogram, hparams)
        if audio is None:
            return False
    
    # Сохраняем аудио
    print(f"Сохранение аудио в {output_file}...")
    if use_hifigan:
        # HiFi-GAN использует частоту дискретизации 22050 Гц
        sf.write(output_file, audio, 22050)
    else:
        # Griffin-Lim использует частоту дискретизации из hparams
        sf.write(output_file, audio, hparams.sampling_rate)
    
    print(f"Аудио успешно сгенерировано и сохранено в {output_file}")
    return True

def main():
    """
    Основная функция для запуска из командной строки
    """
    parser = argparse.ArgumentParser(description='Синтез речи из текста с помощью Tacotron2 и HiFi-GAN')
    parser.add_argument('--text', type=str, required=True, help='Текст для синтеза (на русском языке)')
    parser.add_argument('--output', type=str, help='Путь к выходному файлу WAV')
    parser.add_argument('--tacotron_model', type=str, help='Путь к модели Tacotron2')
    parser.add_argument('--hifigan_model', type=str, help='Путь к модели HiFi-GAN')
    
    args = parser.parse_args()
    
    success = synthesize_speech(
        args.text, 
        args.output, 
        args.tacotron_model, 
        args.hifigan_model
    )
    
    if not success:
        print("Ошибка при синтезе речи.")
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()