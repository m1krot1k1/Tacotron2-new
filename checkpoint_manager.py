#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Модуль для проверки и управления чекпоинтами обучения Tacotron2 и HiFi-GAN
Автор: GitHub Copilot
"""

import os
import glob
import argparse
import torch
import json
from datetime import datetime

def find_latest_checkpoint(model_type, checkpoint_dir=None):
    """
    Находит последний чекпоинт для указанного типа модели
    
    Args:
        model_type: тип модели ('tacotron' или 'hifigan')
        checkpoint_dir: директория с чекпоинтами (если None, используется стандартная)
        
    Returns:
        путь к последнему чекпоинту или None, если чекпоинты не найдены
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if model_type.lower() == 'tacotron':
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(script_dir, "Tacotron2-main", "outdir")
        
        # Ищем все файлы .pt в директории
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
        
        if not checkpoint_files:
            return None
            
        # Сортируем по дате модификации (самый новый последним)
        checkpoint_files.sort(key=os.path.getmtime)
        return checkpoint_files[-1]
        
    elif model_type.lower() == 'hifigan':
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(script_dir, "Tacotron2-main", "hifigan", "checkpoints")
        
        # Ищем файлы генератора (g_*) в директории
        generator_files = glob.glob(os.path.join(checkpoint_dir, "g_*"))
        
        if not generator_files:
            # Проверяем наличие претренированной модели
            universal_path = os.path.join(script_dir, "Tacotron2-main", "hifigan", "UNIVERSAL_V1", "g_02500000")
            if os.path.exists(universal_path):
                return universal_path
            return None
            
        # Сортируем по номеру шага (извлекаем число из имени файла)
        generator_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1]))
        return generator_files[-1]
    
    return None

def get_checkpoint_info(checkpoint_path):
    """
    Получает информацию о чекпоинте
    
    Args:
        checkpoint_path: путь к файлу чекпоинта
        
    Returns:
        словарь с информацией о чекпоинте или None в случае ошибки
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        info = {
            'path': checkpoint_path,
            'filename': os.path.basename(checkpoint_path),
            'modified_time': datetime.fromtimestamp(os.path.getmtime(checkpoint_path)).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Для Tacotron2
        if 'state_dict' in checkpoint:
            info['type'] = 'tacotron2'
            if 'iteration' in checkpoint:
                info['iteration'] = checkpoint['iteration']
            
        # Для HiFi-GAN
        elif 'generator' in checkpoint:
            info['type'] = 'hifigan'
            if 'steps' in checkpoint:
                info['iteration'] = checkpoint['steps']
                
        return info
    except Exception as e:
        print(f"Ошибка при получении информации о чекпоинте {checkpoint_path}: {e}")
        return None

def backup_checkpoint(checkpoint_path, backup_dir=None):
    """
    Создает резервную копию чекпоинта
    
    Args:
        checkpoint_path: путь к файлу чекпоинта
        backup_dir: директория для резервных копий (если None, используется 'checkpoints_backup')
        
    Returns:
        путь к резервной копии или None в случае ошибки
    """
    try:
        if backup_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            backup_dir = os.path.join(script_dir, "checkpoints_backup")
            
        os.makedirs(backup_dir, exist_ok=True)
        
        filename = os.path.basename(checkpoint_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"{os.path.splitext(filename)[0]}_{timestamp}{os.path.splitext(filename)[1]}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Копируем файл
        import shutil
        shutil.copy2(checkpoint_path, backup_path)
        
        return backup_path
    except Exception as e:
        print(f"Ошибка при создании резервной копии чекпоинта {checkpoint_path}: {e}")
        return None

def save_checkpoint_status(model_type, status_file=None):
    """
    Сохраняет информацию о последнем чекпоинте в JSON файл
    
    Args:
        model_type: тип модели ('tacotron' или 'hifigan')
        status_file: путь к файлу статуса (если None, используется стандартный)
        
    Returns:
        True в случае успеха, False в случае ошибки
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        if status_file is None:
            status_file = os.path.join(script_dir, "checkpoint_status.json")
        
        checkpoint_path = find_latest_checkpoint(model_type)
        
        if checkpoint_path is None:
            print(f"Не найдено чекпоинтов для модели {model_type}")
            return False
            
        checkpoint_info = get_checkpoint_info(checkpoint_path)
        
        if checkpoint_info is None:
            return False
            
        # Загружаем текущий статус, если файл существует
        if os.path.exists(status_file):
            with open(status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
        else:
            status = {}
        
        # Обновляем информацию для указанного типа модели
        checkpoint_info['updated_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        status[model_type] = checkpoint_info
        
        # Сохраняем обновленный статус
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(status, f, indent=4, ensure_ascii=False)
            
        return True
    except Exception as e:
        print(f"Ошибка при сохранении статуса чекпоинта: {e}")
        return False

def main():
    """
    Основная функция для запуска из командной строки
    """
    parser = argparse.ArgumentParser(description='Управление чекпоинтами Tacotron2 и HiFi-GAN')
    parser.add_argument('--model', type=str, required=True, choices=['tacotron', 'hifigan'], 
                        help='Тип модели (tacotron или hifigan)')
    parser.add_argument('--action', type=str, default='info', choices=['info', 'backup', 'status'],
                        help='Действие: показать информацию, создать резервную копию или обновить статус')
    parser.add_argument('--checkpoint_dir', type=str, help='Директория с чекпоинтами')
    parser.add_argument('--backup_dir', type=str, help='Директория для резервных копий')
    parser.add_argument('--status_file', type=str, help='Путь к файлу статуса')
    
    args = parser.parse_args()
    
    # Находим последний чекпоинт
    checkpoint_path = find_latest_checkpoint(args.model, args.checkpoint_dir)
    
    if checkpoint_path is None:
        print(f"Не найдено чекпоинтов для модели {args.model}")
        return
    
    if args.action == 'info':
        # Выводим информацию о чекпоинте
        info = get_checkpoint_info(checkpoint_path)
        if info:
            print(f"Информация о последнем чекпоинте модели {args.model}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
    
    elif args.action == 'backup':
        # Создаем резервную копию
        backup_path = backup_checkpoint(checkpoint_path, args.backup_dir)
        if backup_path:
            print(f"Создана резервная копия: {backup_path}")
    
    elif args.action == 'status':
        # Обновляем статус
        if save_checkpoint_status(args.model, args.status_file):
            print(f"Статус чекпоинта модели {args.model} обновлен")

if __name__ == "__main__":
    main()