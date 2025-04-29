import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0

def load_wav(full_path):
    try:
        sampling_rate, data = read(full_path)
        if data.dtype != np.float32:
            data = data.astype(np.float32) / MAX_WAV_VALUE
        return data, sampling_rate
    except Exception as e:
        print(f"Ошибка при загрузке WAV файла {full_path}: {e}")
        return np.array([], dtype=np.float32), 0

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C

def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)

def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)

mel_basis = {}
hann_window = {}

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """
    y: (Batch, Audio_samples)
    """
    global mel_basis, hann_window
    # Кэшируем mel_basis и hann_window по всем параметрам, влияющим на размерность
    key = f"{n_fft}_{num_mels}_{fmin}_{fmax}_{str(y.device)}"
    if key not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[key] = torch.from_numpy(mel).float().to(y.device)

    device_key = f"{win_size}_{str(y.device)}"
    if device_key not in hann_window:
        hann_window[device_key] = torch.hann_window(win_size).to(y.device)

    # Паддинг для STFT
    y = y.unsqueeze(1)  # (Batch, 1, Audio_samples)
    y = torch.nn.functional.pad(y, (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)  # (Batch, Audio_samples + 2 * pad_size)

    # STFT
    spec = torch.stft(
        y, n_fft=n_fft, hop_length=hop_size, win_length=win_size,
        window=hann_window[device_key], center=center, pad_mode='reflect',
        normalized=False, onesided=True, return_complex=True
    )  # (Batch, n_freq, num_frames)

    spec = spec.abs()  # (Batch, n_freq, num_frames)

    # Применение мел-фильтров (batch-овый вариант)
    # mel_basis: (num_mels, n_freq), spec: (Batch, n_freq, num_frames)
    # Нужно получить: (Batch, num_mels, num_frames)
    mel = torch.matmul(mel_basis[key], spec)  # (Batch, num_mels, num_frames)

    mel = spectral_normalize_torch(mel)  # (Batch, num_mels, num_frames)
    return mel

def get_dataset_filelist(a):
    wavs_base_dir = a.input_wavs_dir

    print(f"Чтение тренировочного списка файлов из: {a.input_training_file}")
    print(f"Ожидаемая базовая директория для поиска WAV: {wavs_base_dir}")

    training_files = []
    try:
        with open(a.input_training_file, 'r', encoding='utf-8') as fi:
            for line in fi:
                parts = line.strip().split('|')
                if len(parts) > 0 and len(parts[0]) > 0:
                    filename_only = os.path.basename(parts[0])
                    full_path = os.path.join(wavs_base_dir, filename_only)
                    training_files.append(full_path)
    except FileNotFoundError:
        print(f"Ошибка: Тренировочный файл списка датасета не найден: {a.input_training_file}")
        exit(1)
    except Exception as e:
        print(f"Ошибка при чтении тренировочного файла списка {a.input_training_file}: {e}")
        exit(1)

    print(f"Чтение валидационного списка файлов из: {a.input_validation_file}")
    validation_files = []
    try:
        with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
            for line in fi:
                parts = line.strip().split('|')
                if len(parts) > 0 and len(parts[0]) > 0:
                    filename_only = os.path.basename(parts[0])
                    full_path = os.path.join(wavs_base_dir, filename_only)
                    validation_files.append(full_path)
    except FileNotFoundError:
        print(f"Предупреждение: Валидационный файл списка датасета не найден: {a.input_validation_file}. Валидация будет пропущена.")
        validation_files = []
    except Exception as e:
        print(f"Ошибка при чтении валидационного файла списка {a.input_validation_file}: {e}")
        print("Валидация будет пропущена из-за ошибки чтения файла.")
        validation_files = []

    print(f"Проверка существования первых 5 тренировочных файлов в {wavs_base_dir}...")
    files_to_check = training_files[:5]
    if not files_to_check:
        print("Список тренировочных файлов пуст после чтения.")
    else:
        for i, f in enumerate(files_to_check):
            if not os.path.exists(f):
                print(f"ПРЕДУПРЕЖДЕНИЕ: Тренировочный WAV файл не найден по пути: {f}")

    if validation_files:
        print(f"Проверка существования первых 5 валидационных файлов в {wavs_base_dir}...")
        files_to_check_val = validation_files[:5]
        if not files_to_check_val:
            print("Список валидационных файлов пуст после чтения.")
        else:
            for i, f in enumerate(files_to_check_val):
                if not os.path.exists(f):
                    print(f"ПРЕДУПРЕЖДЕНИЕ: Валидационный WAV файл не найден по пути: {f}")
    else:
        print("Список валидационных файлов пуст, проверка пропущена.")

    return training_files, validation_files

class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate, fmin, fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=False, base_mels_path=None):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss if fmax_loss is not None else fmax
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path

    def __getitem__(self, index):
        filename = self.audio_files[index]

        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename)
            if audio.size == 0 or sampling_rate == 0:
                raise RuntimeError(f"Не удалось загрузить аудио файл: {filename}")

            audio = torch.FloatTensor(audio)
            # Нормализация только один раз!
            # Если уже нормализовано в load_wav, не делим ещё раз
            # audio = audio / MAX_WAV_VALUE  # УБРАНО

            self.cached_wav = audio
            if sampling_rate != self.sampling_rate:
                print(f"Предупреждение: Частота дискретизации файла {filename} ({sampling_rate} Hz) не совпадает с целевой ({self.sampling_rate} Hz).")
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1

        audio = audio.unsqueeze(0)  # (1, Audio_samples)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

            mel = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False)
        else:
            wav_basename = os.path.basename(filename)
            mel_filename = os.path.join(self.base_mels_path, wav_basename + '.npy')
            try:
                mel = np.load(mel_filename)
            except FileNotFoundError:
                print(f"Ошибка FileNotFoundError при загрузке файла мел-спектрограммы: {mel_filename}")
                raise RuntimeError(f"Не удалось загрузить мел-спектрограмму: {mel_filename}")
            except Exception as e:
                print(f"Ошибка при загрузке файла мел-спектрограммы {mel_filename}: {e}")
                raise RuntimeError(f"Ошибка при загрузке мел-спектрограммы: {mel_filename}") from e

            mel = torch.from_numpy(mel)
            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)
                if mel.size(2) >= frames_per_seg:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio_start = mel_start * self.hop_size
                    audio_end = (mel_start + frames_per_seg) * self.hop_size
                    audio = audio[:, audio_start:audio_end]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel_loss = mel_spectrogram(audio, self.n_fft, self.num_mels,
                                   self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax_loss,
                                   center=False)

        return (mel.squeeze(0), audio.squeeze(0), filename, mel_loss.squeeze(0))

    def __len__(self):
        return len(self.audio_files)
