#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORK_DIR="$SCRIPT_DIR"
REPO_DIR="$WORK_DIR/Tacotron2-main"
AUDIO_DIR="$WORK_DIR/audio"
PROCESSED_DIR="$WORK_DIR/processed_audio"
TRANSCRIPTIONS_DIR="$WORK_DIR/transcriptions"
DATASET_DIR="$WORK_DIR/dataset"
METADATA_FILE="$DATASET_DIR/metadata.csv"
TRAIN_METADATA_FILE="$DATASET_DIR/train.csv"
VALIDATION_METADATA_FILE="$DATASET_DIR/validation.csv"
VENV_DIR="$WORK_DIR/tacotron2_venv_py37"

PROCESS_AUDIO_SCRIPT_REPO="$REPO_DIR/process_audio.py"
TRANSCRIBE_SCRIPT_REPO="$REPO_DIR/transcribe.py"
PREPARE_METADATA_SCRIPT_REPO="$REPO_DIR/prepare_metadata.py"
DECORATORS_SCRIPT_REPO="$REPO_DIR/decorators.py"

PROCESS_AUDIO_SCRIPT_WORK="$WORK_DIR/process_audio.py"
TRANSCRIBE_SCRIPT_WORK="$WORK_DIR/transcribe.py"
PREPARE_METADATA_SCRIPT_WORK="$WORK_DIR/prepare_metadata.py"


FINAL_MODEL_DIR="$WORK_DIR/model"
LAST_CHECKPOINT_DIR=""

# Требуемая версия CUDA (должна совпадать с PyTorch)
# ВАЖНО: Скрипт НЕ УСТАНАВЛИВАЕТ CUDA автоматически.
# Он только проверяет наличие установленной версии.
# Установите CUDA Toolkit вручную, следуя инструкциям NVIDIA для вашей ОС и нужной версии.
REQUIRED_CUDA_VERSION="11.7" # Укажите нужную версию CUDA

# Временная директория для сборки Apex
TEMP_BUILD_DIR="/tmp/apex_build_$(date +%s)_$(whoami)"


RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info()      { echo -e "${BLUE}[INFO] $1${NC}"; }
print_success()   { echo -e "${GREEN}[SUCCESS] $1${NC}"; }
print_warning()   { echo -e "${YELLOW}[WARNING] $1${NC}"; }
print_error()     { echo -e "${RED}[ERROR] $1${NC}"; }
wait_for_key()    { echo ""; read -p "Нажмите Enter, чтобы продолжить..."; echo ""; }

setup_virtual_env() {
    # setup_virtual_env start
    if [[ "$VIRTUAL_ENV" == "$VENV_DIR" ]]; then
        return 0
    fi

    print_info "Проверка виртуального окружения с Python 3.7..."

    if ! command -v python3.7 &> /dev/null; then
        print_warning "Python 3.7 не найден. Установка..."
        sudo apt update
        sudo apt install -y python3.7 python3.7-venv python3.7-dev
        if [ $? -ne 0 ]; then
            print_error "Не удалось установить Python 3.7. Пожалуйста, установите его вручную."
            return 1
        fi
    fi

    if [ ! -d "$VENV_DIR" ]; then
        print_info "Создание нового виртуального окружения с Python 3.7..."
        python3.7 -m venv "$VENV_DIR"
        if [ $? -ne 0 ]; then
            print_error "Не удалось создать виртуальное окружение."
            return 1
        fi
        print_success "Виртуальное окружение создано."
    else
        print_info "Виртуальное окружение уже существует."
    fi

    source "$VENV_DIR/bin/activate"

    if [[ "$VIRTUAL_ENV" != "$VENV_DIR" ]]; then
        print_error "Не удалось активировать виртуальное окружение."
        return 1
    fi

    print_success "Виртуальное окружение активировано: $(which python)"
    return 0
    # setup_virtual_env end
}

prepare_directories() {
    # prepare_directories start
    print_info "Создание необходимых директорий..."
    mkdir -p "$AUDIO_DIR" "$PROCESSED_DIR" "$TRANSCRIPTIONS_DIR" "$DATASET_DIR/wavs" "$FINAL_MODEL_DIR"
    print_success "Директории созданы."
    # prepare_directories end
}

install_dependencies() {
    # install_dependencies start
    print_info "Установка системных зависимостей (может потребоваться пароль)..."
    # Удалена автоматическая установка CUDA
    sudo apt update && sudo apt install -y \
        python3.7 \
        python3.7-dev \
        python3.7-venv \
        python3-pip \
        build-essential \
        libsndfile1-dev \
        libopenblas-dev \
        libatlas-base-dev \
        libhdf5-serial-dev \
        libblas-dev \
        liblapack-dev \
        gfortran \
        wget \
        git \
        ffmpeg \
        ninja-build # Добавлено для Apex

    if ! setup_virtual_env; then
        print_error "Не удалось настроить виртуальное окружение. Отмена установки зависимостей."
        return 1
    fi

    print_info "Установка Python-зависимостей в виртуальное окружение..."

    pip install --upgrade pip
    pip install wheel

    print_info "Установка совместимых версий библиотек..."
    pip install numpy==1.16.4 || { print_error "Не удалось установить numpy."; return 1; }
    pip install scipy==1.2.0 || { print_error "Не удалось установить scipy."; return 1; }

    print_info "Установка protobuf==3.20.1..."
    pip uninstall -y protobuf >/dev/null 2>&1 || true
    pip install protobuf==3.20.1 || { print_error "Не удалось установить protobuf==3.20.1."; return 1; }
    PROTOBUF_VERSION=$(pip show protobuf 2>/dev/null | grep Version | awk '{print $2}')
    if [[ "$PROTOBUF_VERSION" != "3.20.1" ]]; then
        print_warning "Установлена версия protobuf $PROTOBUF_VERSION, возможно несовместимая с TensorFlow 1.15.2. Ожидалась 3.20.1"
    else
         print_success "Protobuf установлен корректно (версия $PROTOBUF_VERSION)."
    fi

    print_info "Установка tensorflow==1.15.2..."
    # Установка TensorFlow теперь зависит от наличия CUDA
    if command -v nvidia-smi &> /dev/null && command -v nvcc &> /dev/null; then
        print_info "Обнаружена CUDA. Попытка установки GPU версии TensorFlow 1.15.2..."
        # Определяем путь к nvcc, если он найден
        NVCC_PATH=$(which nvcc)
        # Проверяем версию CUDA, если она найдена
        INSTALLED_CUDA_VERSION=$($NVCC_PATH -V | grep -oP 'release \K[0-9]+\.[0-9]+')
        if [[ "$INSTALLED_CUDA_VERSION" == "$REQUIRED_CUDA_VERSION" ]]; then
             pip install tensorflow-gpu==1.15.2 || { print_error "Не удалось установить tensorflow-gpu==1.15.2. Проверьте совместимость CUDA ($INSTALLED_CUDA_VERSION) и TensorFlow 1.15.2."; return 1; }
        else
             print_warning "Найдена CUDA версии $INSTALLED_CUDA_VERSION, но требуется $REQUIRED_CUDA_VERSION для tensorflow-gpu==1.15.2."
             print_info "Попытка установки CPU версии TensorFlow 1.15.2..."
             pip install tensorflow==1.15.2 || { print_error "Не удалось установить tensorflow==1.15.2."; return 1; }
        fi
    else
        print_info "CUDA не обнаружена. Установка CPU версии TensorFlow 1.15.2..."
        pip install tensorflow==1.15.2 || { print_error "Не удалось установить tensorflow==1.15.2."; return 1; }
    fi
     TF_VERSION=$(pip show tensorflow 2>/dev/null | grep Version | awk '{print $2}')
     if [[ "$TF_VERSION" != "1.15.2" ]]; then
        print_warning "Установлена версия TensorFlow $TF_VERSION, возможно несовместимая. Ожидалась 1.15.2"
    else
         print_success "TensorFlow установлен корректно (версия $TF_VERSION)."
    fi

    print_info "Установка numba==0.48.0..."
    pip uninstall -y numba >/dev/null 2>&1 || true
    pip install numba==0.48.0 || { print_error "Не удалось установить numba==0.48.0."; return 1; }
    NUMBA_VERSION=$(pip show numba 2>/dev/null | grep Version | awk '{print $2}')
     if [[ "$NUMBA_VERSION" != "0.48.0" ]]; then
        print_warning "Установлена версия numba $NUMBA_VERSION, возможно несовместимая. Oжидалась 0.48.0"
    else
         print_success "Numba установлен корректно (версия $NUMBA_VERSION)."
    fi

    print_info "Установка librosa==0.6.0..."
    pip install librosa==0.6.0 || { print_error "Не удалось установить librosa==0.6.0."; return 1; }
    LIBROSA_VERSION=$(pip show librosa 2>/dev/null | grep Version | awk '{print $2}')
     if [[ "$LIBROSA_VERSION" != "0.6.0" ]]; then
        print_warning "Установлена версия librosa $LIBROSA_VERSION, возможно несовместимая. Ожидалась 0.6.0"
    else
         print_success "Librosa установлен корректно (версия $LIBROSA_VERSION)."
    fi

    print_info "Установка matplotlib==2.1.0..."
    pip install matplotlib==2.1.0 || { print_error "Не удалось установить matplotlib==2.1.0."; return 1; }
    MATPLOTLIB_VERSION=$(pip show matplotlib 2>/dev/null | grep Version | awk '{print $2}')
     if [[ "$MATPLOTLIB_VERSION" != "2.1.0" ]]; then
        print_warning "Установлена версия matplotlib $MATPLOTLIB_VERSION, возможно несовместимая. Ожидалась 2.1.0"
    else
         print_success "Matplotlib установлен корректно (версия $MATPLOTLIB_VERSION)."
    fi

    pip install inflect==0.2.5 || { print_error "Не удалось установить inflect."; return 1; }
    pip install Unidecode==1.0.22 || { print_error "Не удалось установить Unidecode."; return 1; }
    pip install pillow || { print_error "Не удалось установить pillow."; return 1; }
    pip install pydub || { print_error "Не удалось установить pydub."; return 1; }
    pip install soundfile || { print_error "Не удалось установить soundfile."; return 1; }

    print_info "Установка tensorboardX..."
    pip install tensorboardX || { print_error "Не удалось установить tensorboardX."; return 1; }

    print_info "Установка transliterate..."
    pip install transliterate || { print_error "Не удалось установить transliterate."; return 1; }

    print_info "Установка opencv-python==4.5.5.64..."
    pip install opencv-python==4.5.5.64 || { print_error "Не удалось установить opencv-python==4.5.5.64."; return 1; }
     OPENCV_VERSION=$(pip show opencv-python 2>/dev/null | grep Version | awk '{print $2}')
     if [[ "$OPENCV_VERSION" != "4.5.5.64" ]]; then
        print_warning "Установлена версия opencv-python $OPENCV_VERSION. Ожидалась 4.5.5.64"
    else
         print_success "OpenCV-Python установлен корректно (версия $OPENCV_VERSION)."
    fi

    print_info "Установка Streamlit..."
    pip install streamlit || { print_error "Не удалось установить Streamlit."; return 1; }

    print_info "Установка Whisper версии, совместимой с Python 3.7..."
    pip uninstall -y whisper >/dev/null 2>&1 || true
    pip install git+https://github.com/openai/whisper.git@v20230117 || { print_error "Не удалось установить Whisper. Проверьте соединение с интернетом."; return 1; }
    WHISPER_VERSION=$(pip show whisper 2>/dev/null | grep Version | awk '{print $2}')
     if [[ "$WHISPER_VERSION" != "20230117" ]]; then
        print_warning "Установлена версия Whisper $WHISPER_VERSION. Ожидалась 20230117. Это может быть нормально при установке из ветки/коммита."
        if ! command -v whisper &> /dev/null; then
             print_error "Команда 'whisper' не найдена после установки. Whisper может быть установлен некорректно."
             return 1
        else
            print_success "Команда 'whisper' найдена."
        fi
    else
         print_success "Whisper установлен корректно (версия $WHISPER_VERSION)."
    fi

    print_info "Исправление проблемы с librosa.cache..."
    LIBROSA_INIT="$VENV_DIR/lib/python3.7/site-packages/librosa/cache.py"
    if [ -f "$LIBROSA_INIT" ]; then
        if ! grep -q "if hasattr(self, \"cachedir\")" "$LIBROSA_INIT"; then
            cp "$LIBROSA_INIT" "${LIBROSA_INIT}.bak"
            sed -i 's/if self.cachedir is not None and self.level >= level:/if hasattr(self, "cachedir") and self.cachedir is not None and self.level >= level:/g' "$LIBROSA_INIT"
            print_success "Файл librosa/cache.py исправлен."
        else
            print_info "Файл librosa/cache.py уже исправлен."
        fi
    else
        print_warning "Файл librosa/cache.py не найден ($LIBROSA_INIT). Пропуск исправления."
    fi

    print_info "Исправление проблемы с numba.decorators..."
    LIBROSA_DECORATORS="$VENV_DIR/lib/python3.7/site-packages/librosa/util/decorators.py"
    if [ -f "$LIBROSA_DECORATORS" ]; then
        if grep -q "from numba.decorators import jit as optional_jit" "$LIBROSA_DECORATORS"; then
            cp "$LIBROSA_DECORATORS" "${LIBROSA_DECORATORS}.bak"
            if [ ! -f "$DECORATORS_SCRIPT_REPO" ]; then
                print_error "Файл decorators.py не найден в репозитории: $DECORATORS_SCRIPT_REPO. Не удалось применить исправление numba.decorators."
                return 1
            fi
            print_info "Копирование decorators.py из репозитория в директорию виртуального окружения..."
            cp "$DECORATORS_SCRIPT_REPO" "$LIBROSA_DECORATORS"
            if [ $? -ne 0 ]; then
                print_error "Не удалось скопировать decorators.py в директорию виртуального окружения."
                return 1
            fi
            print_success "Файл librosa/util/decorators.py успешно заменен на версию из репозитория."
        else
            print_info "Файл librosa/util/decorators.py уже исправлен."
        fi
    else
        print_warning "Файл librosa/util/decorators.py не найден ($LIBROSA_DECORATORS). Пропуск исправления."
    fi

    # --- БЛОК ПРОВЕРКИ НАЛИЧИЯ CUDA Toolkit ---
    echo ""
    print_info "Проверка наличия установленного CUDA Toolkit $REQUIRED_CUDA_VERSION..."
    NVCC_PATH=$(which nvcc)

    if [ -z "$NVCC_PATH" ]; then
        print_error "nvcc не найден."
        print_info "CUDA Toolkit не установлен или не добавлен в PATH."
        print_info "Пожалуйста, установите CUDA Toolkit $REQUIRED_CUDA_VERSION вручную, следуя инструкциям на сайте NVIDIA:"
        print_info "https://developer.nvidia.com/cuda-downloads"
        print_info "После установки CUDA перезапустите терминал и запустите этот скрипт снова."
        return 1 # Возвращаем ошибку, так как CUDA не найдена
    fi

    INSTALLED_CUDA_VERSION=$($NVCC_PATH -V | grep -oP 'release \K[0-9]+\.[0-9]+')
    echo "Найден nvcc: $NVCC_PATH (Версия: $INSTALLED_CUDA_VERSION)"

    if [[ "$INSTALLED_CUDA_VERSION" != "$REQUIRED_CUDA_VERSION" ]]; then
        print_warning "Установленная версия CUDA ($INSTALLED_CUDA_VERSION) не совпадает с требуемой ($REQUIRED_CUDA_VERSION)."
        print_info "Это может вызвать проблемы совместимости с некоторыми библиотеками (например, TensorFlow-GPU, Apex)."
        print_info "Рекомендуется установить CUDA Toolkit версии $REQUIRED_CUDA_VERSION вручную."
        print_info "https://developer.nvidia.com/cuda-downloads"
        print_info "Если вы уверены в совместимости, можете продолжить, но возможны ошибки."
        # Не возвращаем ошибку сразу, позволяем пользователю продолжить на свой страх и риск
    else
        print_success "Установленная версия CUDA ($INSTALLED_CUDA_VERSION) совпадает с требуемой ($REQUIRED_CUDA_VERSION)."
    fi

    # Определяем CUDA_HOME для Apex на основе найденного nvcc
    CUDA_HOME_FOR_APEX=$(dirname $(dirname "$NVCC_PATH"))
    print_info "Используем CUDA_HOME для установки Apex: $CUDA_HOME_FOR_APEX"

    # --- КОНЕЦ БЛОКА ПРОВЕРКИ НАЛИЧИЯ CUDA Toolkit ---


    # --- БЛОК УСТАНОВКИ NVIDIA Apex ---
    echo ""
    print_info "Проверка и установка NVIDIA Apex..."

    # Запуск установки Apex от имени оригинального пользователя (если скрипт запущен с sudo)
    # Иначе, от имени текущего пользователя
    RUN_USER="$USER"
    if [ -n "$SUDO_USER" ]; then
        RUN_USER="$SUDO_USER"
    fi

    print_info "Запуск установки Apex от имени $RUN_USER..."

    # Проверка существования venv перед попыткой активации от имени пользователя
    if [ ! -d "$VENV_DIR" ]; then
        print_error "Директория виртуального окружения не найдена: $VENV_DIR. Невозможно установить Apex."
        return 1
    fi

    # Проверка наличия репозитория Tacotron2, так как Apex клонируется во временную папку
    # и не зависит напрямую от файлов репозитория Tacotron2, но для порядка
    # убедимся, что репозиторий склонирован, т.к. Apex устанавливается в рамках
    # install_dependencies, которая вызывается после clone_repository в do_all_steps.
     if [ ! -d "$REPO_DIR" ]; then
         print_error "Директория репозитория не найдена: $REPO_DIR. Невозможно установить Apex без склонированного репозитория."
         return 1
     fi


    sudo -u "$RUN_USER" bash -c "
        set -e
        echo '--- Начало блока установки Apex (от имени \$USER) ---'
        VENV_PATH_SUB='$VENV_DIR'
        TEMP_BUILD_DIR_SUB='$TEMP_BUILD_DIR'
        CUDA_HOME_FOR_APEX_SUB='$CUDA_HOME_FOR_APEX'

        echo 'Активация виртуального окружения: \$VENV_PATH_SUB'
        source \"\$VENV_PATH_SUB/bin/activate\"
        echo \"Python активного окружения: \$(which python)\"

        if python -c 'import apex' > /dev/null 2>&1; then
            echo 'Apex уже установлен в этом окружении. Пропускаем установку.'
            exit 0
        fi

        echo 'Установка зависимости сборки: packaging...'
        pip install packaging || { echo 'ОШИБКА: Не удалось установить packaging.'; exit 1; }

        echo 'Клонирование NVIDIA Apex во временную папку: \$TEMP_BUILD_DIR_SUB/apex'
        mkdir -p \"\$TEMP_BUILD_DIR_SUB\" || { echo 'ОШИБКА: Не удалось создать временную директорию.'; exit 1; }
        git clone https://github.com/NVIDIA/apex \"\$TEMP_BUILD_DIR_SUB/apex\" || { echo 'ОШИБКА: Не удалось клонировать Apex.'; exit 1; }
        cd \"\$TEMP_BUILD_DIR_SUB/apex\" || { echo 'ОШИБКА: Не удалось перейти в директорию Apex.'; exit 1; }
        echo 'Текущая директория: \$(pwd)'

        echo 'Запуск установки Apex (может занять несколько минут)...'
        export CUDA_HOME=\"\$CUDA_HOME_FOR_APEX_SUB\"
        export CPATH=\"\$CUDA_HOME/include:\$CPATH\"
        export LIBRARY_PATH=\"\$CUDA_HOME/lib64:\$LIBRARY_PATH\"
        echo \"CUDA_HOME для pip: \$CUDA_HOME\"
        echo \"CPATH: \$CPATH\"
        echo \"LIBRARY_PATH: \$LIBRARY_PATH\"

        pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \\
            --config-settings \"--build-option=--cpp_ext\" \\
            --config-settings \"--build-option=--cuda_ext\" \\
            ./ || { echo 'ОШИБКА: Не удалось установить Apex.'; exit 1; }

        echo 'Проверка импорта Apex после установки...'
        python -c 'from apex import amp; print(\"Apex AMP успешно импортирован!\")' || { echo 'ОШИБКА: Не удалось импортировать Apex.'; exit 1; }

        echo 'Возврат в домашнюю директорию...'
        cd \"$HOME\" || { echo 'ОШИБКА: Не удалось вернуться в домашнюю директорию.'; exit 1; }
        echo 'Деактивация виртуального окружения...'
        deactivate > /dev/null 2>&1
        echo '--- Блок установки Apex завершен успешно ---'
    "
    INSTALL_APEX_STATUS=$?

    # --- БЛОК ОЧИСТКИ ---
    echo ""
    print_info "Очистка временной директории сборки $TEMP_BUILD_DIR..."
    # Удаляем временную директорию от имени root, если скрипт запущен с sudo
    if [ "$EUID" -eq 0 ]; then
        rm -rf "$TEMP_BUILD_DIR"
    else
        rm -rf "$TEMP_BUILD_DIR" 2>/dev/null || true # Удаляем от имени текущего пользователя, если не root
    fi
    echo "Временная директория удалена."
    # --- КОНЕЦ БЛОКА ОЧИСТКИ ---


    if [ $INSTALL_APEX_STATUS -ne 0 ]; then
        print_error "Во время установки NVIDIA Apex произошла ошибка. Код ошибки: $INSTALL_APEX_STATUS"
        print_info "Проверьте вывод выше для получения дополнительной информации."
        return 1
    else
        print_success "NVIDIA Apex успешно установлен."
    fi


    print_success "Все необходимые зависимости установлены."
    return 0
    # install_dependencies end
}

clone_repository() {
    # clone_repository start
    print_info "Клонирование репозитория Tacotron2..."

    if ! setup_virtual_env; then
        print_error "Не удалось настроить виртуальное окружение. Отмена клонирования."
        return 1
    fi

    if [ ! -d "$REPO_DIR" ]; then
        print_info "Директория репозитория '$REPO_DIR' не найдена. Выполняю git clone..."
        git clone https://github.com/m1krot1k1/Tacotron2-new.git "$REPO_DIR"
        CLONE_STATUS=$?
        if [ $CLONE_STATUS -ne 0 ]; then
            print_error "Не удалось клонировать репозиторий Tacotron2. Код ошибки: $CLONE_STATUS"
            print_info "Проверьте подключение к интернету и доступность репозитория."
            return 1
        fi
        print_success "Репозиторий успешно склонирован в '$REPO_DIR'."
    else
        print_warning "Директория репозитория '$REPO_DIR' уже существует. Попытка обновления..."
        cd "$REPO_DIR" || { print_error "Не удалось перейти в директорию репозитория '$REPO_DIR'."; return 1; }

        if [ ! -d ".git" ]; then
            print_error "Директория '$REPO_DIR' не является рабочим Git-репозиторием (отсутствует папка .git)."
            print_warning "Пожалуйста, вручную удалите директорию: rm -rf '$REPO_DIR'"
            print_info "После удаления запустите скрипт снова. Он клонирует репозиторий с нуля."
            cd "$WORK_DIR" || { print_error "Не удалось вернуться в рабочую директорию."; return 1; }
            return 1
        fi

        git pull
        PULL_STATUS=$?
        if [ $PULL_STATUS -ne 0 ]; then
            print_warning "Не удалось обновить репозиторий (git pull завершился с кодом $PULL_STATUS). Использование текущей версии."
            print_info "Возможные причины: нет доступа к интернету, изменения в локальных файлах репозитория, или проблемы с Git."
        else
            print_success "Репозиторий успешно обновлён."
        fi
        cd "$WORK_DIR" || { print_error "Не удалось вернуться в рабочую директорию."; return 1; }
    fi

    print_success "Репозиторий подготовлен (файлы репозитория не модифицированы)."
    return 0
    # clone_repository end
}

instruct_recording() {
    # instruct_recording start
    print_info "Инструкции по записи голоса:"
    echo ""
    echo "1. Запишите несколько десятков или сотен коротких аудиофайлов (5-10 секунд каждый)."
    echo "2. Каждая запись должна содержать одно или несколько предложений."
    echo "3. Говорите четко, с нормальной скоростью и интонацией."
    echo "4. Старайтесь использовать разнообразные фразы и интонации."
    echo "5. Записывайте в тихом помещении без фонового шума."
    echo "6. Сохраняйте файлы в формате MP3."
    echo "7. Желательно, чтобы имена файлов были в формате: audio_001.mp3, audio_002.mp3 и т.д."
    echo ""
    echo "Если у вас есть длинные записи, скрипт автоматически разделит их на фрагменты по 10 секунд."
    echo "Эти фрагменты будут названы в формате: исходное_имя_chunk_001.wav и т.д."
    echo ""
    echo "После записи всех файлов, поместите их в директорию:"
    echo "$AUDIO_DIR"
    echo ""
    wait_for_key
    # instruct_recording end
}

check_audio_files() {
    # check_audio_files start
    print_info "Проверка наличия исходных аудиофайлов в $AUDIO_DIR..."
    if [ ! -d "$AUDIO_DIR" ]; then
        print_warning "Директория исходных аудиофайлов '$AUDIO_DIR' не найдена. Будет создана."
        mkdir -p "$AUDIO_DIR"
    fi

    MP3_COUNT=$(find "$AUDIO_DIR" -maxdepth 1 -name "*.mp3" | wc -l)
    if [ "$MP3_COUNT" -eq 0 ]; then
        print_error "В директории $AUDIO_DIR не найдено MP3-файлов."
        print_info "Пожалуйста, сначала запишите аудиофайлы и поместите их в директорию:"
        echo "$AUDIO_DIR"
        return 1
    else
        print_success "Найдено $MP3_COUNT MP3-файлов в $AUDIO_DIR."
        return 0
    fi
    # check_audio_files end
}

process_audio_files() {
    # process_audio_files start
    print_info "Обработка аудиофайлов и нарезка длинных на чанки..."

    if ! setup_virtual_env; then
        print_error "Не удалось настроить виртуальное окружение. Отмена обработки аудио."
        return 1
    fi

    python -c "import pydub" 2>/dev/null
    if [ $? -ne 0 ]; then
         print_warning "pydub не установлен. Попытка установки..."
         pip install pydub || { print_error "Не удалось установить pydub."; return 1; }
    fi


    if ! check_audio_files; then
        print_error "Отмена обработки аудио из-за отсутствия исходных MP3 файлов."
        return 1
    fi

    print_info "Копирование скрипта process_audio.py из репозитория в рабочую директорию..."
    cp "$PROCESS_AUDIO_SCRIPT_REPO" "$PROCESS_AUDIO_SCRIPT_WORK"
    if [ $? -ne 0 ]; then
        print_error "Не удалось скопировать скрипт process_audio.py."
        return 1
    fi
    print_success "Скрипт process_audio.py скопирован."

    print_info "Очистка директории для обработанных аудио: $PROCESSED_DIR..."
    rm -rf "$PROCESSED_DIR"
    mkdir -p "$PROCESSED_DIR"
    print_success "Директория $PROCESSED_DIR очищена."

    python "$PROCESS_AUDIO_SCRIPT_WORK" --input_dir "$AUDIO_DIR" --output_dir "$PROCESSED_DIR" --chunk_length 10
    PROCESS_STATUS=$?
    WAV_COUNT=$(find "$PROCESSED_DIR" -maxdepth 1 -name "*.wav" | wc -l)
    if [ "$PROCESS_STATUS" -ne 0 ] || [ "$WAV_COUNT" -eq 0 ]; then
        print_error "Не удалось создать WAV-файлы. Проверьте входные MP3-файлы и вывод скрипта выше."
        return 1
    else
        print_success "Аудиофайлы обработаны и разделены. Создано $WAV_COUNT WAV-файлов."
        print_info "Файлы сохранены в директории: $PROCESSED_DIR"
        return 0
    fi
    # process_audio_files end
}

transcribe_audio() {
    # transcribe_audio start
    print_info "Транскрипция аудиофайлов с помощью Whisper..."

    if ! setup_virtual_env; then
        print_error "Не удалось настроить виртуальное окружение. Отмена транскрипции."
        return 1
    fi

    python -c "import whisper" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_warning "Whisper не установлен или несовместим. Попытка установки совместимой версии..."
        pip install git+https://github.com/openai/whisper.git@v20230117 || { print_error "Не удалось установить Whisper. Проверьте соединение с интернетом."; return 1; }
        python -c "import whisper" 2>/dev/null
        if [ $? -ne 0 ]; then
            print_error "Не удалось установить Whisper после попытки установки."
            return 1
        fi
    fi

    WAV_COUNT=$(find "$PROCESSED_DIR" -maxdepth 1 -name "*.wav" | wc -l)
    if [ "$WAV_COUNT" -eq 0 ]; then
        print_error "В директории $PROCESSED_DIR не найдено WAV-файлов."
        print_info "Пожалуйста, сначала обработайте аудиофайлы (опция 4)."
        return 1
    fi

    print_info "Копирование скрипта transcribe.py из репозитория в рабочую директорию..."
    cp "$TRANSCRIBE_SCRIPT_REPO" "$TRANSCRIBE_SCRIPT_WORK"
    if [ $? -ne 0 ]; then
        print_error "Не удалось скопировать скрипт transcribe.py."
        return 1
    fi
    print_success "Скрипт transcribe.py скопирован."


    print_info "Очистка директории транскрипций: $TRANSCRIPTIONS_DIR..."
    rm -rf "$TRANSCRIPTIONS_DIR"
    mkdir -p "$TRANSCRIPTIONS_DIR"
    print_success "Директория $TRANSCRIPTIONS_DIR очищена."

    python "$TRANSCRIBE_SCRIPT_WORK" --input_dir "$PROCESSED_DIR" --output_dir "$TRANSCRIPTIONS_DIR" --model base
    TRANSCRIPT_STATUS=$?
    TXT_COUNT=$(find "$TRANSCRIPTIONS_DIR" -maxdepth 1 -name "*.txt" | wc -l)
    if [ "$TRANSCRIPT_STATUS" -ne 0 ] || [ "$TXT_COUNT" -eq 0 ]; then
        print_error "Транскрипция завершена с ошибками или не создано файлов транскрипций. Код ошибки: $TRANSCRIPT_STATUS"
        print_info "Проверьте вывод скрипта транскрипции выше."
        return 1
    else
        print_success "Транскрипция завершена. Создано $TXT_COUNT файлов транскрипций."
        print_info "Результаты сохранены в директории: $TRANSCRIPTIONS_DIR"
        return 0
    fi
    # transcribe_audio end
}

create_dataset() {
    # create_dataset start
    print_info "Создание датасета для Tacotron 2..."

    if ! setup_virtual_env; then
        print_error "Не удалось настроить виртуальное окружение. Отмена создания датасета."
        return 1
    fi

    TXT_COUNT=$(find "$TRANSCRIPTIONS_DIR" -maxdepth 1 -name "*.txt" | wc -l)
    if [ "$TXT_COUNT" -eq 0 ]; then
        print_error "В директории $TRANSCRIPTIONS_DIR не найдено файлов транскрипций."
        print_info "Пожалуйста, сначала выполните транскрипцию аудиофайлов (опция 5)."
        return 1
    fi

     WAV_COUNT=$(find "$PROCESSED_DIR" -maxdepth 1 -name "*.wav" | wc -l)
    if [ "$WAV_COUNT" -eq 0 ]; then
        print_error "В директории $PROCESSED_DIR не найдено WAV-файлов."
        print_info "Пожалуйста, сначала обработайте аудиофайлы (опция 4)."
        return 1
    fi

    print_info "Копирование скрипта prepare_metadata.py из репозитория в рабочую директорию..."
    cp "$PREPARE_METADATA_SCRIPT_REPO" "$PREPARE_METADATA_SCRIPT_WORK"
    if [ $? -ne 0 ]; then
        print_error "Не удалось скопировать скрипт prepare_metadata.py."
        return 1
    fi
    print_success "Скрипт prepare_metadata.py скопирован."


    mkdir -p "$DATASET_DIR/wavs"
    print_info "Очистка старых файлов метаданных в $DATASET_DIR..."
    rm -f "$METADATA_FILE" "$TRAIN_METADATA_FILE" "$VALIDATION_METADATA_FILE"
    touch "$METADATA_FILE" "$TRAIN_METADATA_FILE" "$VALIDATION_METADATA_FILE"
    print_success "Старые файлы метаданных очищены."

    python "$PREPARE_METADATA_SCRIPT_WORK" \
        --trans_dir "$TRANSCRIPTIONS_DIR" \
        --audio_dir "$PROCESSED_DIR" \
        --dataset_dir "$DATASET_DIR" \
        --metadata_file_all "$METADATA_FILE" \
        --train_file "$TRAIN_METADATA_FILE" \
        --validation_file "$VALIDATION_METADATA_FILE" \
        --validation_split 0.1

    DATASET_STATUS=$?
    if [ "$DATASET_STATUS" -ne 0 ] || \
       [ ! -f "$TRAIN_METADATA_FILE" ] || [ ! -s "$TRAIN_METADATA_FILE" ] || \
       [ ! -f "$VALIDATION_METADATA_FILE" ] || [ ! -s "$VALIDATION_METADATA_FILE" ]
    then
        print_error "Не удалось создать файлы метаданных train.csv и/или validation.csv, или они пусты."
        return 1
    fi

    DATASET_WAV_COUNT=$(find "$DATASET_DIR/wavs" -maxdepth 1 -name "*.wav" | wc -l)
    if [ "$DATASET_WAV_COUNT" -eq 0 ]; then
        print_error "В директории $DATASET_DIR/wavs не найдено WAV-файлов после создания датасета. Проверьте вывод скрипта prepare_metadata.py выше."
        return 1
    fi

    print_success "Датасет создан и разделен успешно. Файлы: $TRAIN_METADATA_FILE и $VALIDATION_METADATA_FILE"
    print_success "WAV-файлы скопированы в: $DATASET_DIR/wavs"
    return 0
    # create_dataset end
}
train_model() {
    # train_model start
    print_info "Запуск обучения модели Tacotron 2..."

    if ! setup_virtual_env; then
        print_error "Не удалось настроить виртуальное окружение. Отмена обучения."
        return 1
    fi

    print_info "Проверка необходимых Python-зависимостей перед обучением..."
    python -c "import tensorflow; import torch; import librosa; import numba; import tensorboardX; import transliterate; import cv2; import streamlit; import soundfile; import PIL; import json; import apex" 2>/dev/null # Добавлена проверка Apex
    if [ $? -ne 0 ]; then
        print_error "Некоторые критические Python-зависимости не установлены или не работают корректно (включая Apex)."
        print_info "Пожалуйста, убедитесь, что вы установили все зависимости, включая CUDA Toolkit ($REQUIRED_CUDA_VERSION) и Apex (опция 2)."
        return 1
    else
         print_success "Основные Python-зависимости присутствуют."
    fi

    if [ ! -d "$REPO_DIR" ]; then
        print_error "Директории репозитория не найдена: $REPO_DIR."
        print_info "Пожалуйста, сначала клонируйте репозиторий (опция 1)."
        return 1
    fi

    if [ ! -f "$TRAIN_METADATA_FILE" ] || [ ! -s "$TRAIN_METADATA_FILE" ] || \
       [ ! -f "$VALIDATION_METADATA_FILE" ] || [ ! -s "$VALIDATION_METADATA_FILE" ]
    then
        print_error "Файлы метаданных train.csv и/или validation.csv не найдены или пусты: $TRAIN_METADATA_FILE, $VALIDATION_METADATA_FILE"
        print_info "Пожалуйста, сначала создайте датасет (опция 6)."
        return 1
    fi

    DATASET_WAV_COUNT=$(find "$DATASET_DIR/wavs" -maxdepth 1 -name "*.wav" | wc -l)
    if [ "$DATASET_WAV_COUNT" -eq 0 ]; then
        print_error "В директории $DATASET_DIR/wavs не найдено WAV-файлов."
        print_info "Пожалуйста, сначала создайте датасет (опция 6)."
        return 1
    fi

    OUTPUT_DIR="$REPO_DIR/outdir"
    LOG_DIR="$REPO_DIR/logdir"
    mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

    REPO_DATA_DIR="$REPO_DIR/data"
    mkdir -p "$REPO_DATA_DIR/wavs"

    print_info "Копирование датасета в директорию репозитория '$REPO_DATA_DIR'..."
    print_info "Очистка старых данных в $REPO_DATA_DIR..."
    rm -f "$REPO_DATA_DIR/metadata.csv" "$REPO_DATA_DIR/train.csv" "$REPO_DATA_DIR/validation.csv"
    rm -rf "$REPO_DATA_DIR/wavs/"*
    print_success "Старые данные в $REPO_DATA_DIR очищены."
    if [ -f "$TRAIN_METADATA_FILE" ] && [ -f "$VALIDATION_METADATA_FILE" ]; then
         cp "$TRAIN_METADATA_FILE" "$REPO_DATA_DIR/"
         if [ $? -ne 0 ]; then print_error "Не удалось скопировать файл train.csv в репозиторий."; return 1; fi
         print_success "Файл train.csv скопирован в '$REPO_DATA_DIR/train.csv'."

         cp "$VALIDATION_METADATA_FILE" "$REPO_DATA_DIR/"
         if [ $? -ne 0 ]; then print_error "Не удалось скопировать файл validation.csv в репозиторий."; return 1; fi
         print_success "Файл validation.csv скопирован в '$REPO_DATA_DIR/validation.csv'."
    else
         print_error "Исходные файлы метаданных (train.csv и/или validation.csv) не найдены. Отмена копирования в репозиторий."
         return 1
    fi

    if [ -d "$DATASET_DIR/wavs" ] && [ "$(ls -A "$DATASET_DIR/wavs")" ]; then
         print_info "Копирование WAV-файлов из '$DATASET_DIR/wavs/' в '$REPO_DATA_DIR/wavs/'..."
         cp -r "$DATASET_DIR/wavs/"* "$REPO_DATA_DIR/wavs/"
         if [ $? -ne 0 ]; then print_error "Не удалось скопировать WAV-файлы в репозиторий."; return 1; fi
         print_success "WAV-файлы скопированы."
    elif [ -d "$DATASET_DIR/wavs" ]; then
         print_warning "Исходная директория WAV пуста: $DATASET_DIR/wavs. Скопировано 0 файлов. Обучение может не начаться."
    else
         print_error "Исходная директория WAV не найдена: $DATASET_DIR/wavs. Отмена копирования WAV в репозиторий."
         return 1
    fi

    print_success "Датасет скопирован в директорию репозитория."



    if [ -d "$DATASET_DIR/wavs" ] && [ "$(ls -A "$DATASET_DIR/wavs")" ]; then
         print_info "Копирование WAV-файлов из '$DATASET_DIR/wavs/' в '$REPO_DATA_DIR/wavs/'..."
         cp -r "$DATASET_DIR/wavs/"* "$REPO_DATA_DIR/wavs/"
         if [ $? -ne 0 ]; then print_error "Не удалось скопировать WAV-файлы в репозиторий."; return 1; fi
         print_success "WAV-файлы скопированы."
    elif [ -d "$DATASET_DIR/wavs" ]; then
         print_warning "Исходная директория WAV пуста: $DATASET_DIR/wavs. Скопировано 0 файлов. Обучение может не начаться."
    else
         print_error "Исходная директория WAV не найдена: $DATASET_DIR/wavs. Отмена копирования WAV в репозиторий."
         return 1
    fi

    print_success "Датасет скопирован в директорию репозитория."

    print_info "Запуск процесса обучения..."
    cd "$REPO_DIR" || { print_error "Не удалось перейти в директорию репозитория $REPO_DIR."; return 1; }

    print_info "Запуск TensorBoard в фоне..."
    tensorboard --logdir="$LOG_DIR" --host=0.0.0.0 --port=6006 &
    TENSORBOARD_PID=$!
    print_success "TensorBoard запущен в фоне с PID $TENSORBOARD_PID. Доступен по адресу http://0.0.0.0:6006"
    print_info "Запуск Streamlit демо в фоне..."
    if [ -f "demo.py" ]; then
      streamlit run demo.py &
      STREAMLIT_PID=$!
      print_success "Streamlit демо запущен в фоне с PID $STREAMLIT_PID."
    else
      print_warning "Файл demo.py не найден в директории репозитория. Пропуск запуска Streamlit."
    fi

    print_info "Выполняемая команда обучения:"
    echo "python train.py --output_directory=\"$OUTPUT_DIR\" --log_directory=\"$LOG_DIR\""
    python train.py --output_directory="$OUTPUT_DIR" --log_directory="$LOG_DIR" &> >(tee -a "$LOG_DIR/train.log")

    TRAIN_STATUS=$?

    print_info "Остановка фоновых процессов (TensorBoard и Streamlit)..."
    kill $TENSORBOARD_PID 2>/dev/null || true
    if [ -n "$STREAMLIT_PID" ]; then
      kill $STREAMLIT_PID 2>/dev/null || true
    fi
    print_success "Фоновые процессы остановлены."


    cd "$WORK_DIR" || { print_error "Не удалось вернуться в рабочую директорию скрипта."; return 1; }

    if [ $TRAIN_STATUS -eq 0 ]; then
        LAST_CHECKPOINT_DIR="$OUTPUT_DIR"
        print_info "Путь к чекпоинтам обучения установлен: $LAST_CHECKPOINT_DIR"
    else
        LAST_CHECKPOINT_DIR=""
    fi


    deactivate 2>/dev/null || true
    print_success "Виртуальное окружение деактивировано."
    if [ $TRAIN_STATUS -eq 0 ]; then
        print_success "Обучение успешно завершено. Результаты сохранены в: $OUTPUT_DIR"
        print_info "Проверьте директорию $OUTPUT_DIR на наличие файлов контрольных точек модели (.pt)."
    else
        print_error "Во время обучения произошла ошибка. Код ошибки: $TRAIN_STATUS"
        print_info "Проверьте логи в директории $LOG_DIR/train.log и вывод выше для получения дополнительной информации."
        if [ $TRAIN_STATUS -eq 1 ]; then
             print_info "Возможные причины ошибки (код 1): нехватка памяти (GPU/RAM), ошибки в данных датасета, или несовместимые версии библиотек."
             print_info "Если используете GPU и столкнулись с нехваткой памяти, попробуйте указать конкретную GPU с помощью переменной окружения CUDA_VISIBLE_DEVICES (например, CUDA_VISIBLE_DEVICES=0 bash ваш_скрипт.sh) или уменьшить размер батча (batch_size в hparams.py)."
             print_info "Убедитесь, что ваши драйверы NVIDIA и установленная версия CUDA совместимы с версией TensorFlow 1.15.2 и PyTorch (если используется)."
        else
            print_info "Неизвестная ошибка обучения. Проверьте логи для деталей."
        fi
    fi

    return $TRAIN_STATUS
    # train_model end
}

save_final_model() {
    # save_final_model start
    if [ -z "$LAST_CHECKPOINT_DIR" ]; then
        print_warning "Не удалось найти директорию с чекпоинтами обучения. Финальная модель не будет сохранена отдельно."
        return 1
    fi

    print_info "Поиск последнего чекпоинта в $LAST_CHECKPOINT_DIR для сохранения финальной модели..."

    LAST_CHECKPOINT=$(ls -v "$LAST_CHECKPOINT_DIR"/*.pt 2>/dev/null | tail -n 1)

    if [ -z "$LAST_CHECKPOINT" ]; then
        print_error "Не удалось найти ни одного файла чекпоинта (.pt) в директории $LAST_CHECKPOINT_DIR."
        print_info "Убедитесь, что обучение прошло успешно и чекпоинты были созданы."
        return 1
    fi

    print_info "Найден последний чекпоинт: $(basename "$LAST_CHECKPOINT")"

    mkdir -p "$FINAL_MODEL_DIR"
    if [ $? -ne 0 ]; then
        print_error "Не удалось создать директорию для финальной модели: $FINAL_MODEL_DIR."
        return 1
    fi

    FINAL_MODEL_PATH="$FINAL_MODEL_DIR/final_model.pt"
    print_info "Копирование и переименование последнего чекпоинта в '$FINAL_MODEL_PATH'..."
    cp "$LAST_CHECKPOINT" "$FINAL_MODEL_PATH"
    if [ $? -ne 0 ]; then
        print_error "Не удалось скопировать и переименовать файл чекпоинта в '$FINAL_MODEL_PATH'."
        return 1
    fi

    print_success "Финальная модель успешно сохранена: $FINAL_MODEL_PATH"
    return 0
    # save_final_model end
}

fine_tune_hifigan() {
    # fine_tune_hifigan start
    print_info "Запуск обучения HiFi-GAN (файнтюнинг)..."

    if ! setup_virtual_env; then
        print_error "Не удалось настроить виртуальное окружение. Отмена обучения HiFi-GAN."
        return 1
    fi

    HIFIGAN_DIR="$REPO_DIR/hifigan"
    HIFIGAN_TRAIN_SCRIPT="$HIFIGAN_DIR/train.py"

    if [ ! -d "$HIFIGAN_DIR" ]; then
        print_error "Директория HiFi-GAN не найдена: $HIFIGAN_DIR."
        print_info "Убедитесь, что репозиторий содержит папку HiFi-GAN."
        return 1
    fi

    if [ ! -f "$HIFIGAN_TRAIN_SCRIPT" ]; then
        print_error "Скрипт обучения HiFi-GAN не найден: $HIFIGAN_TRAIN_SCRIPT."
        return 1
    fi

    if [ ! -f "$TRAIN_METADATA_FILE" ] || [ ! -s "$TRAIN_METADATA_FILE" ]; then
        print_error "Файл метаданных train.csv не найден или пуст: $TRAIN_METADATA_FILE."
        print_info "Пожалуйста, сначала создайте датасет (опция 6)."
        return 1
    fi

    HIFIGAN_OUTPUT_DIR="$HIFIGAN_DIR/output"
    mkdir -p "$HIFIGAN_OUTPUT_DIR"

    print_info "Запуск процесса обучения HiFi-GAN..."
    cd "$HIFIGAN_DIR" || { print_error "Не удалось перейти в директорию HiFi-GAN: $HIFIGAN_DIR."; return 1; }

    python train.py --input_wavs_dir "$DATASET_DIR/wavs" \
                    --input_training_file "$TRAIN_METADATA_FILE" \
                    --input_validation_file "$VALIDATION_METADATA_FILE" \
                    --checkpoint_path "$HIFIGAN_OUTPUT_DIR" &> >(tee -a "$HIFIGAN_OUTPUT_DIR/train.log")

    HIFIGAN_TRAIN_STATUS=$?

    cd "$WORK_DIR" || { print_error "Не удалось вернуться в рабочую директорию: $WORK_DIR."; return 1; }

    if [ $HIFIGAN_TRAIN_STATUS -eq 0 ]; then
        print_success "Обучение HiFi-GAN успешно завершено. Результаты сохранены в: $HIFIGAN_OUTPUT_DIR"
    else
        print_error "Во время обучения HiFi-GAN произошла ошибка. Код ошибки: $HIFIGAN_TRAIN_STATUS"
        print_info "Проверьте логи в директории $HIFIGAN_OUTPUT_DIR/train.log для получения дополнительной информации."
    fi

    return $HIFIGAN_TRAIN_STATUS
    # fine_tune_hifigan end
}

do_all_steps() {
    # do_all_steps start
    print_info "Запуск всех шагов последовательно..."
    prepare_directories && \
    clone_repository && \
    install_dependencies && \
    instruct_recording &&
    process_audio_files && \
    transcribe_audio && \
    create_dataset && \
    train_model && \
    fine_tune_hifigan

    if [ $? -eq 0 ]; then
        save_final_model
        return $?
    else
        print_error "Шаг обучения завершился с ошибкой. Пропуск сохранения финальной модели."
        return $?
    fi

    # do_all_steps end
}

show_menu() {
    # show_menu start
    clear
    echo "============================================"
    echo "  СОЗДАНИЕ ДАТАСЕТА ДЛЯ TACOTRON 2 ИЗ ГОЛОСА  "
    echo "=============================================="
    echo ""
    echo "Выберите опцию:"
    echo "0. Выполнить все шаги последовательно (рекомендуется)"
    echo "1. Клонировать репозиторий Tacotron2 (файлы репозитория не модифицируются)"
    echo "2. Установить зависимости (Python 3.7 + библиотеки, ПРОВЕРКА CUDA, Apex)" # Обновлено описание
    echo "3. Инструкции по записи голоса"
    echo "4. Обработать аудиофайлы (MP3 -> WAV, нарезка)"
    echo "5. Выполнить транскрипцию с помощью Whisper"
    echo "6. Создать датасет (метаданные + копирование WAV)"
    echo "7. Запустить обучение модели Tacotron 2"
    echo "8. Сохранить финальную модель (из последнего чекпоинта)"
    echo "9. Выход"
    echo ""
    read -p "Введите номер опции: " choice
    case $choice in
        0) do_all_steps ;;
        1) clone_repository ;;
        2) prepare_directories; install_dependencies ;;
        3) instruct_recording ;;
        4) process_audio_files ;;
        5) transcribe_audio ;;
        6) create_dataset ;;
        7) train_model ;;
        8) save_final_model ;;
        9) print_info "Выход из программы."; exit 0 ;;
        *) print_error "Неверный выбор. Пожалуйста, попробуйте снова." ;;
    esac
    wait_for_key
    # show_menu end
}

main() {
    # main start
    print_info "Добро пожаловать в программу создания датасета для Tacotron 2!"
    prepare_directories
    while true; do
        show_menu
    done
    # main end
}

main

deactivate 2>/dev/null || true

exit 0
