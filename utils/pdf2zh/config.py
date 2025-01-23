import json
from pathlib import Path
from threading import RLock  # RLock으로 변경
import os
import copy


class ConfigManager:
    _instance = None
    _lock = RLock()  # RLock을 사용하여 동일 스레드에서 반복적으로 잠금 획득 가능

    @classmethod
    def get_instance(cls):
        """싱글톤 인스턴스 가져오기"""
        # 먼저 인스턴스가 존재하는지 확인하고, 없다면 잠금을 걸고 초기화 진행
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        # 중복 초기화를 방지
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True

        self._config_path = Path.home() / ".config" / "PDFMathTranslate" / "config.json"
        self._config_data = {}

        # 여기서는 잠금을 다시 걸 필요가 없음 (get_instance에서 이미 잠금 가능)
        self._ensure_config_exists()

    def _ensure_config_exists(self, isInit=True):
        """설정 파일이 존재하는지 확인하고, 없으면 기본 설정 생성"""
        # 여기서도 명시적으로 잠금을 걸 필요가 없음. RLock은 재진입 가능하기 때문에 문제가 없음.
        if not self._config_path.exists():
            if isInit:
                self._config_path.parent.mkdir(parents=True, exist_ok=True)
                self._config_data = {}  # 기본 설정 내용
                self._save_config()
            else:
                raise ValueError(f"config file {self._config_path} not found!")
        else:
            self._load_config()

    def _load_config(self):
        """config.json에서 설정을 로드"""
        with self._lock:  # 잠금을 걸어 스레드 안전성 확보
            with self._config_path.open("r", encoding="utf-8") as f:
                self._config_data = json.load(f)

    def _save_config(self):
        """설정을 config.json에 저장"""
        with self._lock:  # 잠금을 걸어 스레드 안전성 확보
            # 순환 참조를 제거하고 저장
            cleaned_data = self._remove_circular_references(self._config_data)
            with self._config_path.open("w", encoding="utf-8") as f:
                json.dump(cleaned_data, f, indent=4, ensure_ascii=False)

    def _remove_circular_references(self, obj, seen=None):
        """순환 참조 제거 (재귀적으로)"""
        if seen is None:
            seen = set()
        obj_id = id(obj)
        if obj_id in seen:
            return None  # 이미 처리된 객체는 순환 참조로 간주
        seen.add(obj_id)

        if isinstance(obj, dict):
            return {
                k: self._remove_circular_references(v, seen) for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._remove_circular_references(i, seen) for i in obj]
        return obj

    @classmethod
    def custome_config(cls, file_path):
        """사용자 지정 경로로 설정 파일 로드"""
        custom_path = Path(file_path)
        if not custom_path.exists():
            raise ValueError(f"Config file {custom_path} not found!")
        # 잠금
        with cls._lock:
            instance = cls()
            instance._config_path = custom_path
            # isInit=False를 전달하여, 파일이 없으면 에러 발생
            instance._ensure_config_exists(isInit=False)
            cls._instance = instance

    @classmethod
    def get(cls, key, default=None):
        """설정 값 가져오기"""
        instance = cls.get_instance()
        # 읽기 시 잠금을 걸 필요는 없으나, 일관성을 위해 저장 전후로 잠금을 거는 방식 사용
        if key in instance._config_data:
            return instance._config_data[key]

        # 환경 변수에서 해당 키가 있으면 사용하고, 설정 파일에 저장
        if key in os.environ:
            value = os.environ[key]
            instance._config_data[key] = value
            instance._save_config()
            return value

        # default 값이 있으면 설정하고 저장
        if default is not None:
            instance._config_data[key] = default
            instance._save_config()
            return default

        # 없으면 예외 발생
        return default

    @classmethod
    def set(cls, key, value):
        """설정 값 설정 및 저장"""
        instance = cls.get_instance()
        with instance._lock:
            instance._config_data[key] = value
            instance._save_config()

    @classmethod
    def get_translator_by_name(cls, name):
        """name에 따라 translator 설정 가져오기"""
        instance = cls.get_instance()
        translators = instance._config_data.get("translators", [])
        for translator in translators:
            if translator.get("name") == name:
                return translator["envs"]
        return None

    @classmethod
    def set_translator_by_name(cls, name, new_translator_envs):
        """name에 따라 translator 설정 추가 또는 갱신"""
        instance = cls.get_instance()
        with instance._lock:
            translators = instance._config_data.get("translators", [])
            for translator in translators:
                if translator.get("name") == name:
                    translator["envs"] = copy.deepcopy(new_translator_envs)
                    instance._save_config()
                    return
            translators.append(
                {"name": name, "envs": copy.deepcopy(new_translator_envs)}
            )
            instance._config_data["translators"] = translators
            instance._save_config()

    @classmethod
    def get_env_by_translatername(cls, translater_name, name, default=None):
        """name에 따라 translator 설정 가져오기"""
        instance = cls.get_instance()
        translators = instance._config_data.get("translators", [])
        for translator in translators:
            if translator.get("name") == translater_name.name:
                if translator["envs"][name]:
                    return translator["envs"][name]
                else:
                    with instance._lock:
                        translator["envs"][name] = default
                        instance._save_config()
                        return default

        with instance._lock:
            translators = instance._config_data.get("translators", [])
            for translator in translators:
                if translator.get("name") == translater_name.name:
                    translator["envs"][name] = default
                    instance._save_config()
                    return default
            translators.append(
                {
                    "name": translater_name.name,
                    "envs": copy.deepcopy(translater_name.envs),
                }
            )
            instance._config_data["translators"] = translators
            instance._save_config()
            return default

    @classmethod
    def delete(cls, key):
        """설정 값 삭제 및 저장"""
        instance = cls.get_instance()
        with instance._lock:
            if key in instance._config_data:
                del instance._config_data[key]
                instance._save_config()

    @classmethod
    def clear(cls):
        """모든 설정 삭제 및 저장"""
        instance = cls.get_instance()
        with instance._lock:
            instance._config_data = {}
            instance._save_config()

    @classmethod
    def all(cls):
        """모든 설정 항목 반환"""
        instance = cls.get_instance()
        # 읽기 작업만 하므로 일반적으로 잠금을 걸 필요는 없음. 단, 안전성을 위해 잠금을 걸 수도 있음.
        return instance._config_data

    @classmethod
    def remove(cls):
        """설정 파일 삭제"""
        instance = cls.get_instance()
        with instance._lock:
            os.remove(instance._config_path)
