import threading
from typing import Any

# default configuration
defaults = {
    "bilinear_projection_num_newton_steps": 10,
    "cross_entropy_method": "fixed_point",
    "cross_entropy_num_steps": 10,
    "cross_entropy_lambda": 5.0,
}


class Config:
    """Global configuration singleton for the PJAX framework.

    Thread-safe access to algorithmic parameters like projection
    step counts and cross-entropy solver settings.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance

    def _init(self):
        self._config = defaults.copy()

    def update(self, key: str, value: Any) -> None:
        """Update a configuration value."""
        with self._lock:
            self._config[key] = value

    def reset(self):
        self._init()

    def __getattr__(self, name: str) -> Any:
        """Access a configuration value by attribute."""
        return self._config[name]

    def __getitem__(self, name: str) -> Any:
        """Access a configuration value by key."""
        return self._config[name]


config = Config()


def __getattr__(name: str) -> Any:
    """Access a configuration value by attribute."""
    return getattr(config, name)


def __getitem__(name: str) -> Any:
    """Access a configuration value by key."""
    return config[name]


def update(key: str, value: Any) -> None:
    """Update a configuration value."""
    config.update(key, value)
