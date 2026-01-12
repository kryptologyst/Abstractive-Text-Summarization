"""
Configuration management for the abstractive text summarization project.
"""

import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    name: str = "facebook/bart-large-cnn"
    max_length: int = 50
    min_length: int = 25
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    device: str = "auto"
    batch_size: int = 1
    use_fast_tokenizer: bool = True


@dataclass
class UIConfig:
    """Configuration for UI parameters."""
    title: str = "Abstractive Text Summarization"
    description: str = "Generate concise summaries using state-of-the-art transformer models"
    max_text_length: int = 5000
    default_max_summary_length: int = 50
    default_min_summary_length: int = 25
    show_evaluation_metrics: bool = True
    enable_batch_processing: bool = True


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None


@dataclass
class AppConfig:
    """Main application configuration."""
    model: ModelConfig
    ui: UIConfig
    logging: LoggingConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AppConfig':
        """Create AppConfig from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            ui=UIConfig(**config_dict.get('ui', {})),
            logging=LoggingConfig(**config_dict.get('logging', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert AppConfig to dictionary."""
        return {
            'model': asdict(self.model),
            'ui': asdict(self.ui),
            'logging': asdict(self.logging)
        }


class ConfigManager:
    """Manages application configuration loading and saving."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, uses default path.
        """
        self.config_path = config_path or Path("config/config.yaml")
        self._config: Optional[AppConfig] = None
    
    def load_config(self) -> AppConfig:
        """
        Load configuration from file.
        
        Returns:
            AppConfig object with loaded configuration
        """
        if self._config is not None:
            return self._config
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                        config_dict = yaml.safe_load(f)
                    elif self.config_path.suffix.lower() == '.json':
                        config_dict = json.load(f)
                    else:
                        raise ValueError(f"Unsupported config file format: {self.config_path.suffix}")
                
                self._config = AppConfig.from_dict(config_dict)
                
            except Exception as e:
                print(f"Error loading config file: {e}")
                print("Using default configuration")
                self._config = self._get_default_config()
        else:
            print(f"Config file not found: {self.config_path}")
            print("Using default configuration")
            self._config = self._get_default_config()
        
        return self._config
    
    def save_config(self, config: AppConfig) -> None:
        """
        Save configuration to file.
        
        Args:
            config: AppConfig object to save
        """
        # Ensure config directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = config.to_dict()
            
            with open(self.config_path, 'w') as f:
                if self.config_path.suffix.lower() == '.yaml' or self.config_path.suffix.lower() == '.yml':
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                elif self.config_path.suffix.lower() == '.json':
                    json.dump(config_dict, f, indent=2)
                else:
                    raise ValueError(f"Unsupported config file format: {self.config_path.suffix}")
            
            self._config = config
            print(f"Configuration saved to: {self.config_path}")
            
        except Exception as e:
            print(f"Error saving config file: {e}")
    
    def _get_default_config(self) -> AppConfig:
        """Get default configuration."""
        return AppConfig(
            model=ModelConfig(),
            ui=UIConfig(),
            logging=LoggingConfig()
        )
    
    def get_config(self) -> AppConfig:
        """Get current configuration."""
        if self._config is None:
            return self.load_config()
        return self._config


def create_default_config_file(config_path: Path = Path("config/config.yaml")) -> None:
    """Create a default configuration file."""
    config_manager = ConfigManager(config_path)
    default_config = config_manager._get_default_config()
    config_manager.save_config(default_config)


if __name__ == "__main__":
    # Create default config file
    create_default_config_file()
    print("Default configuration file created at config/config.yaml")
