"""
CLEF Command-Line Interface

Provides command-line entry point for running CLEF experiments with YAML configs.

Usage:
    clef-cli --hardware hw.yaml --experiment exp.yaml --algorithm alg.yaml
    clef-cli --validate-config --hardware hw.yaml --experiment exp.yaml --algorithm alg.yaml
    clef-cli --help

Author: Raymond L. Dunn
Version: 1.0.0
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from config.config_manager import ConfigManager
from engine.closed_loop_engine import ClosedLoopEngine


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def setup_argparser() -> argparse.ArgumentParser:
    """
    Set up command-line argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog='clef-cli',
        description='CLEF - Closed-Loop Experiment Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run experiment with custom configs
  clef-cli --hardware my_hardware.yaml --experiment my_exp.yaml --algorithm my_alg.yaml
  
  # Validate configs without running
  clef-cli --validate-config --hardware hw.yaml --experiment exp.yaml --algorithm alg.yaml
  
  # Use defaults for unspecified configs
  clef-cli --hardware hw.yaml
  
  # Show version
  clef-cli --version

For more information, visit: https://github.com/your-org/clef
        """
    )
    
    # Config file arguments
    parser.add_argument(
        '--hardware',
        type=str,
        metavar='PATH',
        help='Path to hardware configuration YAML file (optional, uses defaults if not specified)'
    )
    
    parser.add_argument(
        '--experiment',
        type=str,
        metavar='PATH',
        help='Path to experiment configuration YAML file (optional, uses defaults if not specified)'
    )
    
    parser.add_argument(
        '--algorithm',
        type=str,
        metavar='PATH',
        help='Path to algorithm configuration YAML file (optional, uses defaults if not specified)'
    )
    
    # Action flags
    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate configuration files without running experiment'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='CLEF v2.0.0',
        help='Show version and exit'
    )
    
    # Logging control
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose debug logging'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress all output except errors'
    )
    
    return parser


def validate_config_paths(args: argparse.Namespace) -> bool:
    """
    Validate that specified config files exist.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        True if all specified paths exist, False otherwise
    """
    paths_to_check = []
    
    if args.hardware:
        paths_to_check.append(('hardware', args.hardware))
    if args.experiment:
        paths_to_check.append(('experiment', args.experiment))
    if args.algorithm:
        paths_to_check.append(('algorithm', args.algorithm))
    
    all_valid = True
    for config_type, path in paths_to_check:
        path_obj = Path(path)
        if not path_obj.exists():
            logger.error(f"{config_type.capitalize()} config file not found: {path}")
            all_valid = False
        elif not path_obj.is_file():
            logger.error(f"{config_type.capitalize()} config path is not a file: {path}")
            all_valid = False
        elif path_obj.suffix.lower() not in ['.yaml', '.yml']:
            logger.warning(f"{config_type.capitalize()} config file does not have .yaml/.yml extension: {path}")
    
    return all_valid


def load_configs(args: argparse.Namespace) -> Optional[ConfigManager]:
    """
    Load configuration files using ConfigManager.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        ConfigManager instance with loaded configs, or None on failure
    """
    try:
        config_manager = ConfigManager()
        
        # Convert string paths to Path objects (or None)
        hardware_path = Path(args.hardware) if args.hardware else None
        experiment_path = Path(args.experiment) if args.experiment else None
        algorithm_path = Path(args.algorithm) if args.algorithm else None
        
        logger.info("Loading configuration files...")
        config_manager.load_all_configs(
            hardware_path=hardware_path,
            experiment_path=experiment_path,
            algorithm_path=algorithm_path
        )
        
        logger.info("Configuration files loaded successfully")
        return config_manager
        
    except Exception as e:
        logger.error(f"Failed to load configurations: {e}")
        return None


def validate_configs(config_manager: ConfigManager) -> bool:
    """
    Validate loaded configurations.
    
    Args:
        config_manager: ConfigManager with loaded configs
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        logger.info("Validating configurations...")
        if config_manager.validate_config():
            logger.info("✓ Configuration validation passed")
            return True
        else:
            logger.error("✗ Configuration validation failed")
            return False
    except Exception as e:
        logger.error(f"Configuration validation error: {e}")
        return False


def run_experiment(config_manager: ConfigManager) -> bool:
    """
    Run CLEF experiment with loaded configurations.
    
    Args:
        config_manager: ConfigManager with loaded and validated configs
        
    Returns:
        True if experiment completed successfully, False otherwise
    """
    engine = None
    try:
        logger.info("Initializing CLEF engine...")
        engine = ClosedLoopEngine(
            hardware_config=config_manager.hardware_config,
            experiment_config=config_manager.experiment_config,
            algorithm_config=config_manager.algorithm_config
        )
        
        engine.run()
        
        logger.info("✓ Experiment completed successfully")
        return True
        
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user (Ctrl+C)")
        return False
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return False
        
    finally:
        if engine:
            logger.info("Cleaning up...")
            try:
                engine.cleanup()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    parser = setup_argparser()
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)
    
    # Validate file paths
    if not validate_config_paths(args):
        logger.error("Configuration file validation failed")
        return 1
    
    # Load configurations
    config_manager = load_configs(args)
    if not config_manager:
        return 1
    
    # Validate configurations
    if not validate_configs(config_manager):
        return 1
    
    # If only validating, exit here
    if args.validate_config:
        logger.info("Configuration validation complete (--validate-config mode)")
        return 0
    
    # Run experiment
    success = run_experiment(config_manager)
    return 0 if success else 1


def cli_entry_point():
    """
    Entry point for setup.py console_scripts.
    
    This wrapper catches SystemExit to ensure proper exit codes.
    """
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    cli_entry_point()
