"""
CLEF Command-Line Interface

Usage:
    python -m core.utils.clef_cli ring_attractor
    python -m core.utils.clef_cli --session s.yaml --io io.yaml --logic l.yaml
    python -m core.utils.clef_cli ring_attractor --validate-config
"""

import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from core.config.config_manager import ConfigManager
from core.io.io_manager import IOManager
from core.logic.logic_manager import LogicManager
from core.engine.closed_loop_engine import ClosedLoopEngine

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Where app configs live
APPS_CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "apps" / "config"

# The three required config types and filename patterns used to classify them
CONFIG_TYPES = {
    "session": "session_config",
    "io": "io_config",
    "logic": "logic_config",
}


def find_configs_by_root(root: str, config_dir: Path) -> Dict[str, Path]:
    """Search config_dir for YAML files whose name contains *root*.

    Classify each match as session/io/logic based on whether the filename
    contains the corresponding pattern (e.g. ``session_config``).

    Returns:
        Dict mapping config type ("session", "io", "logic") to Path.
        May contain fewer or more than 3 entries.
    """
    matches: List[Path] = []
    for p in config_dir.rglob("*.yaml"):
        if root in p.stem:
            matches.append(p)
    for p in config_dir.rglob("*.yml"):
        if root in p.stem:
            matches.append(p)

    # Also match if the root is a directory name containing the configs
    for p in config_dir.rglob("*.yaml"):
        if root in str(p.parent.name) and p not in matches:
            matches.append(p)
    for p in config_dir.rglob("*.yml"):
        if root in str(p.parent.name) and p not in matches:
            matches.append(p)

    classified: Dict[str, List[Path]] = {k: [] for k in CONFIG_TYPES}
    unclassified: List[Path] = []

    for p in matches:
        found_type = False
        for cfg_type, pattern in CONFIG_TYPES.items():
            if pattern in p.stem:
                classified[cfg_type].append(p)
                found_type = True
                break
        if not found_type:
            unclassified.append(p)

    # Build final mapping, detecting duplicates / missing
    result: Dict[str, Path] = {}
    errors: List[str] = []

    for cfg_type in CONFIG_TYPES:
        paths = classified[cfg_type]
        if len(paths) == 0:
            errors.append(f"  No {cfg_type} config found (expected filename containing '{CONFIG_TYPES[cfg_type]}')")
        elif len(paths) > 1:
            errors.append(f"  Multiple {cfg_type} configs found:")
            for p in paths:
                errors.append(f"    - {p}")
        else:
            result[cfg_type] = paths[0]

    if unclassified:
        errors.append("  Unclassified YAML files found:")
        for p in unclassified:
            errors.append(f"    - {p}")

    if errors:
        print(f"Config discovery errors for root '{root}':")
        for e in errors:
            print(e)
        return {}

    return result


def validate_configs(config_manager: ConfigManager) -> bool:
    """Attempt to load all three configs and report validation results."""
    try:
        print("Configuration validation passed.")
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False


def prompt_user(configs: Dict[str, Path]) -> bool:
    """Show discovered configs and ask for confirmation."""
    print("\nDiscovered configs:")
    for cfg_type, path in configs.items():
        print(f"  {cfg_type:>8}: {path}")
    print()
    response = input("Proceed? [y/n]: ").strip().lower()
    return response in ("y", "yes")


def run(config_manager: ConfigManager) -> bool:
    """Instantiate managers and run the closed loop."""
    engine = None
    try:
        io_manager = IOManager(config_manager)
        logic_manager = LogicManager(config_manager, io_manager)
        engine = ClosedLoopEngine(config_manager, io_manager, logic_manager)

        io_manager.connect()
        logic_manager.initialize_model()

        num_samples = config_manager.session_config.session_parameters.get("num_samples", 1)
        if num_samples == -1:
            logger.warning("num_samples=-1: running indefinitely until Ctrl+C")
        else:
            logger.info(f"Starting closed loop for {num_samples} samples...")
        engine.loop(iterations=num_samples)

        logger.info("Loop complete.")
        engine.save_md()
        engine.save_data()
        return True

    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C)")
        return False

    except Exception as e:
        logger.error(f"Run failed: {e}", exc_info=True)
        return False

    finally:
        if engine:
            engine.close()


def setup_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='clef',
        description='CLEF - Closed-Loop Experiment Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  clef ring_attractor          # auto-discover configs by root name
  clef --session s.yaml --io io.yaml --logic l.yaml  # explicit paths
  clef ring_attractor --validate-config               # validate only
        """
    )

    parser.add_argument(
        'config_root',
        nargs='?',
        default=None,
        help='Root name to search for configs (e.g. "ring_attractor")'
    )
    parser.add_argument('--session', type=str, metavar='PATH', help='Path to session config YAML')
    parser.add_argument('--io', type=str, metavar='PATH', help='Path to IO config YAML')
    parser.add_argument('--logic', type=str, metavar='PATH', help='Path to logic config YAML')
    parser.add_argument('--validate-config', action='store_true', help='Validate configs without running')
    parser.add_argument('--version', action='version', version='CLEF v2.0.0')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable debug logging')
    parser.add_argument('-q', '--quiet', action='store_true', help='Suppress all output except errors')

    return parser


def main() -> int:
    parser = setup_argparser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.ERROR)

    # Determine config paths
    if args.config_root:
        configs = find_configs_by_root(args.config_root, APPS_CONFIG_DIR)
        if len(configs) != 3:
            print("Expected exactly 3 configs (session, io, logic). Exiting.")
            return 1

        if not prompt_user(configs):
            print("Aborted.")
            return 0

        session_path = configs["session"]
        io_path = configs["io"]
        logic_path = configs["logic"]

    elif args.session or args.io or args.logic:
        # Explicit mode: all three must be provided
        if not (args.session and args.io and args.logic):
            print("When using explicit paths, all three must be provided: --session, --io, --logic")
            return 1

        for label, p in [("session", args.session), ("io", args.io), ("logic", args.logic)]:
            if not Path(p).exists():
                print(f"{label} config not found: {p}")
                return 1

        session_path = Path(args.session)
        io_path = Path(args.io)
        logic_path = Path(args.logic)

    else:
        parser.print_help()
        return 1

    # Load and validate configs
    config_manager = ConfigManager()
    try:
        config_manager.load_all_configs(
            session_path=session_path,
            io_path=io_path,
            logic_path=logic_path,
        )
    except Exception as e:
        print(f"Config loading/validation failed: {e}")
        return 1

    if args.validate_config:
        print("All configs valid.")
        return 0

    # Generate and inject session_id
    session_id = datetime.now().strftime("%Y%m%d-%H-%M-%S")
    config_manager.session_config.session_id = session_id
    logger.info(f"Session ID: {session_id}")

    # Create session directory inside sample_data_dir
    session_dir = Path(config_manager.session_config.sample_data_dir) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    config_manager.session_config.sample_data_dir = str(session_dir)
    logger.info(f"Session data dir: {session_dir}")

    # Run
    success = run(config_manager)
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
