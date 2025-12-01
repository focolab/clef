"""
Tests for CLEF CLI

Tests command-line interface argument parsing, config validation,
and experiment execution flow.
"""

import pytest
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
from io import StringIO

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from cli import (
    setup_argparser,
    validate_config_paths,
    load_configs,
    validate_configs,
    run_experiment,
    main,
)


@pytest.fixture
def temp_config_dir():
    """Create temporary directory with test config files."""
    temp_dir = tempfile.mkdtemp()
    config_dir = Path(temp_dir)
    
    # Create minimal valid YAML configs
    hw_config = """
backend: dummy
stim_interface: dummy
microscope_name: test_scope
"""
    
    exp_config = """
experiment_name: test_exp
output_dir: ./test_output
save_images: false
save_metadata: true
acquisition:
  num_frames: 10
  z_planes: 1
  z_step: 1.0
subject:
  genotype: test_strain
"""
    
    alg_config = """
algorithm_type: dummy
gui_mode: neural_imaging
algorithm_params:
  stim_cooldown_frames: 5
stimulus_params:
  duration_frames_options: [2]
  intensity_percent_options: [10]
"""
    
    (config_dir / "hardware.yaml").write_text(hw_config)
    (config_dir / "experiment.yaml").write_text(exp_config)
    (config_dir / "algorithm.yaml").write_text(alg_config)
    
    yield config_dir
    
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestArgumentParser:
    """Test CLI argument parsing."""
    
    def test_parser_setup(self):
        """Test parser is configured correctly."""
        parser = setup_argparser()
        
        assert parser.prog == 'clef-cli'
        assert parser.description == 'CLEF - Closed-Loop Experiment Framework'
    
    def test_parse_all_configs(self, temp_config_dir):
        """Test parsing all three config arguments."""
        parser = setup_argparser()
        
        args = parser.parse_args([
            '--hardware', str(temp_config_dir / 'hardware.yaml'),
            '--experiment', str(temp_config_dir / 'experiment.yaml'),
            '--algorithm', str(temp_config_dir / 'algorithm.yaml')
        ])
        
        assert args.hardware == str(temp_config_dir / 'hardware.yaml')
        assert args.experiment == str(temp_config_dir / 'experiment.yaml')
        assert args.algorithm == str(temp_config_dir / 'algorithm.yaml')
        assert not args.validate_config
    
    def test_parse_validate_flag(self):
        """Test --validate-config flag."""
        parser = setup_argparser()
        args = parser.parse_args(['--validate-config'])
        
        assert args.validate_config is True
    
    def test_parse_verbose_flag(self):
        """Test --verbose flag."""
        parser = setup_argparser()
        args = parser.parse_args(['--verbose'])
        
        assert args.verbose is True
    
    def test_parse_quiet_flag(self):
        """Test --quiet flag."""
        parser = setup_argparser()
        args = parser.parse_args(['--quiet'])
        
        assert args.quiet is True
    
    def test_help_message(self):
        """Test --help displays usage."""
        parser = setup_argparser()
        
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(['--help'])
        
        assert exc_info.value.code == 0
    
    def test_version_message(self):
        """Test --version displays version."""
        parser = setup_argparser()
        
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(['--version'])
        
        assert exc_info.value.code == 0
    
    def test_parse_partial_configs(self, temp_config_dir):
        """Test parsing with only some configs specified."""
        parser = setup_argparser()
        
        args = parser.parse_args([
            '--hardware', str(temp_config_dir / 'hardware.yaml')
        ])
        
        assert args.hardware is not None
        assert args.experiment is None
        assert args.algorithm is None
    
    def test_parse_no_args(self):
        """Test parsing with no arguments (uses defaults)."""
        parser = setup_argparser()
        args = parser.parse_args([])
        
        assert args.hardware is None
        assert args.experiment is None
        assert args.algorithm is None
        assert not args.validate_config


class TestConfigPathValidation:
    """Test config file path validation."""
    
    def test_validate_existing_files(self, temp_config_dir):
        """Test validation passes for existing files."""
        parser = setup_argparser()
        args = parser.parse_args([
            '--hardware', str(temp_config_dir / 'hardware.yaml'),
            '--experiment', str(temp_config_dir / 'experiment.yaml'),
            '--algorithm', str(temp_config_dir / 'algorithm.yaml')
        ])
        
        assert validate_config_paths(args) is True
    
    def test_validate_missing_file(self, temp_config_dir):
        """Test validation fails for missing file."""
        parser = setup_argparser()
        args = parser.parse_args([
            '--hardware', str(temp_config_dir / 'nonexistent.yaml')
        ])
        
        assert validate_config_paths(args) is False
    
    def test_validate_directory_instead_of_file(self, temp_config_dir):
        """Test validation fails when path is directory."""
        parser = setup_argparser()
        args = parser.parse_args([
            '--hardware', str(temp_config_dir)  # Directory, not file
        ])
        
        assert validate_config_paths(args) is False
    
    def test_validate_no_configs_specified(self):
        """Test validation passes when no configs specified (uses defaults)."""
        parser = setup_argparser()
        args = parser.parse_args([])
        
        # Should pass - no paths to validate
        assert validate_config_paths(args) is True
    
    def test_validate_warns_non_yaml_extension(self, temp_config_dir):
        """Test validation warns for non-YAML extension."""
        # Create file with wrong extension
        wrong_ext_file = temp_config_dir / "config.txt"
        wrong_ext_file.write_text("backend: dummy")
        
        parser = setup_argparser()
        args = parser.parse_args([
            '--hardware', str(wrong_ext_file)
        ])
        
        # Should still return True (warning, not error)
        assert validate_config_paths(args) is True


class TestConfigLoading:
    """Test configuration loading."""
    
    def test_load_all_configs(self, temp_config_dir):
        """Test loading all three config files."""
        parser = setup_argparser()
        args = parser.parse_args([
            '--hardware', str(temp_config_dir / 'hardware.yaml'),
            '--experiment', str(temp_config_dir / 'experiment.yaml'),
            '--algorithm', str(temp_config_dir / 'algorithm.yaml')
        ])
        
        config_manager = load_configs(args)
        
        assert config_manager is not None
        assert config_manager.hardware_config is not None
        assert config_manager.experiment_config is not None
        assert config_manager.algorithm_config is not None
    
    def test_load_with_defaults(self):
        """Test loading with default configs when none specified."""
        parser = setup_argparser()
        args = parser.parse_args([])
        
        config_manager = load_configs(args)
        
        # Should load defaults
        assert config_manager is not None
    
    def test_load_partial_configs(self, temp_config_dir):
        """Test loading with some configs specified, others default."""
        parser = setup_argparser()
        args = parser.parse_args([
            '--hardware', str(temp_config_dir / 'hardware.yaml')
        ])
        
        config_manager = load_configs(args)
        
        assert config_manager is not None
        assert config_manager.hardware_config is not None
    
    def test_load_invalid_yaml(self, temp_config_dir):
        """Test loading fails gracefully with invalid YAML."""
        # Create invalid YAML
        bad_yaml = temp_config_dir / "bad.yaml"
        bad_yaml.write_text("this is: [not: valid yaml")
        
        parser = setup_argparser()
        args = parser.parse_args([
            '--hardware', str(bad_yaml)
        ])
        
        config_manager = load_configs(args)
        
        assert config_manager is None
    
    def test_load_yaml_with_validation_errors(self, temp_config_dir):
        """Test loading fails with Pydantic validation errors."""
        # Create YAML with invalid values
        bad_config = temp_config_dir / "invalid.yaml"
        bad_config.write_text("""
backend: invalid_backend_name
stim_interface: dummy
""")
        
        parser = setup_argparser()
        args = parser.parse_args([
            '--hardware', str(bad_config)
        ])
        
        config_manager = load_configs(args)
        
        assert config_manager is None


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_validate_valid_configs(self, temp_config_dir):
        """Test validation passes for valid configs."""
        parser = setup_argparser()
        args = parser.parse_args([
            '--hardware', str(temp_config_dir / 'hardware.yaml'),
            '--experiment', str(temp_config_dir / 'experiment.yaml'),
            '--algorithm', str(temp_config_dir / 'algorithm.yaml')
        ])
        
        config_manager = load_configs(args)
        assert config_manager is not None
        
        assert validate_configs(config_manager) is True
    
    def test_validate_without_loaded_configs(self):
        """Test validation fails if configs not loaded."""
        from config.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        # Don't load any configs
        
        assert validate_configs(config_manager) is False


class TestExperimentExecution:
    """Test experiment execution flow."""
    
    def test_run_experiment_success(self, temp_config_dir):
        """Test successful experiment execution."""
        parser = setup_argparser()
        args = parser.parse_args([
            '--hardware', str(temp_config_dir / 'hardware.yaml'),
            '--experiment', str(temp_config_dir / 'experiment.yaml'),
            '--algorithm', str(temp_config_dir / 'algorithm.yaml')
        ])
        
        config_manager = load_configs(args)
        assert config_manager is not None
        
        # Run experiment
        success = run_experiment(config_manager)
        
        # With dummy backend, should complete successfully
        assert success is True
    
    def test_run_experiment_cleanup_on_interrupt(self, temp_config_dir):
        """Test cleanup happens on keyboard interrupt."""
        parser = setup_argparser()
        args = parser.parse_args([
            '--hardware', str(temp_config_dir / 'hardware.yaml'),
            '--experiment', str(temp_config_dir / 'experiment.yaml'),
            '--algorithm', str(temp_config_dir / 'algorithm.yaml')
        ])
        
        config_manager = load_configs(args)
        
        # Mock the run_acquisition_loop to raise KeyboardInterrupt
        with patch('engine.closed_loop_engine.ClosedLoopEngine.run_acquisition_loop',
                   side_effect=KeyboardInterrupt):
            success = run_experiment(config_manager)
        
        # Should return False but not crash
        assert success is False
    
    def test_run_experiment_cleanup_on_error(self, temp_config_dir):
        """Test cleanup happens on error."""
        parser = setup_argparser()
        args = parser.parse_args([
            '--hardware', str(temp_config_dir / 'hardware.yaml'),
            '--experiment', str(temp_config_dir / 'experiment.yaml'),
            '--algorithm', str(temp_config_dir / 'algorithm.yaml')
        ])
        
        config_manager = load_configs(args)
        
        # Mock initialization to raise error
        with patch('engine.closed_loop_engine.ClosedLoopEngine.initialize_hardware',
                   side_effect=RuntimeError("Test error")):
            success = run_experiment(config_manager)
        
        assert success is False


class TestMainFunction:
    """Test main CLI entry point."""
    
    def test_main_validate_only(self, temp_config_dir):
        """Test main with --validate-config flag."""
        test_args = [
            'clef-cli',
            '--validate-config',
            '--hardware', str(temp_config_dir / 'hardware.yaml'),
            '--experiment', str(temp_config_dir / 'experiment.yaml'),
            '--algorithm', str(temp_config_dir / 'algorithm.yaml')
        ]
        
        with patch('sys.argv', test_args):
            exit_code = main()
        
        # Should exit 0 after validation
        assert exit_code == 0
    
    def test_main_missing_file(self, temp_config_dir):
        """Test main with missing config file."""
        test_args = [
            'clef-cli',
            '--hardware', str(temp_config_dir / 'nonexistent.yaml')
        ]
        
        with patch('sys.argv', test_args):
            exit_code = main()
        
        # Should exit with error
        assert exit_code == 1
    
    def test_main_run_experiment(self, temp_config_dir):
        """Test main runs full experiment."""
        test_args = [
            'clef-cli',
            '--hardware', str(temp_config_dir / 'hardware.yaml'),
            '--experiment', str(temp_config_dir / 'experiment.yaml'),
            '--algorithm', str(temp_config_dir / 'algorithm.yaml')
        ]
        
        with patch('sys.argv', test_args):
            exit_code = main()
        
        # Should complete successfully with dummy backend
        assert exit_code == 0
    
    def test_main_verbose_logging(self, temp_config_dir):
        """Test main with verbose logging."""
        test_args = [
            'clef-cli',
            '--verbose',
            '--validate-config',
            '--hardware', str(temp_config_dir / 'hardware.yaml')
        ]
        
        with patch('sys.argv', test_args):
            exit_code = main()
        
        assert exit_code == 0
    
    def test_main_quiet_logging(self, temp_config_dir):
        """Test main with quiet logging."""
        test_args = [
            'clef-cli',
            '--quiet',
            '--validate-config',
            '--hardware', str(temp_config_dir / 'hardware.yaml')
        ]
        
        with patch('sys.argv', test_args):
            exit_code = main()
        
        assert exit_code == 0


class TestErrorHandling:
    """Test error handling and messages."""
    
    def test_helpful_error_for_bad_yaml(self, temp_config_dir):
        """Test helpful error message for malformed YAML."""
        bad_yaml = temp_config_dir / "bad.yaml"
        bad_yaml.write_text("this is: [invalid yaml")
        
        test_args = [
            'clef-cli',
            '--hardware', str(bad_yaml)
        ]
        
        with patch('sys.argv', test_args):
            exit_code = main()
        
        assert exit_code == 1
    
    def test_helpful_error_for_validation_failure(self, temp_config_dir):
        """Test helpful error message for validation failure."""
        invalid_config = temp_config_dir / "invalid.yaml"
        invalid_config.write_text("""
backend: not_a_valid_backend
""")
        
        test_args = [
            'clef-cli',
            '--hardware', str(invalid_config)
        ]
        
        with patch('sys.argv', test_args):
            exit_code = main()
        
        assert exit_code == 1
    
    def test_error_message_for_missing_file(self, temp_config_dir):
        """Test clear error for missing file."""
        test_args = [
            'clef-cli',
            '--hardware', str(temp_config_dir / 'does_not_exist.yaml')
        ]
        
        with patch('sys.argv', test_args):
            exit_code = main()
        
        assert exit_code == 1


class TestUsageExamples:
    """Test usage examples from documentation."""
    
    def test_example_all_configs(self, temp_config_dir):
        """Test: clef-cli --hardware hw.yaml --experiment exp.yaml --algorithm alg.yaml"""
        test_args = [
            'clef-cli',
            '--hardware', str(temp_config_dir / 'hardware.yaml'),
            '--experiment', str(temp_config_dir / 'experiment.yaml'),
            '--algorithm', str(temp_config_dir / 'algorithm.yaml')
        ]
        
        with patch('sys.argv', test_args):
            exit_code = main()
        
        assert exit_code == 0
    
    def test_example_validate_only(self, temp_config_dir):
        """Test: clef-cli --validate-config"""
        test_args = [
            'clef-cli',
            '--validate-config',
            '--hardware', str(temp_config_dir / 'hardware.yaml')
        ]
        
        with patch('sys.argv', test_args):
            exit_code = main()
        
        assert exit_code == 0
    
    def test_example_defaults(self):
        """Test: clef-cli (using all defaults)"""
        test_args = ['clef-cli', '--validate-config']
        
        with patch('sys.argv', test_args):
            exit_code = main()
        
        # Should work with defaults
        assert exit_code == 0
