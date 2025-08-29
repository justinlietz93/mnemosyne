#!/usr/bin/env python3
"""Universal Codebase XML Generator

A standalone utility that scans any directory structure and generates comprehensive XML reports containing:
1. Report header with generation metadata
2. Table of contents for easy navigation
3. ASCII codebase map visualization (within XML structure)
4. Hierarchical XML representation with file paths and content
5. Analysis report with statistics and configuration details

Features:
- Respects .gitignore patterns automatically
- Scans from wherever the script is executed
- Handles nested directories and large codebases
- Generates comprehensive XML output with all components
- Safe file reading with encoding detection
- Comprehensive error handling
- LLM-friendly structured XML format

Usage:
    python universal_codebase_generator.py [options]
    
Options:
    --output-dir PATH    Directory to save output files (default: ./output)
    --max-file-size MB   Maximum file size to include in MB (default: 10)
    --include-hidden     Include hidden files/directories (default: False)
    --exclude PATTERNS   Additional patterns to exclude (comma-separated)
"""

import os
import sys
import argparse
import fnmatch
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from datetime import datetime
import mimetypes

# Try to import chardet, fall back to basic encoding detection
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
    chardet = None

# Try to import yaml for configuration
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('codebase_generator.log')
    ]
)
logger = logging.getLogger(__name__)

class ConfigLoader:
    """Load and manage configuration from YAML file."""
    
    def __init__(self, config_path: str = "map_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with fallback defaults."""
        default_config = {
            'file_processing': {
                'max_file_size_mb': 10.0,
                'max_content_size_kb': 100,
                'include_hidden': False,
                'ignore_gitignore': False
            },
            'output': {
                'output_dir': './output',
                'xml_only': False,
                'ascii_only': False,
                'combined_output': True
            },
            'exclusions': {
                'additional_patterns': []
            },
            'critical_skip_dirs': [
                '__pycache__', 'venv', 'env', 'ENV', '.venv', '.env', 'virtualenv',
                'node_modules', '.git', '.svn', '.hg', 'build', 'dist', 'lib',
                'lib64', '.cache', 'cache', 'tmp', 'temp', '.mypy_cache', 
                '.pytest_cache', 'htmlcov', '.tox', '.coverage', 'logs',
                '.eggs', 'eggs', 'wheels', 'parts', 'var', 'downloads', 'output'
            ],
            'skip_extensions': [
                '.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib', '.exe', 
                '.bin', '.log', '.bak', '.backup', '.old', '.swp', '.swo'
            ],
            'text_extensions': [
                '.txt', '.md', '.rst', '.py', '.js', '.ts', '.html', '.css', '.scss',
                '.yaml', '.yml', '.json', '.xml', '.ini', '.cfg', '.conf', '.toml',
                '.sh', '.bash', '.bat', '.ps1', '.sql', '.r', '.rb', '.go', '.rs',
                '.java', '.c', '.cpp', '.h', '.hpp', '.cs', '.php', '.pl', '.swift',
                '.kt', '.scala', '.clj', '.hs', '.elm', '.vue', '.svelte', '.jsx',
                '.tsx', '.dockerfile', '.gitignore', '.gitattributes', '.editorconfig'
            ],
            'logging': {
                'level': 'INFO',
                'log_to_file': True,
                'log_filename': 'codebase_generator.log'
            },
            'performance': {
                'encoding_detection_size_kb': 8,
                'text_detection_sample_kb': 1
            }
        }
        
        if not self.config_path.exists():
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return default_config
            
        if not HAS_YAML:
            logger.warning("PyYAML not available, using default configuration")
            return default_config
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
                
            # Merge with defaults (loaded config takes precedence)
            def merge_configs(default, loaded):
                if isinstance(default, dict) and isinstance(loaded, dict):
                    result = default.copy()
                    for key, value in loaded.items():
                        if key in result and isinstance(result[key], dict):
                            result[key] = merge_configs(result[key], value)
                        else:
                            result[key] = value
                    return result
                return loaded if loaded is not None else default
                
            return merge_configs(default_config, loaded_config)
            
        except Exception as e:
            logger.error(f"Error loading config file {self.config_path}: {e}")
            logger.info("Using default configuration")
            return default_config
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'file_processing.max_file_size_mb')."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
                
        return value

# Configure logging with config support
def setup_logging(config: ConfigLoader):
    """Setup logging based on configuration."""
    log_level = getattr(logging, config.get('logging.level', 'INFO').upper())
    log_filename = config.get('logging.log_filename', 'codebase_generator.log')
    log_to_file = config.get('logging.log_to_file', True)
    
    handlers = [logging.StreamHandler()]
    if log_to_file:
        handlers.append(logging.FileHandler(log_filename))
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True  # Override existing configuration
    )

class GitIgnoreParser:
    """Parse and match against .gitignore patterns."""
    
    def __init__(self, root_path: Path):
        self.root_path = Path(root_path).resolve()
        self.patterns = self._load_gitignore_patterns()
        
    def _load_gitignore_patterns(self) -> List[str]:
        """Load patterns from all .gitignore files in the directory tree."""
        patterns = []
        
        # Start from root and work down
        current = self.root_path
        while current.parent != current:  # Until we reach the filesystem root
            gitignore_path = current / '.gitignore'
            if gitignore_path.exists():
                try:
                    with open(gitignore_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            # Skip empty lines and comments
                            if line and not line.startswith('#'):
                                patterns.append(line)
                    logger.info(f"Loaded {len(patterns)} patterns from {gitignore_path}")
                    # Debug: print some patterns
                    logger.info(f"Sample patterns: {patterns[:5]}")
                    break  # Use the first .gitignore found when walking up
                except Exception as e:
                    logger.warning(f"Error reading .gitignore {gitignore_path}: {e}")
            current = current.parent
            
        # Add comprehensive patterns to avoid scanning problematic directories
        comprehensive_patterns = [
            # Version control
            '.git/',
            '.svn/',
            '.hg/',
            # Python cache and environments
            '__pycache__/',
            '*.pyc',
            '*.pyo',
            '*.pyd',
            '.Python',
            'build/',
            'develop-eggs/',
            'dist/',
            'downloads/',
            'eggs/',
            '.eggs/',
            'lib/',
            'lib64/',
            'parts/',
            'sdist/',
            'var/',
            'wheels/',
            '*.egg-info/',
            '.installed.cfg',
            '*.egg',
            'MANIFEST',
            # Virtual environments
            'venv/',
            'env/',
            'ENV/',
            '.venv/',
            '.env/',
            'virtualenv/',
            # Node.js
            'node_modules/',
            'npm-debug.log*',
            'yarn-debug.log*',
            'yarn-error.log*',
            # IDE and editor files
            '.vscode/',
            '.idea/',
            '*.swp',
            '*.swo',
            '*~',
            '.DS_Store',
            'Thumbs.db',
            # OS generated files
            'desktop.ini',
            # Large binary/data directories
            '.cache/',
            'cache/',
            'tmp/',
            'temp/',
            # Language specific caches
            '.mypy_cache/',
            '.pytest_cache/',
            '.coverage',
            'htmlcov/',
            # Log files
            '*.log',
            'logs/',
            # Backup files
            '*.bak',
            '*.backup',
            '*.old',
            # Package manager files
            'Pipfile.lock',
            'poetry.lock',
            'package-lock.json',
            'yarn.lock',
            # Output directories
            'output/'
        ]
        
        if not patterns:
            patterns = comprehensive_patterns
            logger.info(f"No .gitignore found, using {len(patterns)} comprehensive default patterns")
        else:
            # Combine existing patterns with comprehensive ones, avoiding duplicates
            all_patterns = list(patterns)
            for pattern in comprehensive_patterns:
                if pattern not in all_patterns:
                    all_patterns.append(pattern)
            patterns = all_patterns
            logger.info(f"Combined .gitignore with comprehensive patterns: {len(patterns)} total patterns")
            
        return patterns
        
    def should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored based on gitignore patterns."""
        try:
            # Get relative path from root
            rel_path = path.relative_to(self.root_path)
            rel_str = str(rel_path).replace('\\', '/')
            
            # Check if it's a directory
            is_dir = path.is_dir()
            if is_dir and not rel_str.endswith('/'):
                rel_str += '/'
                
            # Test against all patterns
            for pattern in self.patterns:
                # Handle negation patterns (starting with !)
                if pattern.startswith('!'):
                    continue  # Skip negation for simplicity
                    
                # Handle directory-only patterns (ending with /)
                if pattern.endswith('/'):
                    if not is_dir:
                        continue
                    pattern_to_check = pattern[:-1]  # Remove trailing slash
                    if fnmatch.fnmatch(path.name, pattern_to_check):
                        return True
                    # Also check if the pattern matches any parent directory name
                    for parent in path.parents:
                        if fnmatch.fnmatch(parent.name, pattern_to_check):
                            return True
                    continue
                    
                # Handle root-relative patterns (starting with /)
                if pattern.startswith('/'):
                    pattern = pattern[1:]
                    if fnmatch.fnmatch(rel_str, pattern):
                        return True
                else:
                    # Check if pattern matches any part of the path
                    path_parts = rel_str.split('/')
                    for i in range(len(path_parts)):
                        partial_path = '/'.join(path_parts[i:])
                        if fnmatch.fnmatch(partial_path, pattern):
                            return True
                        # Also check the pattern against just the filename
                        if fnmatch.fnmatch(path_parts[-1], pattern):
                            return True
                            
            return False
            
        except ValueError:
            # Path is not relative to root
            return True
        except Exception as e:
            logger.warning(f"Error checking ignore status for {path}: {e}")
            return False

class CodebaseGenerator:
    """Main codebase analysis and XML generation engine."""
    
    def __init__(self, root_path: str, config: ConfigLoader, 
                 max_file_size_mb: float = None, include_hidden: bool = None, 
                 additional_excludes: List[str] = None, ignore_gitignore: bool = None):
        self.root_path = Path(root_path).resolve()
        self.config = config
        
        # Use config values with command-line overrides
        self.max_file_size = (max_file_size_mb or config.get('file_processing.max_file_size_mb', 10.0)) * 1024 * 1024
        self.max_content_size = config.get('file_processing.max_content_size_kb', 100) * 1024
        self.include_hidden = include_hidden if include_hidden is not None else config.get('file_processing.include_hidden', False)
        self.ignore_gitignore = ignore_gitignore if ignore_gitignore is not None else config.get('file_processing.ignore_gitignore', False)
        
        # Get patterns from config
        self.critical_skip_dirs = set(config.get('critical_skip_dirs', []))
        self.skip_extensions = set(config.get('skip_extensions', []))
        self.text_extensions = set(config.get('text_extensions', []))
        
        if not self.ignore_gitignore:
            self.gitignore = GitIgnoreParser(self.root_path)
        else:
            self.gitignore = None
            logger.info("GitIgnore patterns disabled")
        
        # Additional patterns to exclude
        config_excludes = config.get('exclusions.additional_patterns', [])
        cmd_excludes = additional_excludes or []
        self.additional_excludes = config_excludes + cmd_excludes
        
        # Statistics
        self.stats = {
            'total_files': 0,
            'total_dirs': 0,
            'skipped_files': 0,
            'total_size': 0,
            'errors': 0
        }
        
    def _should_skip(self, path: Path) -> bool:
        """Determine if a path should be skipped."""
        # Always skip critical directories from config
        if path.name in self.critical_skip_dirs:
            logger.debug(f"Skipping critical directory: {path}")
            return True
            
        # Skip common cache and binary extensions from config
        if path.is_file():
            if path.suffix.lower() in self.skip_extensions:
                logger.debug(f"Skipping file with problematic extension: {path}")
                return True
        
        # Check if hidden and we're not including hidden files
        if not self.include_hidden and path.name.startswith('.'):
            return True
            
        # Check gitignore patterns only if we have a parser
        if self.gitignore and self.gitignore.should_ignore(path):
            return True
            
        # Check additional excludes
        for pattern in self.additional_excludes:
            if fnmatch.fnmatch(path.name, pattern):
                return True
                
        # Check file size if it's a file
        if path.is_file():
            try:
                if path.stat().st_size > self.max_file_size:
                    logger.warning(f"Skipping large file: {path} ({path.stat().st_size / 1024 / 1024:.1f}MB)")
                    return True
            except OSError:
                return True
                
        return False
        
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding safely."""
        try:
            sample_size = self.config.get('performance.encoding_detection_size_kb', 8) * 1024
            
            if HAS_CHARDET:
                with open(file_path, 'rb') as f:
                    raw_data = f.read(sample_size)
                    result = chardet.detect(raw_data)
                    encoding = result.get('encoding', 'utf-8')
                    return encoding if encoding else 'utf-8'
            else:
                # Fallback: try common encodings
                sample_size_text = min(sample_size, 1024)
                for encoding in ['utf-8', 'utf-16', 'latin-1', 'cp1252']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            f.read(sample_size_text)
                        return encoding
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                return 'utf-8'  # Final fallback
        except Exception:
            return 'utf-8'
            
    def _is_text_file(self, file_path: Path) -> bool:
        """Check if a file is likely a text file."""
        try:
            # Check MIME type
            mime_type, _ = mimetypes.guess_type(str(file_path))
            if mime_type:
                if mime_type.startswith('text/'):
                    return True
                if mime_type in ['application/json', 'application/xml', 'application/javascript']:
                    return True
                    
            # Check against configured text file extensions
            if file_path.suffix.lower() in self.text_extensions:
                return True
                
            # Try to read as text (small sample)
            try:
                sample_size = self.config.get('performance.text_detection_sample_kb', 1) * 1024
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.read(sample_size)
                return True
            except (UnicodeDecodeError, PermissionError):
                return False
                
        except Exception:
            return False
            
    def _read_file_content(self, file_path: Path) -> Optional[str]:
        """Safely read file content."""
        try:
            if not self._is_text_file(file_path):
                return f"<Binary file: {file_path.suffix or 'no extension'}>"
                
            encoding = self._detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
                
                # Clean content for XML safety
                if content:
                    # Remove null bytes and other problematic characters
                    content = content.replace('\x00', '').replace('\x01', '').replace('\x02', '')
                    # Remove or replace other control characters that might break XML
                    import re
                    content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
                    
                    # Limit content size to avoid memory issues
                    if len(content) > self.max_content_size:
                        content = content[:self.max_content_size] + "\n... [Content truncated - file too large]"
                        
                return content
                
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error reading {file_path}: {e}")
            return f"<Error reading file: {str(e)}>"
            
    def _scan_directory(self) -> Dict[str, Any]:
        """Scan directory and build tree structure."""
        logger.info(f"Scanning directory: {self.root_path}")
        
        def build_tree(current_path: Path) -> Dict[str, Any]:
            """Recursively build directory tree."""
            tree = {}
            
            try:
                items = sorted(current_path.iterdir(), 
                             key=lambda x: (not x.is_dir(), x.name.lower()))
                
                for item in items:
                    if self._should_skip(item):
                        continue
                        
                    try:
                        if item.is_dir():
                            self.stats['total_dirs'] += 1
                            tree[item.name] = {
                                'type': 'directory',
                                'path': str(item.relative_to(self.root_path)),
                                'children': build_tree(item)
                            }
                        elif item.is_file():
                            self.stats['total_files'] += 1
                            file_size = item.stat().st_size
                            self.stats['total_size'] += file_size
                            
                            tree[item.name] = {
                                'type': 'file',
                                'path': str(item.relative_to(self.root_path)),
                                'size': file_size,
                                'content': self._read_file_content(item)
                            }
                    except Exception as e:
                        self.stats['errors'] += 1
                        logger.error(f"Error processing {item}: {e}")
                        continue
                        
            except PermissionError:
                logger.warning(f"Permission denied: {current_path}")
            except Exception as e:
                logger.error(f"Error scanning {current_path}: {e}")
                
            return tree
            
        return build_tree(self.root_path)
        
    def generate_ascii_map(self, tree: Dict[str, Any]) -> List[str]:
        """Generate ASCII tree representation."""
        lines = []
        
        def format_tree(tree_dict: Dict[str, Any], prefix: str = "", is_last: bool = True):
            """Recursively format tree as ASCII."""
            items = sorted(tree_dict.items())
            
            for i, (name, info) in enumerate(items):
                is_last_item = i == len(items) - 1
                connector = "└── " if is_last_item else "├── "
                
                if info['type'] == 'directory':
                    lines.append(f"{prefix}{connector}📁 {name}/")
                    extension = "    " if is_last_item else "│   "
                    format_tree(info['children'], prefix + extension, is_last_item)
                else:
                    size_str = f" ({info['size']} bytes)" if info['size'] < 1024 else f" ({info['size']/1024:.1f}KB)"
                    lines.append(f"{prefix}{connector}📄 {name}{size_str}")
                    
        # Add header
        lines.append(f"📁 {self.root_path.name}/")
        format_tree(tree)
        
        return lines
        
    def generate_xml(self, tree: Dict[str, Any]) -> ET.Element:
        """Generate XML representation of codebase."""
        
        def build_xml_tree(tree_dict: Dict[str, Any], parent: ET.Element):
            """Recursively build XML tree."""
            for name, info in sorted(tree_dict.items()):
                if info['type'] == 'directory':
                    dir_elem = ET.SubElement(parent, 'directory')
                    dir_elem.set('name', name)
                    dir_elem.set('path', info['path'])
                    build_xml_tree(info['children'], dir_elem)
                else:
                    file_elem = ET.SubElement(parent, 'file')
                    file_elem.set('name', name)
                    file_elem.set('path', info['path'])
                    file_elem.set('size', str(info['size']))
                    
                    # Add content in CDATA section
                    if info['content']:
                        content_elem = ET.SubElement(file_elem, 'content')
                        content_elem.text = info['content']
                        
        # Create root element
        root = ET.Element('codebase')
        root.set('name', self.root_path.name)
        root.set('path', str(self.root_path))
        root.set('generated', datetime.now().isoformat())
        root.set('total_files', str(self.stats['total_files']))
        root.set('total_directories', str(self.stats['total_dirs']))
        root.set('total_size', str(self.stats['total_size']))
        
        build_xml_tree(tree, root)
        return root
        
    def generate_report(self, tree: Dict[str, Any] = None) -> str:
        """Generate comprehensive summary report with metadata."""
        gitignore_count = len(self.gitignore.patterns) if self.gitignore else 0
        
        # Calculate file type statistics if tree is provided
        file_type_stats = {}
        total_content_size = 0
        
        if tree:
            def analyze_tree(tree_dict):
                for name, info in tree_dict.items():
                    if info['type'] == 'directory':
                        analyze_tree(info['children'])
                    else:
                        # Count by file extension
                        ext = Path(name).suffix.lower() or 'no-extension'
                        file_type_stats[ext] = file_type_stats.get(ext, 0) + 1
                        
                        # Track content size
                        nonlocal total_content_size
                        if info.get('content') and not info['content'].startswith('<Binary file'):
                            total_content_size += len(info['content'])
            
            analyze_tree(tree)
        
        # Sort file types by count
        sorted_file_types = sorted(file_type_stats.items(), key=lambda x: x[1], reverse=True)
        file_types_display = '\n'.join([f"  {ext}: {count} files" for ext, count in sorted_file_types[:10]])
        if len(sorted_file_types) > 10:
            file_types_display += f"\n  ... and {len(sorted_file_types) - 10} more types"
        
        if not file_types_display:
            file_types_display = "  (Analysis requires tree data)"
        
        return f"""
Codebase Analysis Report
========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Root Path: {self.root_path}
Script Version: Universal Codebase XML Generator v1.0

File System Statistics:
-----------------------
Total Files: {self.stats['total_files']:,}
Total Directories: {self.stats['total_dirs']:,}
Total File Size: {self.stats['total_size'] / 1024 / 1024:.2f} MB
Total Content Size: {total_content_size / 1024 / 1024:.2f} MB
Skipped Files: {self.stats['skipped_files']:,}
Processing Errors: {self.stats['errors']:,}

File Type Distribution:
----------------------
{file_types_display}

Configuration Settings:
----------------------
Max File Size Limit: {self.max_file_size / 1024 / 1024:.1f} MB
Max Content Size Limit: {self.max_content_size / 1024:.1f} KB
Include Hidden Files: {self.include_hidden}
GitIgnore Processing: {'Disabled' if self.ignore_gitignore else 'Enabled'}
GitIgnore Patterns Loaded: {gitignore_count}
Additional Exclude Patterns: {', '.join(self.additional_excludes) if self.additional_excludes else 'None'}
Config File Used: {self.config.config_path}

Processing Metadata:
-------------------
Python Version: {sys.version.split()[0]}
Script Location: {Path(__file__).resolve()}
Working Directory: {Path.cwd()}
Character Encoding Detection: {'chardet library' if HAS_CHARDET else 'built-in fallback'}
YAML Support: {'Available' if HAS_YAML else 'Not available (using defaults)'}

Critical Directories Skipped:
-----------------------------
{', '.join(sorted(self.critical_skip_dirs))}

Performance Notes:
-----------------
- Files larger than {self.max_file_size / 1024 / 1024:.1f}MB are excluded from content analysis
- File content is truncated at {self.max_content_size / 1024:.1f}KB per file to prevent memory issues
- Control characters are stripped from file content for XML safety
- Binary files are identified and marked without content extraction
- Configuration loaded from: {self.config.config_path}
"""

def save_outputs(generator: CodebaseGenerator, tree: Dict[str, Any], output_dir: Path):
    """Save generated XML output to file."""
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir.resolve()}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"codebase_{generator.root_path.name}_{timestamp}"
    print(f"📝 Base filename: {base_name}")
    
    # Create comprehensive XML output file
    xml_file = output_dir / f"{base_name}_complete.xml"
    
    try:
        # Create root XML element with comprehensive structure
        root = ET.Element("codebase_report")
        
        # Add generation metadata to root
        generation_time = datetime.now()
        root.set("generated", generation_time.isoformat())
        root.set("generator_version", "v2.0")
        root.set("python_version", sys.version.split()[0])
        root.set("platform", sys.platform)
        root.set("total_files", str(generator.stats['total_files']))
        root.set("total_directories", str(generator.stats['total_dirs']))
        root.set("total_size", str(generator.stats['total_size']))
        
        # Create header section
        header = ET.SubElement(root, "header")
        title = ET.SubElement(header, "title")
        title.text = "UNIVERSAL CODEBASE GENERATOR - COMPLETE ANALYSIS REPORT"
        
        generation_info = ET.SubElement(header, "generation_info")
        gen_time = ET.SubElement(generation_info, "generated_time")
        gen_time.text = generation_time.strftime("%Y-%m-%d %H:%M:%S")
        gen_path = ET.SubElement(generation_info, "root_path")
        gen_path.text = str(generator.root_path)
        gen_output = ET.SubElement(generation_info, "output_file")
        gen_output.text = xml_file.name
        gen_version = ET.SubElement(generation_info, "generator_version")
        gen_version.text = "v2.0"
        gen_python = ET.SubElement(generation_info, "python_version")
        gen_python.text = sys.version.split()[0]
        gen_platform = ET.SubElement(generation_info, "platform")
        gen_platform.text = sys.platform
        
        # Create table of contents section
        toc = ET.SubElement(root, "table_of_contents")
        toc_desc = ET.SubElement(toc, "description")
        toc_desc.text = "Comprehensive listing of all sections in this report"
        
        toc_sections = ET.SubElement(toc, "sections")
        section1 = ET.SubElement(toc_sections, "section")
        section1.set("number", "1")
        section1.set("name", "ASCII_CODEBASE_MAP")
        section1.text = "Visual directory tree structure with file sizes"
        
        section2 = ET.SubElement(toc_sections, "section")
        section2.set("number", "2")
        section2.set("name", "XML_STRUCTURE")
        section2.text = "Machine-readable hierarchical data with file content"
        
        section3 = ET.SubElement(toc_sections, "section")
        section3.set("number", "3")
        section3.set("name", "ANALYSIS_REPORT")
        section3.text = "Statistics, metadata, and configuration details"
        
        # Create ASCII codebase map section
        ascii_section = ET.SubElement(root, "ascii_codebase_map")
        ascii_desc = ET.SubElement(ascii_section, "description")
        ascii_desc.text = "Visual tree representation of the directory structure. 📁 = Directory, 📄 = File, with file sizes shown in parentheses."
        
        ascii_content = ET.SubElement(ascii_section, "content")
        ascii_map = generator.generate_ascii_map(tree)
        ascii_content.text = '\n'.join(ascii_map)
        print(f"✅ ASCII map added to XML structure")
        
        # Create hierarchical XML structure section (the actual codebase data)
        xml_structure_section = ET.SubElement(root, "xml_structure")
        xml_desc = ET.SubElement(xml_structure_section, "description")
        xml_desc.text = "Complete hierarchical XML representation including file paths, sizes, and content (for text files under size limit). Binary files and oversized files are marked but content is excluded."
        
        # Generate the actual codebase XML and embed it
        codebase_xml = generator.generate_xml(tree)
        xml_structure_section.append(codebase_xml)
        print(f"✅ XML structure added to comprehensive XML")
        
        # Create analysis report section
        analysis_section = ET.SubElement(root, "analysis_report")
        analysis_desc = ET.SubElement(analysis_section, "description")
        analysis_desc.text = "Detailed statistics, configuration, and metadata about the analysis"
        
        # Add file system statistics
        fs_stats = ET.SubElement(analysis_section, "file_system_statistics")
        fs_stats.set("total_files", str(generator.stats['total_files']))
        fs_stats.set("total_directories", str(generator.stats['total_dirs']))
        fs_stats.set("total_size_bytes", str(generator.stats['total_size']))
        fs_stats.set("total_size_mb", f"{generator.stats['total_size'] / 1024 / 1024:.2f}")
        fs_stats.set("skipped_files", str(generator.stats['skipped_files']))
        fs_stats.set("processing_errors", str(generator.stats['errors']))
        
        # Calculate and add file type distribution
        file_type_stats = {}
        total_content_size = 0
        
        def analyze_tree_for_stats(tree_dict):
            nonlocal total_content_size
            for name, info in tree_dict.items():
                if info['type'] == 'directory':
                    analyze_tree_for_stats(info['children'])
                else:
                    # Count by file extension
                    ext = Path(name).suffix.lower() or 'no-extension'
                    file_type_stats[ext] = file_type_stats.get(ext, 0) + 1
                    
                    # Track content size
                    if info.get('content') and not info['content'].startswith('<Binary file'):
                        total_content_size += len(info['content'])
        
        analyze_tree_for_stats(tree)
        
        # Add file type distribution
        file_types_section = ET.SubElement(analysis_section, "file_type_distribution")
        file_types_section.set("total_content_size_bytes", str(total_content_size))
        file_types_section.set("total_content_size_mb", f"{total_content_size / 1024 / 1024:.2f}")
        
        for ext, count in sorted(file_type_stats.items(), key=lambda x: x[1], reverse=True):
            type_elem = ET.SubElement(file_types_section, "file_type")
            type_elem.set("extension", ext)
            type_elem.set("count", str(count))
            type_elem.set("percentage", f"{(count / generator.stats['total_files']) * 100:.1f}")
        
        # Add configuration settings
        config_section = ET.SubElement(analysis_section, "configuration_settings")
        config_section.set("max_file_size_mb", f"{generator.max_file_size / 1024 / 1024:.1f}")
        config_section.set("max_content_size_kb", f"{generator.max_content_size / 1024:.1f}")
        config_section.set("include_hidden", str(generator.include_hidden))
        config_section.set("gitignore_processing", "Disabled" if generator.ignore_gitignore else "Enabled")
        config_section.set("gitignore_patterns_loaded", str(len(generator.gitignore.patterns) if generator.gitignore else 0))
        config_section.set("config_file_used", str(generator.config.config_path))
        
        # Add additional excludes if any
        if generator.additional_excludes:
            excludes_elem = ET.SubElement(config_section, "additional_excludes")
            for exclude in generator.additional_excludes:
                exclude_elem = ET.SubElement(excludes_elem, "pattern")
                exclude_elem.text = exclude
        
        # Add processing metadata
        metadata_section = ET.SubElement(analysis_section, "processing_metadata")
        metadata_section.set("python_version", sys.version.split()[0])
        metadata_section.set("script_location", str(Path(__file__).resolve()))
        metadata_section.set("working_directory", str(Path.cwd()))
        metadata_section.set("chardet_available", str(HAS_CHARDET))
        metadata_section.set("yaml_support", str(HAS_YAML))
        
        # Add critical directories that were skipped
        skip_dirs_section = ET.SubElement(analysis_section, "skipped_directories")
        for skip_dir in sorted(generator.critical_skip_dirs):
            skip_elem = ET.SubElement(skip_dirs_section, "directory")
            skip_elem.text = skip_dir
        
        # Add file generation summary
        summary_section = ET.SubElement(analysis_section, "generation_summary")
        summary_section.set("output_file", str(xml_file.resolve()))
        summary_section.set("sections_included", "ASCII Map, XML Structure, Analysis Report")
        
        # Write XML to file with pretty formatting
        from xml.dom import minidom
        xml_string = ET.tostring(root, encoding='unicode')
        dom = minidom.parseString(xml_string)
        pretty_xml = dom.toprettyxml(indent="  ")
        
        # Remove empty lines that toprettyxml adds
        pretty_xml = '\n'.join([line for line in pretty_xml.split('\n') if line.strip()])
        
        with open(xml_file, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
        
        # Update file size in generation summary after file is written
        file_size = xml_file.stat().st_size
        summary_section.set("file_size_bytes", str(file_size))
        summary_section.set("file_size_kb", f"{file_size / 1024:.1f}")
        
        logger.info(f"XML output saved to: {xml_file}")
        print(f"✅ Comprehensive XML file created successfully: {xml_file}")
        print(f"📊 File size: {file_size / 1024:.1f} KB")
        
    except Exception as e:
        logger.error(f"Error saving XML file: {e}")
        print(f"❌ Error creating XML file: {e}")
        
    # List all created files
    print(f"\n📋 Files created in {output_dir}:")
    try:
        for file_path in output_dir.iterdir():
            if file_path.is_file():
                size = file_path.stat().st_size
                print(f"   📄 {file_path.name} ({size:,} bytes)")
    except Exception as e:
        print(f"   ❌ Error listing files: {e}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Universal Codebase XML Generator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('path', nargs='?', default='.', 
                       help='Path to scan (default: current directory)')
    parser.add_argument('--config', '-c', default='map_config.yaml',
                       help='Configuration file path (default: map_config.yaml)')
    parser.add_argument('--output-dir', '-o', 
                       help='Output directory (overrides config)')
    parser.add_argument('--max-file-size', '-s', type=float,
                       help='Maximum file size in MB (overrides config)')
    parser.add_argument('--include-hidden', '-H', action='store_true',
                       help='Include hidden files and directories (overrides config)')
    parser.add_argument('--exclude', '-e', 
                       help='Additional patterns to exclude (comma-separated, adds to config)')
    parser.add_argument('--ignore-gitignore', action='store_true',
                       help='Ignore .gitignore patterns completely (overrides config)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigLoader(args.config)
    
    # Setup logging based on config and verbose flag
    if args.verbose:
        config.config['logging']['level'] = 'DEBUG'
    setup_logging(config)
    
    # Get logger after setup
    logger = logging.getLogger(__name__)
    
    # Parse additional excludes
    additional_excludes = []
    if args.exclude:
        additional_excludes = [p.strip() for p in args.exclude.split(',')]
        
    try:
        # Initialize generator with config
        generator = CodebaseGenerator(
            root_path=args.path,
            config=config,
            max_file_size_mb=args.max_file_size,
            include_hidden=args.include_hidden,
            additional_excludes=additional_excludes,
            ignore_gitignore=args.ignore_gitignore
        )
        
        logger.info("Starting codebase analysis...")
        
        # Scan directory
        tree = generator._scan_directory()
        
        if not tree:
            logger.warning("No files found or all files were excluded")
            return 1
            
        # Determine output settings
        output_dir_path = args.output_dir or config.get('output.output_dir', './output')
        
        # Save outputs (always comprehensive XML)
        output_dir = Path(output_dir_path)
        save_outputs(generator, tree, output_dir)
        
        # Print ASCII map to console for quick preview
        print("\n" + "="*50)
        print("ASCII CODEBASE MAP PREVIEW")
        print("="*50)
        ascii_map = generator.generate_ascii_map(tree)
        print('\n'.join(ascii_map[:20]))  # Show first 20 lines
        if len(ascii_map) > 20:
            print(f"... and {len(ascii_map) - 20} more lines (see XML file for complete output)")
        
        print(f"\n✅ Complete XML report generated successfully!")
        print(f"📁 Output directory: {output_dir.resolve()}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
