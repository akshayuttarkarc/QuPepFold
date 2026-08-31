"""Provenance tracking for reproducibility.

Captures git hash, package versions, timestamps, and session info.
"""

import subprocess
from datetime import datetime
from typing import Dict, Any, Optional
import platform


def get_git_hash(repo_path: Optional[str] = None) -> Optional[str]:
    """Get current git commit hash.
    
    Args:
        repo_path: Path to git repository (default: current directory).
        
    Returns:
        Git commit hash or None if not available.
    """
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        if repo_path:
            cmd = ["git", "-C", repo_path, "rev-parse", "HEAD"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_qiskit_versions() -> Dict[str, str]:
    """Get versions of installed Qiskit packages."""
    versions = {}
    
    packages = [
        "qiskit",
        "qiskit-aer",
        "qiskit-ibm-runtime",
    ]
    
    for pkg in packages:
        try:
            # Handle different package naming conventions
            import_name = pkg.replace("-", "_")
            module = __import__(import_name)
            versions[pkg] = getattr(module, "__version__", "unknown")
        except ImportError:
            versions[pkg] = "not installed"
    
    return versions


def get_provenance(
    backend_name: Optional[str] = None,
    session_id: Optional[str] = None,
    job_ids: Optional[list] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Build complete provenance dictionary.
    
    Args:
        backend_name: Name of quantum backend used.
        session_id: IBM Runtime session ID (if applicable).
        job_ids: List of job IDs submitted.
        seed: Random seed used.
        
    Returns:
        Provenance dictionary.
    """
    provenance = {
        "timestamp": datetime.now().isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "python_version": platform.python_version(),
        },
        "git_hash": get_git_hash(),
        "qiskit_versions": get_qiskit_versions(),
    }
    
    if backend_name:
        provenance["backend"] = backend_name
    
    if session_id:
        provenance["session_id"] = session_id
    
    if job_ids:
        provenance["job_ids"] = job_ids
    
    if seed is not None:
        provenance["seed"] = seed
    
    return provenance


def format_provenance(provenance: Dict) -> str:
    """Format provenance dict as human-readable string.
    
    Args:
        provenance: Provenance dictionary.
        
    Returns:
        Formatted string.
    """
    lines = ["=== Provenance ==="]
    
    lines.append(f"Timestamp: {provenance.get('timestamp', 'N/A')}")
    
    if "git_hash" in provenance:
        lines.append(f"Git Hash: {provenance['git_hash'][:12]}...")
    
    if "qiskit_versions" in provenance:
        lines.append("Qiskit Versions:")
        for pkg, ver in provenance["qiskit_versions"].items():
            lines.append(f"  {pkg}: {ver}")
    
    if "backend" in provenance:
        lines.append(f"Backend: {provenance['backend']}")
    
    if "seed" in provenance:
        lines.append(f"Seed: {provenance['seed']}")
    
    return "\n".join(lines)
