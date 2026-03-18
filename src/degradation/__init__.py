"""
AFS-VFM Degradation Engine
===========================

Public API
----------
- ``DegradationPipeline``           — class for fine-grained control
- ``generate_degradation_sequence`` — one-call convenience function
"""

from .degradation import DegradationPipeline, generate_degradation_sequence

__all__ = [
    "DegradationPipeline",
    "generate_degradation_sequence",
]
