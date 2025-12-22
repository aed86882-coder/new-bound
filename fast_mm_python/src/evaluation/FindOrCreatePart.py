"""
Find a part in the given level that matches "shape" and "identifier".
If not found, create one and return.
Currently, "identifier" consists of the ID of the parent part and the hashing region (1 to 3).
"""

import numpy as np
from .FindPartByIdentifier import FindPartByIdentifier


def CreatePartInstance(level, shape):
    """
    Create the appropriate PartInfo instance based on level and shape.
    
    Args:
        level: Level of the part
        shape: Shape tuple (i, j, k)
    
    Returns:
        Part instance (PartInfo, PartInfoLv2, or PartInfoZero)
    """
    from .PartInfoLv2 import PartInfoLv2
    from .PartInfoZero import PartInfoZero
    from .PartInfo import PartInfo
    
    if level == 2:
        return PartInfoLv2()
    elif min(shape) == 0:
        return PartInfoZero()
    else:
        return PartInfo()


def FindOrCreatePart(parts, level, shape, identifier):
    """
    Find a part in the given level that matches "shape" and "identifier".
    If not found, create one and return.
    
    Args:
        parts: Global parts dictionary {level: {id: part}}
        level: Level to search/create in
        shape: Shape tuple (i, j, k)
        identifier: Identifier tuple [parent_id, region]
    
    Returns:
        (idx, ptr, is_new): Part ID, Part instance, whether it's newly created
    """
    shape_tuple = tuple(shape) if not isinstance(shape, tuple) else shape
    
    # Try to find existing part
    idx = FindPartByIdentifier(parts, level, shape_tuple, identifier)
    
    if idx != -1:
        # Found existing part
        ptr = parts[level][idx]
        is_new = False
        return idx, ptr, is_new
    
    # Create new part
    if level not in parts:
        parts[level] = {}
    
    # Assign new ID
    if parts[level]:
        idx = max(parts[level].keys()) + 1
    else:
        idx = 0
    
    # Create appropriate instance
    ptr = CreatePartInstance(level, shape_tuple)
    parts[level][idx] = ptr
    is_new = True
    
    return idx, ptr, is_new

