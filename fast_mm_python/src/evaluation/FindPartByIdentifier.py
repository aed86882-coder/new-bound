"""
Find a part in the given level that matches shape and identifier.
If not found, return -1; otherwise, return the ID of the part.
"""

import numpy as np


def FindPartByIdentifier(parts, level, shape, identifier):
    """
    Find a part in the given level that matches "shape" and "identifier".
    
    Args:
        parts: Global parts dictionary {level: {id: part}}
        level: Level to search in
        shape: Shape tuple (i, j, k)
        identifier: Identifier tuple [parent_id, region]
    
    Returns:
        id: Part ID if found, -1 otherwise
    """
    if level not in parts or not parts[level]:
        return -1
    
    shape_tuple = tuple(shape) if not isinstance(shape, tuple) else shape
    identifier_tuple = tuple(identifier) if not isinstance(identifier, tuple) else identifier
    
    for part_id, part in parts[level].items():
        part_shape = tuple(part.shape) if not isinstance(part.shape, tuple) else part.shape
        part_identifier = tuple(part.identifier) if not isinstance(part.identifier, tuple) else part.identifier
        
        if part_shape == shape_tuple and part_identifier == identifier_tuple:
            return part_id
    
    return -1

