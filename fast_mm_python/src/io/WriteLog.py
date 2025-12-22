"""
Logging utilities for experiment tracking.
"""

import os
from datetime import datetime


def WriteLog(message, expinfo=None):
    """
    Write a log message to expname/log.txt and print to screen.
    
    Args:
        message: Message to log
        expinfo: Experiment info dict (will use global if None)
    """
    if expinfo is None:
        from ..control import get_expinfo
        expinfo = get_expinfo()
    
    exp_name = expinfo.get('exp_name', 'default_exp')
    log_path = os.path.join(exp_name, 'log.txt')
    
    # Ensure directory exists
    if not os.path.exists(exp_name):
        os.makedirs(exp_name)
    
    # Format timestamp
    timestamp = datetime.now().strftime('%m-%d %H:%M:%S')
    log_line = f'[{timestamp}] {message}'
    
    # Write to file
    with open(log_path, 'a') as f:
        f.write(log_line + '\n')
    
    # Print to screen
    print(log_line)

