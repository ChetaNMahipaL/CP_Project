#!/usr/bin/env python3
"""Debug script to check pickle format"""

import pickle
import sys

# Check one of the pickles to understand the format
template_dir = '../EMNLP2018/templates'
test_name = 'simple_agrmt'

pickle_file = f'{template_dir}/{test_name}.pickle'

try:
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Loaded {pickle_file}")
    print(f"Type: {type(data)}")
    print(f"Keys: {data.keys() if isinstance(data, dict) else 'N/A'}")
    
    # Get first item
    if isinstance(data, dict):
        first_key = list(data.keys())[0]
        first_value = data[first_key]
        print(f"\nFirst key: {first_key}")
        print(f"Value type: {type(first_value)}")
        print(f"Value length: {len(first_value)}")
        
        if len(first_value) > 0:
            print(f"\nFirst item type: {type(first_value[0])}")
            print(f"First item: {repr(first_value[0])}")
            
            if isinstance(first_value[0], (list, tuple)) and len(first_value[0]) > 0:
                print(f"Sub-item type: {type(first_value[0][0])}")
                print(f"Sub-item: {repr(first_value[0][0])}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
