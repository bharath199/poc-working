import json
import os
from pathlib import Path
from collections import Counter

def check_json_labels(json_path):
    """
    Check if a JSON file has exactly two labels: one 'good' and one 'poor_solder'
    Returns (True, labels) if condition is met, (False, labels) otherwise
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if 'shapes' not in data:
            return False, []
        
        labels = [shape.get('label') for shape in data['shapes'] if 'label' in shape]
        
        # Check if we have exactly 2 labels, one 'good' and one 'poor_solder'
        label_counts = Counter(labels)
        
        if len(labels) == 2 and label_counts.get('good') == 1 and label_counts.get('poor_solder') == 1:
            return True, labels
        
        return False, labels
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return False, []

def main():
    base_path = Path(r'c:\Users\bhara\Desktop\poc\data_local\SolDef_AI\Filtered')
    
    # Track results
    good_jsons = []
    removed_jsons = []
    stats = {'total': 0, 'kept': 0, 'removed': 0}
    
    # Iterate through c1-c4 and good/poor subdirectories
    for c_folder in ['c1', 'c2', 'c3', 'c4']:
        c_path = base_path / c_folder
        
        for category in ['good', 'poor']:
            category_path = c_path / category
            
            if not category_path.exists():
                continue
            
            json_files = list(category_path.glob('*.json'))
            
            for json_file in json_files:
                stats['total'] += 1
                is_valid, labels = check_json_labels(json_file)
                
                if is_valid:
                    stats['kept'] += 1
                    good_jsons.append(str(json_file))
                    print(f"✓ KEEP: {json_file.name} (labels: {labels})")
                else:
                    stats['removed'] += 1
                    removed_jsons.append(str(json_file))
                    print(f"✗ REMOVE: {json_file.name} (labels: {labels})")
    
    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Total JSON files: {stats['total']}")
    print(f"  Kept (exactly 'good' + 'poor_solder'): {stats['kept']}")
    print(f"  Removed (other combinations): {stats['removed']}")
    print(f"{'='*80}")
    
    # Save results to files
    with open(r'c:\Users\bhara\Desktop\poc\good_jsons.txt', 'w') as f:
        for path in good_jsons:
            f.write(path + '\n')
    
    with open(r'c:\Users\bhara\Desktop\poc\removed_jsons.txt', 'w') as f:
        for path in removed_jsons:
            f.write(path + '\n')
    
    print(f"\nResults saved to:")
    print(f"  - good_jsons.txt ({len(good_jsons)} files)")
    print(f"  - removed_jsons.txt ({len(removed_jsons)} files)")

if __name__ == '__main__':
    main()
