CLASS_NAMES = {
    0: 'Bed',
    1: 'Bed_Cover',
    2: 'Duvet',
    3: 'Duvet_Cover',
    4: 'Mattress_protector',
    5: 'obstacles',
    6: 'Pillow',
    7: 'Pillow_cover',
    8: 'Wall'
}

BED_CLASSES = ['Bed']  
BEDDING_CLASSES = ['Bed_Cover', 'Duvet', 'Duvet_Cover', 'Mattress_protector']
PILLOW_CLASSES = ['Pillow', 'Pillow_cover']
OBSTACLE_CLASSES = ['obstacles', 'Wall']

def get_class_category(class_name):
    """Categorize detected objects"""
    if class_name in BED_CLASSES:
        return 'bed'
    elif class_name in BEDDING_CLASSES:
        return 'bedding'
    elif class_name in PILLOW_CLASSES:
        return 'pillow'
    elif class_name in OBSTACLE_CLASSES:
        return 'obstacle'
    return 'unknown'
