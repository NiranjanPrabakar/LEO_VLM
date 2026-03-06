import json
from pathlib import Path

class SymbolicStateEncoder:
    def __init__(self):
        pass
    
    def encode_scene(self, frame_data):
        """
        Convert detection + accessibility into symbolic state
        
        Input: frame_data from accessibility_dataset.json
        Output: Structured text description for VLM
        """
        detections = frame_data.get('detections', {})
        accessibility = frame_data.get('accessibility', {})
        
        # Build object list
        objects_present = []
        
        # Count beds
        if detections.get('bed'):
            objects_present.append('1 bed')
        
        # Count bedding items (duvets, sheets)
        bedding_count = len(detections.get('bedding', []))
        if bedding_count > 0:
            objects_present.append(f"{bedding_count} bedding item(s)")
        
        # Count pillows
        pillow_count = len(detections.get('pillows', []))
        if pillow_count > 0:
            objects_present.append(f"{pillow_count} pillow(s)")
        
        # Count obstacles
        obstacle_count = len(detections.get('obstacles', []))
        if obstacle_count > 0:
            objects_present.append(f"{obstacle_count} obstacle(s)")
        
        # Build symbolic state
        symbolic_state = {
            'frame_id': frame_data.get('frame_id'),
            'objects': objects_present,
            'accessibility': accessibility,
            'scene_configuration': self._describe_configuration(accessibility)
        }
        
        return symbolic_state
    
    def _describe_configuration(self, accessibility):
        """Generate natural language description of accessibility"""
        if not accessibility:
            return "Unable to determine bed accessibility"
        
        free_sides = [side for side, status in accessibility.items() if status == 'free']
        blocked_sides = [side for side, status in accessibility.items() if status == 'blocked']
        partial_sides = [side for side, status in accessibility.items() if status == 'partially_blocked']
        
        # Generate description
        if len(free_sides) == 0:
            return "Bed has no fully accessible sides (all blocked or partially blocked)"
        elif len(free_sides) == 4:
            return "Bed is fully accessible from all four sides"
        elif len(free_sides) >= 2:
            return f"Bed is accessible from {len(free_sides)} sides: {', '.join(free_sides)}"
        else:
            return f"Bed has limited access from only the {free_sides[0]} side"
    
    def to_text_prompt(self, symbolic_state):
        """Convert symbolic state to VLM prompt format"""
        
        frame_id = symbolic_state.get('frame_id', 'unknown')
        objects = symbolic_state.get('objects', [])
        accessibility = symbolic_state.get('accessibility', {})
        config = symbolic_state.get('scene_configuration', '')
        
        # Build structured prompt
        prompt = f"""=== SCENE ANALYSIS: Frame {frame_id} ===

DETECTED OBJECTS:
{', '.join(objects) if objects else 'None detected'}

BED ACCESSIBILITY CONSTRAINTS:
- Left side: {accessibility.get('left', 'unknown')}
- Right side: {accessibility.get('right', 'unknown')}  
- Head side: {accessibility.get('head', 'unknown')}
- Foot side: {accessibility.get('foot', 'unknown')}

CONFIGURATION SUMMARY:
{config}

TASK: Generate a step-by-step plan to make this bed, respecting all accessibility constraints.
"""
        return prompt


def test_encoder():
    """Test the encoder on your actual dataset"""
    
    # Load your processed dataset
    dataset_path = Path('results/accessibility_dataset.json')
    
    if not dataset_path.exists():
        print("❌ Error: accessibility_dataset.json not found!")
        return
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print(f"✓ Loaded dataset with {len(dataset)} frames\n")
    
    # Initialize encoder
    encoder = SymbolicStateEncoder()
    
    # Test on first 5 frames
    output_dir = Path('results/symbolic_states')
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("GENERATING SYMBOLIC STATES FOR SAMPLE FRAMES")
    print("="*60 + "\n")
    
    for i, frame_data in enumerate(dataset[:5]):
        print(f"\n--- Frame {i+1}/5 ---")
        
        # Encode the scene
        symbolic_state = encoder.encode_scene(frame_data)
        
        # Convert to text prompt
        text_prompt = encoder.to_text_prompt(symbolic_state)
        
        # Display
        print(text_prompt)
        
        # Save to file
        frame_id = frame_data.get('frame_id', f'frame_{i}')
        output_file = output_dir / f'symbolic_state_{frame_id}.txt'
        
        with open(output_file, 'w') as f:
            f.write(text_prompt)
        
        print(f"✓ Saved to: {output_file}")
    
    print("\n" + "="*60)
    print(f"✓ Generated 5 sample symbolic states in {output_dir}")
    print("="*60)


if __name__ == "__main__":
    test_encoder()
