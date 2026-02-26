import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def generate_methodology_diagram():
    # Create directory if it doesn't exist
    os.makedirs('assets/reports', exist_ok=True)
    
    # Setup Figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    
    # Define Boxes (simplified steps)
    steps = [
        {"x": 0.1, "y": 0.5, "w": 0.15, "h": 0.2, "bg": "#bdc3c7", "text": "Step 1\nInput\nVideo"},
        {"x": 0.3, "y": 0.5, "w": 0.15, "h": 0.2, "bg": "#3498db", "text": "Step 2\nYOLO AI\nDetection"},
        {"x": 0.5, "y": 0.5, "w": 0.15, "h": 0.2, "bg": "#9b59b6", "text": "Step 3\nPothole\nProcessing"},
        {"x": 0.7, "y": 0.5, "w": 0.15, "h": 0.2, "bg": "#e74c3c", "text": "Step 4\nAlert & \nWarning"}
    ]
    
    # Draw Boxes and Arrows
    for i, step in enumerate(steps):
        # Draw Box
        rect = patches.FancyBboxPatch(
            (step["x"], step["y"]), step["w"], step["h"],
            boxstyle="round,pad=0.05",
            linewidth=2, edgecolor="#2c3e50", facecolor=step["bg"]
        )
        ax.add_patch(rect)
        
        # Add Text
        ax.text(
            step["x"] + step["w"]/2, step["y"] + step["h"]/2, 
            step["text"], 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white'
        )
        
        # Draw Arrow to next step (except last one)
        if i < len(steps) - 1:
            ax.annotate("", 
                        xy=(steps[i+1]["x"], step["y"] + step["h"]/2), 
                        xytext=(step["x"] + step["w"], step["y"] + step["h"]/2),
                        arrowprops=dict(arrowstyle="->", lw=3, color="#2c3e50"))

    plt.title("Rakshak AI: Simplified Methodology Flowchart", fontsize=16)
    plt.tight_layout()
    plt.savefig('assets/reports/methodology_flowchart.png')
    print("Generated: assets/reports/methodology_flowchart.png")

if __name__ == "__main__":
    generate_methodology_diagram()
