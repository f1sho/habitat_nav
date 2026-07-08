import os
import time
from ultralytics import YOLO

def evaluate_perception_model(model_path, data_yaml, output_dir="evaluate/results"):
    # Create the output directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading model from {model_path} for true evaluation...")
    model = YOLO(model_path)

    print(f"Starting validation on dataset {data_yaml}...")
    # Execute the built in validation engine
    # split=val specifies running on the validation set
    metrics = model.val(data=data_yaml, split='val', plots=False)

    # Extract true accuracy metrics
    box_map50 = metrics.box.map50
    box_map50_95 = metrics.box.map
    mask_map50 = metrics.seg.map50
    mask_map50_95 = metrics.seg.map
    inference_speed = metrics.speed['inference']

    # Generate a formatted text report
    report_lines = [
        "=== True Perception Model Evaluation ===",
        f"Model: {model_path}",
        f"Dataset: {data_yaml}",
        "----------------------------------------",
        "[Bounding Box Metrics]",
        f"Box mAP@50: {box_map50:.4f}",
        f"Box mAP@50-95: {box_map50_95:.4f}",
        "----------------------------------------",
        "[Instance Segmentation Metrics]",
        f"Mask mAP@50: {mask_map50:.4f} ",
        f"Mask mAP@50-95: {mask_map50_95:.4f}",
        "----------------------------------------",
        f"Hardware Inference Speed: {inference_speed:.2f} ms per image",
        "========================================"
    ]
    report_content = "\n".join(report_lines)

    # Save the report to a text file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"model_metrics_{timestamp}.txt")
    
    with open(output_file, "w") as f:
        f.write(report_content)

    print(report_content)
    print(f"\nModel evaluation saved permanently to {output_file}")

if __name__ == "__main__":
    # Replace with your actual dataset configuration file
    evaluate_perception_model(
        model_path="yolo26n-seg.pt",
        data_yaml="coco8-seg.yaml" 
    )