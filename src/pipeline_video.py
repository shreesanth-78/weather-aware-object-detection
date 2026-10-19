"""
pipeline_video.py
End-to-end adaptive video processing pipeline for dashcam footage.
Executes temporal classification, conditional image enhancement, and YOLO11n detection.
"""

import os
import time
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

import config
from model import WeatherClassifier
from enhance_fog import dehaze_image
from enhance_rain import derain_image
from enhance_night import brighten_image
from detector import detect_objects

# 1. Initialize Weather Classifier
def load_classifier():
    model = WeatherClassifier(num_classes=config.NUM_CLASSES, pretrained=False)
    state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    model.load_state_dict(state_dict, strict=True)
    model.to(config.DEVICE)
    model.eval()
    return model

classifier_model = load_classifier()

classifier_transform = transforms.Compose([
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_weather_frame(image_bgr):
    """Predicts weather condition with softmax confidence."""
    rgb_img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)
    input_tensor = classifier_transform(pil_img).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        logits = classifier_model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    return config.CLASSES[pred_idx.item()], conf.item() * 100.0

def process_video(input_path, output_path=None, classify_interval=15, max_frames=None, resize_dim=None):
    """
    Processes video stream through the weather-aware pipeline.
    
    Args:
        input_path: Path to input .mp4 file or webcam index (e.g., 0).
        output_path: Path to save the side-by-side comparison video.
        classify_interval: How often (in frames) to re-evaluate the weather classifier.
        max_frames: Max frames to process (useful for quick evaluation).
        resize_dim: Optional (width, height) tuple to scale input frames for faster processing.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {input_path}")

    # Video properties
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    proc_w = resize_dim[0] if resize_dim else orig_w
    proc_h = resize_dim[1] if resize_dim else orig_h

    # Output video dimensions: Side-by-side (Raw Left, Enhanced+YOLO Right)
    out_w = proc_w * 2
    out_h = proc_h

    if output_path is None:
        os.makedirs(os.path.join(config.ROOT_DIR, 'output'), exist_ok=True)
        output_path = os.path.join(config.ROOT_DIR, 'output', 'pipeline_video_output.mp4')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    print("=" * 60)
    print("INITIALIZING VIDEO INFERENCE PIPELINE")
    print(f"Source: {input_path} | Native: {orig_w}x{orig_h} @ {fps:.1f} FPS")
    print(f"Output: {output_path} | Render Size: {out_w}x{out_h}")
    print(f"Temporal Classification Interval: Every {classify_interval} frames")
    print("=" * 60)

    current_weather = "clear"
    current_conf = 0.0
    frame_count = 0
    t_start_total = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if resize_dim:
                frame = cv2.resize(frame, (proc_w, proc_h))

            frame_count += 1
            if max_frames and frame_count > max_frames:
                break

            t0 = time.time()

            # 1. Periodic Weather Classification
            if frame_count == 1 or (frame_count % classify_interval == 0):
                current_weather, current_conf = predict_weather_frame(frame)

            # 2. Dynamic Adaptive Enhancement
            t_enh_start = time.time()
            if current_weather == 'fog':
                enhanced_frame = dehaze_image(frame)
                enhancer_name = "DehazeFormer-S"
            elif current_weather == 'rain':
                enhanced_frame = derain_image(frame)
                enhancer_name = "Restormer"
            elif current_weather == 'night':
                enhanced_frame = brighten_image(frame)
                enhancer_name = "Zero-DCE++"
            else:
                enhanced_frame = frame.copy()
                enhancer_name = "Bypass (Clear)"
            t_enh = time.time() - t_enh_start

            # 3. Downstream Detection (YOLO11n)
            t_det_start = time.time()
            annotated_frame, raw_results = detect_objects(enhanced_frame)
            t_det = time.time() - t_det_start

            total_frame_time = time.time() - t0
            instant_fps = 1.0 / max(total_frame_time, 1e-5)

            # 4. Telemetry Overlay
            # Left pane: Raw Frame
            left_pane = frame.copy()
            cv2.rectangle(left_pane, (10, 10), (380, 85), (0, 0, 0), -1)
            cv2.putText(left_pane, f"RAW INPUT STREAM", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(left_pane, f"Class: {current_weather.upper()} ({current_conf:.1f}%)", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Right pane: Processed Frame
            num_detections = len(raw_results.boxes) if raw_results and hasattr(raw_results, 'boxes') else 0
            cv2.rectangle(annotated_frame, (10, 10), (450, 115), (0, 0, 0), -1)
            cv2.putText(annotated_frame, f"ADAPTIVE PIPELINE + YOLO11n", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Active Module: {enhancer_name}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(annotated_frame, f"Detections: {num_detections} | FPS: {instant_fps:.1f}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

            # Side-by-side stitch
            composite = np.hstack((left_pane, annotated_frame))
            out_writer.write(composite)

            if frame_count % 10 == 0:
                print(f"[Frame {frame_count:04d}/{total_frames if total_frames > 0 else '?'}] "
                      f"State: {current_weather.upper():<5} | Enh: {enhancer_name:<14} | "
                      f"Boxes: {num_detections:<2} | Pipeline: {instant_fps:.1f} FPS")

    finally:
        cap.release()
        out_writer.release()

    elapsed = time.time() - t_start_total
    avg_fps = frame_count / max(elapsed, 1e-5)
    print("=" * 60)
    print(f"VIDEO PROCESSING COMPLETE")
    print(f"Total Frames Processed: {frame_count}")
    print(f"Elapsed Time: {elapsed:.2f}s | Average Speed: {avg_fps:.2f} FPS")
    print(f"Output Video Saved: {output_path}")
    print("=" * 60)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Adaptive Weather-Aware Video Inference")
    parser.add_argument('--input', type=str, default=None, help="Path to input video file (e.g., dashcam.mp4)")
    parser.add_argument('--output', type=str, default=None, help="Path to output video file")
    parser.add_argument('--interval', type=int, default=15, help="Frame interval for weather classification")
    parser.add_argument('--max_frames', type=int, default=None, help="Max frames to process")
    parser.add_argument('--resize', type=int, nargs=2, default=None, help="Target width and height (e.g., 640 360)")
    args = parser.parse_args()

    if args.input is None or not os.path.exists(args.input):
        print("\n[!] No input video specified or file not found.")
        print("Usage Example:")
        print("    python src/pipeline_video.py --input test_dashcam.mp4 --resize 640 360")
    else:
        resize_tuple = tuple(args.resize) if args.resize else None
        process_video(input_path=args.input, 
                      output_path=args.output, 
                      classify_interval=args.interval, 
                      max_frames=args.max_frames, 
                      resize_dim=resize_tuple)