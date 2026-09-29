from src.utils.generate_labels_mediapipe import MediaPipeLabelGenerator

generator = MediaPipeLabelGenerator(output_dir="src/data")
generator.process_video("boxing_sample.mp4", sample_rate=5)

# 라벨링