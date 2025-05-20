import cv2
import torch
from torchvision import transforms


def process_video(video_path, frame_limit=16, size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret or count >= frame_limit:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, size)
        frames.append(frame)
        count += 1
    cap.release()

    # Convert to tensor and normalize
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    frames = [transform(frame) for frame in frames]
    frames = torch.stack(frames)  # [T, C, H, W]
    return frames.permute(1, 0, 2, 3)  # [C, T, H, W]


# Example usage
video_tensor = process_video("path/to/video.mp4")
