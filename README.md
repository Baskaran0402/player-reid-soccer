⚽ Player Re-identification in Soccer Videos
This project implements player re-identification for a 15-second 720p soccer video (15sec_input_720p.mp4), aligning with Option 2 of the assignment. The goal is to maintain consistent player IDs across frames, even when players leave and re-enter the frame (e.g., during goal events). A fine-tuned YOLOv11 model (best.pt) is used for detection, combined with an IoU- and feature-based tracking system.

📌 Overview

Detect players and goalkeepers using a custom YOLOv11 model.
Assign unique IDs and maintain identity across frames.
Handle temporary occlusion and re-entry for up to 45 frames.
Generate an annotated output video with green bounding boxes and labels.


🔧 Requirements

Python 3.8+
opencv-python==4.12.0.88
ultralytics==8.0.145
numpy==1.21.6
scipy==1.7.3


⚙️ Setup Instructions
1. Clone the Repository
git clone https://github.com/Baskaran0402/player-reid-soccer.git
cd player-reid-soccer

2. Create and Activate Virtual Environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # For Windows PowerShell
# or
venv\Scripts\activate       # For Windows CMD

3. Install Dependencies
pip install -r requirements.txt

4. Download Required Files
Download the following from Google Drive folder:

15sec_input_720p.mp4 – Input video
best.pt – YOLOv11 model weights
output_tracked.mp4 – Output result

Place them in the project root directory.

🚀 Run the Script
python player.py


Output: output_tracked.mp4
The output video shows green bounding boxes with ID: <number> annotations for players and goalkeepers.


🎞️ Output & Performance

Total Frames: 375
Processing Time: ~2554 seconds on CPU (~0.15 FPS)
Result: Green bounding boxes with consistent ID: <number> annotations
Example Frames: Shown in report.pdf (Figures 1 and 2)


📁 Project Structure
player-reid-soccer/
├── player.py               # Main script for detection & tracking
├── report.pdf              # Final report with methods & results
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── .gitignore              # Ignored files
├── 15sec_input_720p.mp4    # Input video (external, via Google Drive)
├── best.pt                 # YOLOv11 model (external, via Google Drive)
└── output_tracked.mp4      # Annotated output video (external, via Google Drive)


💡 Notes

Detects players (class 2) and goalkeepers (class 1) using YOLOv11.
Uses IoU threshold = 0.3 for frame-to-frame tracking and cosine distance = 0.4 for re-identification after up to 45-frame disappearance (~3 seconds at 25 FPS).
Performance can be improved with GPU acceleration (CUDA) or a smaller model (e.g., yolov11s.pt).
The project is self-contained, with all dependencies listed and instructions provided for reproducibility.


📚 Report
Refer to report.pdf for detailed insights into:

Model architecture
Tracking methodology
Challenges & future scope
Result analysis


👨‍💻 Author
Baskaran SeennavasanGitHub: @Baskaran0402

📜 License
This project is provided for academic purposes. For reuse or extension, please contact the author.