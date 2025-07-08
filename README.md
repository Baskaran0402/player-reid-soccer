

---

# ⚽ Player Re-identification in Soccer Videos

This project implements **player re-identification** for a 15-second 720p soccer video (`15sec_input_720p.mp4`), aligning with **Option 2** of the assignment. The objective is to maintain **consistent player IDs across video frames**, even when players **temporarily leave and re-enter** the scene (e.g., during goal events). The system uses a **fine-tuned YOLOv11 model (`best.pt`)** for detection, combined with an **IoU- and feature-based tracking algorithm** for identity preservation.

---

## 📌 Overview

* Detect players and goalkeepers using a custom **YOLOv11** model
* Assign and maintain **unique player IDs** across frames
* Handle **temporary occlusion** and player re-entry for up to **45 frames**
* Generate an **annotated output video** with green bounding boxes and player labels

---

## 🔧 Requirements

* Python 3.8+
* `opencv-python==4.12.0.88`
* `ultralytics==8.0.145`
* `numpy==1.21.6`
* `scipy==1.7.3`

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Baskaran0402/player-reid-soccer.git
cd player-reid-soccer
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1   # For Windows PowerShell
# or
venv\Scripts\activate         # For Windows CMD
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Required Files

Download the following from the provided **Google Drive** link and place them in the **project root directory**:

* `15sec_input_720p.mp4` – Input video
* `best.pt` – YOLOv11 model weights
* `output_tracked.mp4` – Output video (sample result)

---

## 🚀 Running the Script

```bash
python player.py
```

### 🎯 Output

* Generates `output_tracked.mp4`
* Includes green bounding boxes with `ID:<number>` labels for both players and goalkeepers

---

## 🎞️ Output & Performance

| Metric          | Value                                                          |
| --------------- | -------------------------------------------------------------- |
| Total Frames    | 375                                                            |
| Processing Time | \~2554 seconds on CPU                                          |
| Inference Speed | \~0.15 FPS                                                     |
| Output          | Green bounding boxes with consistent `ID:<number>` annotations |

> Example annotated frames are available in `report.pdf` (see Figures 1 and 2).

---

## 📁 Project Structure

```
player-reid-soccer/
├── player.py               # Main detection and tracking script
├── report.pdf              # Report with methodology and results
├── requirements.txt        # Dependency list
├── README.md               # Project documentation
├── .gitignore              # Git ignored files
├── 15sec_input_720p.mp4    # Input video (external)
├── best.pt                 # YOLOv11 model weights (external)
└── output_tracked.mp4      # Annotated result video (external)
```

---

## 💡 Technical Notes

* **Classes Detected**:

  * `Class 1`: Goalkeepers
  * `Class 2`: Outfield players
* **Tracking Parameters**:

  * IoU threshold: **0.3**
  * Cosine distance threshold: **0.4**
  * Max disappearance frames: **45** (i.e., \~3 seconds at 25 FPS)
* Performance can be significantly improved with **GPU acceleration (CUDA)** or by switching to a smaller model like `yolov11s.pt`.

---

## 📚 Report

Please refer to `report.pdf` for detailed information on:

* YOLOv11 model architecture
* Tracking algorithm and re-identification logic
* Experimental results
* Challenges faced and future improvements

---

## 👨‍💻 Author

**Baskaran Seennavasan**
GitHub: [@Baskaran0402](https://github.com/Baskaran0402)

---

## 📜 License

This project is provided **for academic and educational purposes**. For reuse, modification, or commercial use, please contact the author.

---

