# Beyblade Battle Analysis with Computer Vision

## Overview

This project uses computer vision techniques to analyze beyblade battle video. The system employs a YOLO model for detection and K-Means clustering for team classification. It also provides detailed battle analysis, including determining the winner, tracking battle duration, and measuring beyblade spin times. These insights offer a comprehensive breakdown of each Beyblade match, enabling players to review and refine their strategies.

## Key Features

- **Beyblade, Hand, and Launcher Detection:** Utilizes YOLOv8 for detecting beyblades, hands, and launchers in video frames.
- **Team Classification:** Differentiates beyblades based on colors using K-Means clustering for team assignment.
- **Battle Analysis:** Tracks various aspects of the battle, including determining the winner, measuring battle duration, and analyzing individual Beyblade spin times.

## Dataset

The dataset used for training the YOLO model consists of annotated images of beyblades, hands, and launchers. It can be accessed and downloaded from the following [Roboflow link](https://universe.roboflow.com/computer-vision-naktq/beyblade-ddrrd).

## Trained Models

The trained YOLOv8 model used for detecting beyblades, hands, and launchers can be downloaded from [this link](blomada).

## Requirements

To run this project, ensure you have Python 3.x installed and install the required libraries using:

```bash
pip install ultralytics supervision opencv-python numpy matplotlib pandas
```

## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/syahvan/beyblade-analysis.git
   cd beyblade-analysis
   ```

2. **Download the YOLOv8 Model:**

   Place the YOLOv8 model in the `models` directory or specify the path in the configuration.

3. **Run the Analysis Script:**

   ```bash
   python main.py --input_video path_to_your_video --model_path path_to_your_model
   ```

   Replace `path_to_your_video` with the path to your input video and `path_to_your_model` with the path to the YOLOv8 model file.

4. **Review Results:**

   Processed videos and analysis results will be saved in the `output` directory. Check this directory for battle outcomes, duration, and detailed performance metrics.