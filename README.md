# Enhanced Anomaly Detection Framework for IoT Networks

BSc Computer Science Final Year Project, University of Reading.

A 1D-CNN for multi-class classification of IoT network traffic, trained on the CICIoT2023 dataset.

## Usage

Place the CICIoT2023 CSVs in `data/MERGED_CSV/`, then run:

```bash
# 34-class baseline
python main.py

# 15-class merged variant
python main_merged_classes.py
```