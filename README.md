# Image Captioning based on KNN using FAISS

This project implements an algorithm for Image Captioning using K-Nearest Neighbors (KNN) approach. The implementation is based on the paper [A Distributed Representation Based Query Expansion Approach for Image Captioning](https://aclanthology.org/P15-2018.pdf).

## Overview

Instead of using modern Vision Language Models (VLMs), this project explores an interesting approach from earlier years that uses KNN for image captioning. The implementation uses FAISS library for efficient similarity search and demonstrates surprisingly good results.

### File Structure

```
Image-Captioning-based-on-KNN-using-FAISS/
│
├── README.md                   # Project documentation
├── requirements.txt            # Project dependencies
├── image_captioning_knn.ipynb  # Jupyter notebooks for exploration
│
├── src/                        # Source code
│    ├── __init__.py
│    ├── model.py               # Main KNN model implementation
│    ├── utils.py               # Helper functions
│    ├── main.py  
│    └── evaluation.py          # Evaluation metrics
│
└── data/                       # Directory for data files
   ├── embeddings/             	# Pre-computed embeddings
   │   ├── coco_captions.npy
   │   └── coco_imgs.npy
   └── raw/                     # Original COCO dataset files
       ├── val2014/
       └── annotations/
```

### Key Features

- KNN-based image captioning implementation
- FAISS for efficient nearest neighbor computation
- CLIP embeddings for images and text
- BLEU score evaluation
- Various indexing options for performance optimization

## Data

The project uses:

- MS COCO 2014 validation set
- Pre-computed CLIP embeddings for images and captions
- 5 captions per image in the dataset

## Implementation Details

### Algorithm Steps

For each image:

- Find k nearest images using image embeddings
- Compute query vector as weighted sum of captions from nearest images
- Predict caption by finding closest caption embedding to query vector

### Key Components

1. KNN Image Captioning Model class with methods:

   - findSimilarity(): Computes similar images using FAISS
   - queryExpansion(): Creates expanded query vectors
   - findBestCaption(): Retrieves most suitable caption
   - computeBleuScore(): Evaluates prediction quality
2. Indexing Options:

   - Flat (default)
   - L2norm,Flat
   - HNSW32
   - HNSW32,Flat

## Results

The implementation achieves:

- Average BLEU score: ~0.84
- Different indexing methods show varying performance in terms of speed
- HNSW32,Flat provides the best balance of accuracy and speed

## Usage

1. Install requirements:
   ```
   pip install -r requirements.txt
   ```
2. Prepare data:

- Download COCO validation set
- Download pre-computed embeddings
- Place them in appropriate data directories

2. Run the model:

```python
from src.model import kNNImageCaptioningModel

model = kNNImageCaptioningModel(k=3)
captions = model.generate_captions(images)
```

## Requirements

- Python 3.7+
- PyTorch
- FAISS
- NLTK
- NumPy
- Matplotlib
