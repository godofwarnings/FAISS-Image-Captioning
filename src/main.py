import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from .model import kNNImageCaptioningModel
from .utils import load_coco_dataset, prepare_captions, load_embeddings

def evaluate_indexing_methods():
    """Evaluate different FAISS indexing methods."""
    IF_Method = ["Flat", "L2norm,Flat", "HNSW32", "HNSW32,Flat"]
    TimeTaken = []
    AvgBleuScores = []
    k = 3

    for index in IF_Method:
        print(f"Index: {index}")
        start = time.time()
        model = kNNImageCaptioningModel(k, index)
        similarities, indices = model.findSimilarity(image_embeddings)
        expanded_queries = model.queryExpansion(caption_embeddings, similarities)
        best_caption_index, best_caption_distance = model.findBestCaption(
            expanded_queries, caption_embeddings)
        bleu_scores, num_different_predictions = model.computeBleuScore(
            best_caption_index, captions_np)
        avg_bleu_score = np.mean(bleu_scores)
        end = time.time()
        
        TimeTaken.append(end - start)
        AvgBleuScores.append(avg_bleu_score)
        
        print(f"Average BLEU score: {avg_bleu_score}")
        print(f"Time taken: {end - start:.2f} seconds\n")
    
    return IF_Method, TimeTaken, AvgBleuScores

def plot_results(IF_Method, TimeTaken):
    """Plot indexing method comparison results."""
    plt.figure(figsize=(12, 6))
    plt.plot(IF_Method, TimeTaken)
    plt.xlabel("Index")
    plt.ylabel("Time taken (in seconds)")
    plt.title("Time taken vs Index")
    plt.show()

def main():
    # Load COCO dataset
    coco_dataset = load_coco_dataset(
        root_dir='data/raw/val2014',
        annFile='data/raw/annotations/captions_val2014.json'
    )
    
    # Prepare captions
    global captions_np
    captions_np, captions_flat = prepare_captions(coco_dataset)
    print(f"Total captions: {len(captions_flat)}")
    
    # Load embeddings
    global image_embeddings, caption_embeddings
    image_embeddings, caption_embeddings = load_embeddings(
        'data/embeddings/coco_imgs.npy',
        'data/embeddings/coco_captions.npy'
    )
    print("Embeddings loaded:", image_embeddings.shape, caption_embeddings.shape)
    
    # Basic model evaluation
    model = kNNImageCaptioningModel(k=3)
    similarities, indices = model.findSimilarity(image_embeddings)
    expanded_queries = model.queryExpansion(caption_embeddings, similarities)
    best_caption_index, best_caption_distance = model.findBestCaption(
        expanded_queries, caption_embeddings)
    bleu_scores, num_different_predictions = model.computeBleuScore(
        best_caption_index, captions_np)
    avg_bleu_score = np.mean(bleu_scores)
    
    print(f"\nBasic model results:")
    print(f"Number of different predictions: {num_different_predictions}")
    print(f"Average BLEU score: {avg_bleu_score}")
    
    # Evaluate different indexing methods
    print("\nEvaluating different indexing methods...")
    IF_Method, TimeTaken, AvgBleuScores = evaluate_indexing_methods()
    
    # Plot results
    plot_results(IF_Method, TimeTaken)

if __name__ == "__main__":
    main()