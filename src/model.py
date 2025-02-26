import torch
import faiss
import numpy as np
from tqdm import tqdm
from .evaluation import accuracy, accuracy_v2

class kNNImageCaptioningModel:
    def __init__(self, k, index_factory=None):
        """
        Initialize KNN Image Captioning Model.
        
        Args:
            k (int): Number of nearest neighbors to consider
            index_factory (str, optional): FAISS index factory string
        """
        self.k = k
        self.captions_np_flat = []
        self.index_factory = index_factory

    def findSimilarity(self, image_embeddings):
        """
        Find similar images using FAISS.
        
        Args:
            image_embeddings (np.ndarray): Image embeddings matrix
            
        Returns:
            tuple: (similarities, indices) of nearest neighbors
        """
        if(self.index_factory == None):
            index = faiss.IndexFlatL2(image_embeddings.shape[1])
        else:
            index = faiss.index_factory(image_embeddings.shape[1], self.index_factory)

        index.add(image_embeddings)
        
        distances, indices = index.search(image_embeddings, self.k + 1)
        
        # Exclude the self-neighbor
        distances = distances[:, 1:]
        indices = indices[:, 1:]

        Z = 1.25  # Normalization factor
        similarities = 1 - distances / Z

        return similarities, indices

    def queryExpansion(self, captions, similarities):
        """
        Perform query expansion using caption embeddings.
        
        Args:
            captions (np.ndarray): Caption embeddings
            similarities (np.ndarray): Similarity scores
            
        Returns:
            np.ndarray: Expanded queries
        """
        expanded_queries = []
        print("Creating query")
        for num_images in tqdm(range(len(captions)), total=len(captions)):
            q = []
            for i in range(self.k):
                for j in range(5):
                    q.append(captions[num_images][j] * similarities[num_images][i])
            q = np.array(q)
            q = np.sum(q, axis=0) / (5 * self.k)
            expanded_queries.append(q)
        return np.array(expanded_queries)

    def findBestCaption(self, expanded_queries, caption_embeddings):
        """
        Find best matching caption for each query.
        
        Args:
            expanded_queries (np.ndarray): Expanded query vectors
            caption_embeddings (np.ndarray): Caption embeddings
            
        Returns:
            tuple: (indices, distances) of best matching captions
        """
        index_flat_ip = faiss.IndexFlatIP(caption_embeddings[0].shape[1])
        index_flat_ip = faiss.index_cpu_to_all_gpus(index_flat_ip)

        caption_embeddings_flat = caption_embeddings.reshape(-1, caption_embeddings.shape[-1])
        caption_embeddings_flat = caption_embeddings_flat / np.linalg.norm(caption_embeddings_flat, axis=1, keepdims=True)

        index_flat_ip.add(caption_embeddings_flat)
        best_caption_distance, best_caption_index = index_flat_ip.search(expanded_queries, 1)

        return best_caption_index, best_caption_distance

    def computeBleuScore(self, best_caption_index, captions_np):
        """
        Compute BLEU scores for predictions.
        
        Args:
            best_caption_index (np.ndarray): Indices of predicted captions
            captions_np (np.ndarray): Ground truth captions
            
        Returns:
            tuple: (bleu_scores, num_different_predictions)
        """
        num_different_predictions = 0
        self.captions_np_flat = captions_np.flatten()
        
        bleu_scores = []
        for idx in range(len(best_caption_index)):
            predicted_caption = self.captions_np_flat[best_caption_index[idx][0]]
            true_captions = captions_np[idx]

            if predicted_caption not in true_captions:
                predicted_caption = [predicted_caption]
                true_captions = [true_captions]
                num_different_predictions += 1
                accuracy = accuracy_v2(predicted_caption, true_captions)
                bleu_scores.append(accuracy)
            else:
                bleu_scores.append(1.0)

        return bleu_scores, num_different_predictions