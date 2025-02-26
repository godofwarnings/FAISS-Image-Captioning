import torch
import numpy as np
from PIL import Image
from torchvision import transforms, datasets

def get_transform():
    """Get image transformation pipeline."""
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        )
    ])

def load_coco_dataset(root_dir, annFile):
    """
    Load COCO dataset.
    
    Args:
        root_dir (str): Root directory containing images
        annFile (str): Path to annotation file
        
    Returns:
        torch.utils.data.Dataset: COCO dataset
    """
    transform = get_transform()
    return datasets.CocoCaptions(
        root=root_dir,
        annFile=annFile,
        transform=transform
    )

def prepare_captions(coco_dataset):
    """
    Prepare captions from COCO dataset.
    
    Args:
        coco_dataset: COCO dataset object
        
    Returns:
        tuple: (captions_np, captions_flat)
    """
    ids = list(sorted(coco_dataset.coco.imgs.keys()))
    captions = []
    for i in range(len(ids)):
        captions.append([ele['caption'] for ele in 
                        coco_dataset.coco.loadAnns(
                            coco_dataset.coco.getAnnIds(ids[i])
                        )][:5])
    captions_np = np.array(captions)
    captions_flat = captions_np.flatten().tolist()
    return captions_np, captions_flat

def load_embeddings(img_path, cap_path):
    """
    Load pre-computed embeddings.
    
    Args:
        img_path (str): Path to image embeddings
        cap_path (str): Path to caption embeddings
        
    Returns:
        tuple: (image_embeddings, caption_embeddings)
    """
    image_embeddings = np.load(img_path)
    caption_embeddings = np.load(cap_path)
    return image_embeddings, caption_embeddings