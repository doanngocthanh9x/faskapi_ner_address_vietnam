"""
Script to download and explore the Vietnamese address NER dataset
"""
from datasets import load_dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Download and display dataset information"""
    logger.info("Downloading dataset: dathuynh1108/ner-address-standard-dataset")
    
    try:
        dataset = load_dataset("dathuynh1108/ner-address-standard-dataset")
        
        logger.info(f"\nDataset structure:")
        logger.info(f"{dataset}")
        
        if 'train' in dataset:
            logger.info(f"\nTrain set size: {len(dataset['train'])}")
            logger.info(f"Sample from train set:")
            logger.info(f"{dataset['train'][0]}")
        
        if 'validation' in dataset:
            logger.info(f"\nValidation set size: {len(dataset['validation'])}")
        
        if 'test' in dataset:
            logger.info(f"\nTest set size: {len(dataset['test'])}")
        
        # Display feature info
        if 'train' in dataset:
            logger.info(f"\nFeatures:")
            logger.info(f"{dataset['train'].features}")
        
        logger.info("\nDataset downloaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to download dataset: {e}")
        logger.info("\nPlease check:")
        logger.info("1. Internet connection")
        logger.info("2. Dataset name: dathuynh1108/ner-address-standard-dataset")
        logger.info("3. Hugging Face datasets library is installed")


if __name__ == "__main__":
    main()
