"""
ONNX Runtime NER Model Implementation
"""
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from typing import List, Dict, Tuple
from . import config
import os


class ONNXNERModel:
    """
    NER Model using ONNX Runtime for inference
    """
    
    def __init__(self, model_path: str = None, tokenizer_path: str = None):
        """
        Initialize the ONNX NER model
        
        Args:
            model_path: Path to the ONNX model file
            tokenizer_path: Path to the tokenizer directory
        """
        self.model_path = model_path or str(config.MODEL_FILE)
        self.tokenizer_path = tokenizer_path or str(config.MODEL_DIR)
        
        # Check if model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please run 'python faskapi/download_model.py' to download the model."
            )
        
        # Load ONNX model
        print(f"Loading ONNX model from: {self.model_path}")
        self.session = ort.InferenceSession(
            self.model_path,
            providers=['CPUExecutionProvider']
        )
        
        # Load tokenizer from model directory
        print(f"Loading tokenizer from: {self.tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        
        # Get model input/output names
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Load label mapping from config
        import json
        config_path = os.path.join(self.tokenizer_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                model_config = json.load(f)
                self.id2label = {int(k): v for k, v in model_config.get('id2label', {}).items()}
                print(f"Loaded {len(self.id2label)} labels from config")
        else:
            # Fallback labels
            self.id2label = {
                0: "O",
                1: "B-STREET",
                2: "I-STREET",
                3: "B-WARD",
                4: "I-WARD",
                5: "B-DISTRICT",
                6: "I-DISTRICT",
                7: "B-PROVINCE",
                8: "I-PROVINCE",
            }
        
        print("Model loaded successfully!")
    
    def tokenize(self, text: str) -> Dict[str, np.ndarray]:
        """
        Tokenize input text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with tokenized inputs
        """
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.MAX_LENGTH,
            return_tensors="np"
        )
        # Ensure correct dtype for ONNX
        return {
            'input_ids': encoded['input_ids'].astype(np.int64),
            'attention_mask': encoded['attention_mask'].astype(np.int64)
        }
    
    def predict(self, text: str) -> List[Tuple[str, str]]:
        """
        Perform NER prediction on input text
        
        Args:
            text: Input text
            
        Returns:
            List of (token, label) tuples
        """
        # Tokenize
        encoded = self.tokenize(text)
        
        # Prepare inputs for ONNX model
        onnx_inputs = {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }
        
        # Run inference with explicit input/output names
        outputs = self.session.run(['logits'], onnx_inputs)
        
        # Get predictions from logits
        logits = outputs[0]  # Shape: [batch_size, seq_len, num_labels]
        predictions = np.argmax(logits, axis=-1)[0]  # Shape: [seq_len]
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        
        # Create result pairs (skip special tokens)
        results = []
        for token, pred_id in zip(tokens, predictions):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                label = self.id2label.get(int(pred_id), "O")
                results.append((token, label))
        
        return results
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with entity types as keys and entity values as lists
        """
        predictions = self.predict(text)
        
        entities = {
            "STREET": [],
            "WARD": [],
            "DISTRICT": [],
            "PROVINCE": []
        }
        
        current_entity = ""
        current_type = None
        
        for token, label in predictions:
            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity and current_type:
                    entities[current_type].append(current_entity.strip())
                
                # Start new entity
                current_type = label[2:]
                # Handle subword tokens (##)
                if token.startswith("##"):
                    current_entity = token[2:]  # Remove ##
                else:
                    current_entity = token
                
            elif label.startswith("I-") and current_type:
                # Continue current entity
                entity_type = label[2:]
                if entity_type == current_type:
                    # Handle subword tokens (##)
                    if token.startswith("##"):
                        current_entity += token[2:]  # Append without space
                    else:
                        current_entity += " " + token  # Append with space
            else:
                # Save current entity if exists
                if current_entity and current_type:
                    entities[current_type].append(current_entity.strip())
                current_entity = ""
                current_type = None
        
        # Save last entity if exists
        if current_entity and current_type:
            entities[current_type].append(current_entity.strip())
        
        return entities


# Global model instance
_model_instance = None


def get_model() -> ONNXNERModel:
    """
    Get or create the global model instance
    
    Returns:
        ONNXNERModel instance
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = ONNXNERModel()
    return _model_instance
