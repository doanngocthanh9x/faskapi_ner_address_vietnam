"""
Script to train a NER model and convert it to ONNX format

This script:
1. Loads the Vietnamese address NER dataset
2. Trains a transformer model for NER
3. Converts the trained model to ONNX format
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import load_dataset
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_dataset():
    """Load and prepare the Vietnamese address NER dataset"""
    logger.info("Loading dataset from dathuynh1108/ner-address-standard-dataset")
    
    try:
        dataset = load_dataset("dathuynh1108/ner-address-standard-dataset")
        logger.info(f"Dataset loaded successfully: {dataset}")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Creating a dummy dataset for testing")
        
        # Create dummy dataset for testing
        from datasets import Dataset
        dummy_data = {
            'tokens': [
                ['Số', '123', 'Đường', 'Nguyễn', 'Văn', 'Linh', ',', 'Quận', '7', ',', 'TP', 'HCM'],
                ['456', 'Lê', 'Lai', ',', 'Phường', '1', ',', 'Quận', '5']
            ],
            'ner_tags': [
                [0, 0, 7, 7, 7, 7, 0, 3, 4, 0, 1, 2],
                [0, 7, 7, 0, 5, 6, 0, 3, 4]
            ]
        }
        return {'train': Dataset.from_dict(dummy_data), 'validation': Dataset.from_dict(dummy_data)}


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    """Tokenize inputs and align labels with tokens"""
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
        padding='max_length',
        max_length=128
    )

    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Simple accuracy
    correct = sum(
        sum(p == l for p, l in zip(pred, label))
        for pred, label in zip(true_predictions, true_labels)
    )
    total = sum(len(label) for label in true_labels)
    
    return {
        'accuracy': correct / total if total > 0 else 0
    }


def train_model(model_name="vinai/phobert-base", num_labels=9):
    """Train the NER model"""
    logger.info(f"Training model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Load dataset
    dataset = prepare_dataset()
    
    # Tokenize dataset
    tokenized_datasets = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets.get('validation', tokenized_datasets['train']),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    output_dir = "./models/pytorch_model"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Model saved to {output_dir}")
    return model, tokenizer


def convert_to_onnx(model_path="./models/pytorch_model", output_path="./models/model.onnx"):
    """Convert PyTorch model to ONNX format"""
    logger.info(f"Converting model to ONNX format")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    
    # Set model to eval mode
    model.eval()
    
    # Create dummy input
    dummy_input = tokenizer(
        "Số 123 Đường Nguyễn Văn Linh, Quận 7, TP HCM",
        return_tensors="pt",
        padding='max_length',
        max_length=128
    )
    
    # Export to ONNX
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    input_names = ['input_ids', 'attention_mask']
    output_names = ['logits']
    
    dynamic_axes = {
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'logits': {0: 'batch_size', 1: 'sequence'}
    }
    
    torch.onnx.export(
        model,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        output_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        do_constant_folding=True
    )
    
    logger.info(f"ONNX model saved to {output_path}")
    
    # Verify the ONNX model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verification passed")


def main():
    """Main function to train and convert model"""
    logger.info("Starting model training and conversion pipeline")
    
    # Train model
    model, tokenizer = train_model()
    
    # Convert to ONNX
    convert_to_onnx()
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
