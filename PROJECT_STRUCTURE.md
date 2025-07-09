# Image Caption Generator - Clean Project Structure

## Project Overview

This is a browser-based image captioning system that generates high-quality, contextually relevant captions for uploaded images using advanced AI models.

## File Structure

```
DL/
├── app.py                          # Main Flask web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── PROJECT_STRUCTURE.md            # This file - describes project structure
├── PROJECT_SUMMARY.md              # Summary of the project
├── TECHNICAL_REPORT.md             # Technical details of the implementation
├── COMPLETE_DOCUMENTATION.md       # Complete documentation
├── CONFIG.md                       # Configuration guide
├── models/                         # AI models
│   ├── cnn_feature_extractor.py   # CNN feature extraction
│   ├── transformers_captioner.py  # Transformer-based captioning (main AI model)
│   ├── ensemble_captioner.py      # Combines multiple captioning models
│   ├── enhanced_captioner.py      # Rule-based captioning
│   ├── scene_captioner.py         # Scene understanding captioning
│   ├── object_detector.py         # Object detection
│   └── FEATURE_EXTRACTOR_README.md # Documentation for feature extractor
├── templates/                      # Web UI templates
│   └── index.html                 # Main web interface
├── uploads/                        # User uploaded images
└── .venv/                         # Virtual environment
```

## Key Components

### Main Application (`app.py`)

- Flask web server
- Image processing pipeline
- Multiple caption generation using ensemble approach
- Web API endpoints

### AI Models

- **TransformersCaptioner**: Uses ViT-GPT2 for high-quality captions
- **EnsembleCaptioner**: Combines multiple models for diverse results
- **SceneCaptioner**: Scene understanding and context-aware captions
- **EnhancedCaptioner**: Rule-based captioning with image analysis
- **CNNFeatureExtractor**: Extracts features from images using VGG16

### Web Interface (`templates/index.html`)

- Modern, responsive design
- Drag-and-drop image upload
- Multiple caption display with selection
- Image analysis results

## How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Start the server: `python app.py`
3. Open browser to: `http://localhost:5000`

## Testing

Run the test script to generate multiple captions for sample images:

```bash
python test_multiple_captions.py [optional_image_path]
```

## Features

- Multiple high-quality caption generation
- Diverse caption styles (descriptive, artistic, technical, etc.)
- Real-time image analysis
- Object detection and scene understanding
- Responsive web interface
- Ensemble AI model approach for best results
