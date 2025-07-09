# Image Caption Generator - Configuration Guide

## Project Setup

### GitHub Repository Setup

1. **Create a new repository on GitHub**
   - Go to [GitHub](https://github.com) and create a new repository
   - Choose an appropriate name (e.g., "image-caption-generator")
   - Add a description and README if desired

2. **Initialize Git in your local project**
   ```bash
   cd /Users/bandisanjay/Desktop/DL
   git init
   ```

3. **Add your files to Git**
   ```bash
   git add .
   ```

4. **Commit your changes**
   ```bash
   git commit -m "Initial commit: Image Caption Generator with translation feature"
   ```

5. **Add the remote repository**
   ```bash
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY-NAME.git
   ```

6. **Push to GitHub**
   ```bash
   git push -u origin main
   ```
   Note: If your default branch is "master" instead of "main", replace "main" with "master"

### Required Files for GitHub

The following files should be included in your GitHub repository:

```
DL/
├── .gitignore                      # Git ignore file (excludes unnecessary files)
├── app.py                          # Main Flask web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── PROJECT_STRUCTURE.md            # Description of project structure
├── PROJECT_SUMMARY.md              # Summary of the project
├── TECHNICAL_REPORT.md             # Technical details
├── COMPLETE_DOCUMENTATION.md       # Complete documentation
├── CONFIG.md                       # This file - setup and deployment guide
├── models/                         # AI models
│   ├── cnn_feature_extractor.py   # CNN feature extraction
│   ├── transformers_captioner.py  # Transformer-based captioning
│   ├── ensemble_captioner.py      # Ensemble captioning
│   ├── enhanced_captioner.py      # Rule-based captioning
│   ├── scene_captioner.py         # Scene understanding
│   ├── object_detector.py         # Object detection
│   └── FEATURE_EXTRACTOR_README.md # Feature extractor documentation
├── templates/                      # Web UI templates
│   └── index.html                 # Main web interface
└── uploads/                        # User uploaded images
    └── .gitkeep                   # Empty file to preserve directory
```

## Environment Configuration

### Local Development

1. **Python Version**
   - Python 3.8 or higher is required

2. **Virtual Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```
   The application will be available at http://127.0.0.1:5000

### Environment Variables

For added security, consider using environment variables for any sensitive information:

```bash
# Example .env file (not tracked by Git)
FLASK_SECRET_KEY=your_secret_key_here
MODEL_PATH=/path/to/models
```

## Deployment Options

### Option 1: Heroku Deployment

1. **Create a Procfile**
   ```
   web: gunicorn app:app
   ```

2. **Add gunicorn to requirements.txt**
   ```
   gunicorn==20.1.0
   ```

3. **Create a runtime.txt**
   ```
   python-3.9.7
   ```

4. **Deploy to Heroku**
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### Option 2: Docker Deployment

1. **Create a Dockerfile**
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   EXPOSE 5000

   CMD ["python", "app.py"]
   ```

2. **Build and Run Docker Container**
   ```bash
   docker build -t image-caption-generator .
   docker run -p 5000:5000 image-caption-generator
   ```

### Option 3: AWS Elastic Beanstalk

1. **Install AWS EB CLI**
   ```bash
   pip install awsebcli
   ```

2. **Initialize EB Application**
   ```bash
   eb init -p python-3.8 image-caption-generator
   ```

3. **Create Environment and Deploy**
   ```bash
   eb create caption-generator-env
   ```

## Model Management

### Downloading Pre-trained Models

Some models are downloaded automatically when the application is first run. To pre-download:

```python
from models.transformers_captioner import TransformersCaptioner
captioner = TransformersCaptioner()  # This will download the model
```

### Model Storage

By default, models are stored in the cache directory specified by the transformers library. You can configure a custom location:

```python
import os
os.environ["TRANSFORMERS_CACHE"] = "/path/to/your/models"
```

## Troubleshooting

### Common Issues

1. **Model Download Issues**
   - Check your internet connection
   - Set up a proxy if needed

2. **Memory Issues**
   - Reduce batch size
   - Use CPU-only mode if GPU memory is limited

3. **Translation Service Issues**
   - Check that googletrans is properly installed
   - Make sure you have internet access for translation API

## Maintenance

### Updating Models

```bash
pip install --upgrade transformers
```

### Backing Up Data

```bash
# Backup generated captions
zip -r captions_backup.zip uploads/
```

### Monitoring

Consider adding application monitoring with tools like Prometheus + Grafana or AWS CloudWatch.

## Security Considerations

1. Implement HTTPS in production
2. Set appropriate file upload limits
3. Validate file types and content
4. Implement rate limiting for API endpoints

## License

This project is licensed under the MIT License - see the LICENSE file for details.
