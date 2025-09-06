# ASL Hand Gesture Recognition System

Real-time American Sign Language (ASL) digit recognition system that classifies hand gestures for digits 0-9 using an ensemble of deep learning models with live webcam integration.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-v2.12-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-v1.44+-red.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-v4.11+-green.svg)

## 🌟 Features

- **Real-time Recognition**: Live webcam-based hand gesture classification
- **Ensemble Learning**: Combines Custom CNN and MobileNetV2 for robust predictions
- **High Accuracy**: Custom CNN achieves 98% test accuracy
- **Interactive Interface**: User-friendly Streamlit web application
- **ROI-based Processing**: Focused region detection for improved accuracy
- **Lightweight Deployment**: Optimized for real-time inference

## 🏗️ Architecture

### Model Ensemble
The system employs a dual-model ensemble approach:

1. **Custom CNN Architecture**
   - 3 Convolutional blocks with increasing filters (32→64→128)
   - MaxPooling layers for dimensionality reduction
   - Dense layers: 64→128→128→10 (softmax)
   - **Test Accuracy: 98.0%**

2. **MobileNetV2 Transfer Learning**
   - Pre-trained ImageNet weights (frozen base)
   - Custom classification head
   - **Test Accuracy: 79.0%**

3. **Ensemble Method**
   - Simple averaging of model predictions
   - Final classification via argmax

### Processing Pipeline
```
Webcam Input → ROI Extraction → Preprocessing → Dual Model Inference → Ensemble → Display
```

## 📋 Requirements

```
streamlit==1.44.1
tensorflow==2.12.0
opencv-python==4.11.0.86
numpy==1.26.4
```

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/smkcv3/ASL_sign_detection.git
   cd ASL_sign_detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pre-trained models**
   - Ensure `custom_cnn.h5` and `custom_mobilenet.h5` are in the project root
   - Models are included in the repository (Total size: ~22MB)

## 💻 Usage

### Running the Application
```bash
streamlit run app.py
```

### Using the Interface
1. **Position your hand** within the orange ROI rectangle
2. **Form digit gestures** (0-9) with your hand
3. **View real-time predictions** displayed on screen
4. **Ensure good lighting** and clear background for optimal results

### Controls
- The application runs continuously until stopped
- Press `Ctrl+C` in terminal to exit
- Webcam permissions required for operation

## 📊 Dataset

### Dataset Statistics
- **Total Images**: 704
- **Training Set**: 604 images
- **Test Set**: 100 images
- **Classes**: 10 (digits 0-9)
- **Image Format**: 64x64 RGB JPEG
- **Distribution**: ~60 images per class (training)

### Dataset Structure
```
Dataset/
├── Train/
│   ├── 0/ (60 images)
│   ├── 1/ (60 images)
│   ├── 2/ (60 images)
│   └── ... (up to 9)
└── Test/
    ├── 0/ (10 images)
    ├── 1/ (10 images)
    └── ... (up to 9)
```

### Data Preprocessing
- **Augmentation**: Rotation (±20°), zoom (±20%), horizontal flip
- **Normalization**: MobileNetV2 preprocessing (range [-1, 1])
- **Target Size**: 64x64 pixels

## 🎯 Performance

| Model | Training Accuracy | Test Accuracy | Model Size |
|-------|------------------|---------------|------------|
| Custom CNN | 94.87% | **98.0%** | 5.0 MB |
| MobileNetV2 | 83.28% | 79.0% | 17.3 MB |
| **Ensemble** | - | **Enhanced** | 22.3 MB |

### Training Configuration
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 10
- **Epochs**: 10 (with early stopping)
- **Callbacks**: ReduceLROnPlateau, EarlyStopping

## 📁 Project Structure

```
ASL_sign_detection/
├── app.py                      # Streamlit web application
├── ASLp.ipynb                 # Training notebook (Google Colab)
├── custom_cnn.h5              # Trained Custom CNN model
├── custom_mobilenet.h5        # Trained MobileNetV2 model
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
├── Dataset/                   # Training and test data
│   ├── Train/                 # Training images (604 total)
│   └── Test/                  # Test images (100 total)
└── docs/                      # Project documentation
    ├── capstone report.pdf    # Detailed project report
    ├── arun final ppt.pptx   # Project presentation
    └── content report.pdf     # Additional documentation
```

## 🔧 Technical Details

### ROI Configuration
- **Top**: 100px, **Bottom**: 300px
- **Left**: 350px, **Right**: 150px
- **Size**: 200x200 pixel region
- **Purpose**: Focused gesture detection area

### Model Specifications
- **Input Shape**: (64, 64, 3)
- **Output Classes**: 10 (digits 0-9)
- **Class Mapping**: {0:'One', 1:'Ten', 2:'Two', 3:'Three', 4:'Four', 5:'Five', 6:'Six', 7:'Seven', 8:'Eight', 9:'Nine'}

### Real-time Processing
- **Frame Processing**: BGR→RGB conversion
- **Inference Speed**: Real-time (30+ FPS capable)
- **Memory Usage**: Optimized for standard hardware

## 🚧 Limitations

- **Dataset Size**: Limited to 704 images (potential overfitting)
- **Fixed ROI**: Requires hand positioning in specific area
- **Lighting Sensitivity**: Performance varies with lighting conditions
- **Single Hand**: Designed for one hand gestures only
- **Static Background**: Optimal performance with minimal background clutter

## 🔮 Future Enhancements

- [ ] **Expand Dataset**: Collect more diverse hand gesture images
- [ ] **Dynamic ROI**: Implement hand detection for automatic ROI
- [ ] **Multi-hand Support**: Enable recognition of multiple hands
- [ ] **Background Subtraction**: Improve robustness to background variations
- [ ] **Mobile Deployment**: Create mobile app version
- [ ] **Additional Gestures**: Extend beyond digits to full ASL alphabet
- [ ] **Model Optimization**: Quantization for faster inference
- [ ] **Cloud Deployment**: Web-based application hosting

## 🛠️ Development

### Training the Models
The models were trained using Google Colab with GPU acceleration:

1. **Data Loading**: ImageDataGenerator with augmentation
2. **Model Creation**: Custom CNN and MobileNetV2 architectures
3. **Training**: Individual model training with callbacks
4. **Ensemble**: Averaging layer for combined predictions
5. **Evaluation**: Performance testing on validation set

### Retraining Models
To retrain the models with new data:

1. Update dataset paths in `ASLp.ipynb`
2. Modify augmentation parameters if needed
3. Run the notebook cells sequentially
4. Save updated models as `.h5` files

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Authors

- **smkcv3** - *Initial work* - [GitHub](https://github.com/smkcv3)

## 🙏 Acknowledgments

- **Dataset**: Custom collected ASL digit gesture dataset
- **MobileNetV2**: Pre-trained ImageNet weights from TensorFlow/Keras
- **Streamlit**: For the interactive web interface
- **TensorFlow**: Deep learning framework
- **OpenCV**: Computer vision library

## 📚 References

1. Sandler, M., et al. "MobileNetV2: Inverted Residuals and Linear Bottlenecks." CVPR 2018.
2. ASL Digit Recognition techniques and methodologies
3. Ensemble learning approaches in computer vision

---

**⭐ Star this repository if you find it helpful!**

For questions or issues, please [open an issue](https://github.com/smkcv3/ASL_sign_detection/issues) or contact the repository maintainer.
