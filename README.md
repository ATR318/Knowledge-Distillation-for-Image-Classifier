# Knowledge Distillation for Image Classification

## 📌 Overview
This project demonstrates **Knowledge Distillation (KD)** for compressing a large CNN (Teacher) into a smaller CNN (Student) while maintaining competitive accuracy.  
We experimented on **CIFAR-10** and **Fashion-MNIST** datasets.

## 🧠 Approach
- Train a **Teacher Model** (deep CNN) on the dataset.
- Distill knowledge into a **Student Model**:
  - Soft labels from the Teacher.
  - Hard labels from the dataset.
  - Optional feature distillation.
- Evaluate performance on metrics such as accuracy, inference time, entropy, KL divergence, and compression ratio.

## 📊 Results (Sample)
| Metric                | Teacher Model | Student Model |
|------------------------|---------------|---------------|
| Parameters             | 14.9M         | 357K          |
| Compression Ratio      | –             | 41.88× smaller|
| Accuracy (example)     | ~92%          | ~89%          |
| Inference Time (ms)    | 87.83         | 89.90         |
| KL Divergence          | –             | 0.4666        |

## 📂 Datasets
- CIFAR-10  
- Fashion-MNIST  

## 🚀 Requirements
- Python 3.x
- TensorFlow / PyTorch
- Numpy, Matplotlib, etc.  

Install dependencies:
pip install -r requirements.txt

## ▶️ Usage
1. Train the Teacher model:
   python train_teacher.py

2. Distill into Student model:
   python distill_student.py

3. Evaluate:
   python evaluate.py

## 📌 Future Work
- Add pruning/quantization after distillation.
- Test on more complex datasets.
- Deploy Student model on edge devices.

## 🙌 Credits
Team Members:
Tarun A - @ATR318
Kavin Karthik A - @Krypton-KK
Santhosh - @Santhosh75-dom
Sri Anish G - @SriAnish189
Inspired by Hinton et al., Distilling the Knowledge in a Neural Network (2015)
