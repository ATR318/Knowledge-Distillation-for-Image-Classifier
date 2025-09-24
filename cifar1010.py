import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import os

# Set appearance mode and color theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class ModelComparisonApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Teacher(10 epoch)-Student(20 epoch) Model Comparison")
        self.geometry("1400x900")
        
        # CIFAR-10 class labels
        self.class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                            'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Initialize models as None
        self.teacher_model = None
        self.student_model = None
        self.current_image = None
        self.processed_image = None
        
        # Create the GUI
        self.create_widgets()
        
        # Try to load models
        self.load_models()
    
    def create_widgets(self):
        # Main container
        main_container = ctk.CTkFrame(self)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left panel - Image selection and display
        left_panel = ctk.CTkFrame(main_container)
        left_panel.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Title
        title_label = ctk.CTkLabel(left_panel, text="Image Selection", 
                                   font=ctk.CTkFont(size=20, weight="bold"))
        title_label.pack(pady=10)
        
        # File selection frame
        file_frame = ctk.CTkFrame(left_panel)
        file_frame.pack(fill="x", padx=10, pady=5)
        
        self.file_path_var = ctk.StringVar(value="No file selected")
        self.file_label = ctk.CTkLabel(file_frame, textvariable=self.file_path_var,
                                       anchor="w")
        self.file_label.pack(side="left", fill="x", expand=True, padx=5)
        
        self.select_btn = ctk.CTkButton(file_frame, text="Select Image", 
                                        command=self.select_image, width=100)
        self.select_btn.pack(side="right", padx=5)
        
        # Image display
        self.image_frame = ctk.CTkFrame(left_panel, height=300)
        self.image_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.image_frame.pack_propagate(False)
        
        self.image_label = ctk.CTkLabel(self.image_frame, text="No image loaded",
                                        font=ctk.CTkFont(size=14))
        self.image_label.pack(expand=True)
        
        # Process button
        self.process_btn = ctk.CTkButton(left_panel, text="Process Image", 
                                         command=self.process_image,
                                         height=40, font=ctk.CTkFont(size=16, weight="bold"),
                                         state="disabled")
        self.process_btn.pack(pady=10, padx=10, fill="x")
        
        # Right panel - Results
        right_panel = ctk.CTkFrame(main_container)
        right_panel.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        # Results title
        results_label = ctk.CTkLabel(right_panel, text="Model Comparison Results",
                                     font=ctk.CTkFont(size=20, weight="bold"))
        results_label.pack(pady=10)
        
        # Model status
        status_frame = ctk.CTkFrame(right_panel)
        status_frame.pack(fill="x", padx=10, pady=5)
        
        self.teacher_status = ctk.CTkLabel(status_frame, text="Teacher Model: Not Loaded",
                                           text_color="red")
        self.teacher_status.pack(side="left", padx=20)
        
        self.student_status = ctk.CTkLabel(status_frame, text="Student Model: Not Loaded",
                                           text_color="red")
        self.student_status.pack(side="right", padx=20)
        
        # Tabview for results
        self.tabview = ctk.CTkTabview(right_panel)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create tabs
        self.tabview.add("Predictions")
        self.tabview.add("Metrics")
        self.tabview.add("Visualization")
        
        
        # Predictions tab
        pred_frame = self.tabview.tab("Predictions")
        
        # Teacher predictions
        teacher_pred_frame = ctk.CTkFrame(pred_frame)
        teacher_pred_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(teacher_pred_frame, text="Teacher Model Predictions",
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        self.teacher_pred_text = ctk.CTkTextbox(teacher_pred_frame, height=150)
        self.teacher_pred_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Student predictions
        student_pred_frame = ctk.CTkFrame(pred_frame)
        student_pred_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        ctk.CTkLabel(student_pred_frame, text="Student Model Predictions",
                    font=ctk.CTkFont(size=16, weight="bold")).pack(pady=5)
        
        self.student_pred_text = ctk.CTkTextbox(student_pred_frame, height=150)
        self.student_pred_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Metrics tab
        metrics_frame = self.tabview.tab("Metrics")
        
        self.metrics_text = ctk.CTkTextbox(metrics_frame)
        self.metrics_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Visualization tab
        self.viz_frame = self.tabview.tab("Visualization")

        self.metrics_text = ctk.CTkTextbox(metrics_frame)
        self.metrics_text.pack(fill="both", expand=True, padx=5, pady=5)
    
    def load_models(self):
        """Load the trained teacher and student models"""
        try:
            # Check if saved models exist
            if os.path.exists('./models/teacher_model1010.h5'):
                self.teacher_model = tf.keras.models.load_model('./models/teacher_model1010.h5')
                self.teacher_status.configure(text="Teacher Model: Loaded", text_color="green")
            else:
                self.teacher_status.configure(text="Teacher Model: Not Found (save as teacher_model.h5)", 
                                            text_color="orange")
                

            if os.path.exists('./models/student_model1010.h5'):
                self.student_model = tf.keras.models.load_model('./models/student_model1010.h5')
                self.student_status.configure(text="Student Model: Loaded", text_color="green")
            else:
                self.student_status.configure(text="Student Model: Not Found (save as student_model.h5)", 
                                            text_color="orange")
                
            if not self.teacher_model or not self.student_model:
                messagebox.showwarning("Models Not Found", 
                                      "Please save your trained models as:\n"
                                      "- teacher_model.h5\n"
                                      "- student_model.h5\n"
                                      "in the same directory as this script.")
        except Exception as e:
            messagebox.showerror("Error Loading Models", f"Error: {str(e)}")
    
    def select_image(self):
        """Open file dialog to select an image"""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        
        if file_path:
            self.file_path_var.set(os.path.basename(file_path))
            self.load_and_display_image(file_path)
            self.process_btn.configure(state="normal")
    '''
    def load_and_display_image(self, file_path):
        """Load and display the selected image"""
        try:
            # Load image
            image = Image.open("your_image.png").convert("RGB")  # Force 3 channels
            img = image.resize((32, 32))  # Resize to CIFAR-10 size
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 32, 32, 3)
            self.current_image = img_array.copy()
            
            # Resize for display
            display_size = (250, 250)
            image_display = image.resize(display_size, Image.Resampling.LANCZOS)
            
            # Convert to CTkImage
            ctk_image = ctk.CTkImage(light_image=image_display, 
                                     dark_image=image_display,
                                     size=display_size)
            
            # Update label
            self.image_label.configure(image=ctk_image, text="")
            
            # Prepare image for model (32x32 for CIFAR-10)
            image_resized = image.resize((32, 32), Image.Resampling.LANCZOS)
            self.processed_image = np.array(image_resized).astype('float32') / 255.0
            self.processed_image = np.expand_dims(self.processed_image, axis=0)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    '''
    
    def load_and_preprocess_image(self, file_path):
        """Helper to load, convert, resize, and normalize an image for CIFAR models"""
        try:
            # Force RGB (3 channels) to avoid RGBA issues
            image = Image.open(file_path).convert("RGB")
            
            # Resize to CIFAR-10 model input size
            image_resized = image.resize((32, 32), Image.Resampling.LANCZOS)
            
            # Convert to numpy and normalize
            img_array = np.array(image_resized).astype("float32") / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # (1, 32, 32, 3)
            return image, img_array
        except Exception as e:
            messagebox.showerror("Error", f"Image preprocessing failed: {str(e)}")
            return None, None


    def load_and_display_image(self, file_path):
        """Load and display the selected image"""
        try:
            # Use helper to ensure correct format
            image, processed = self.load_and_preprocess_image(file_path)
            if image is None:
                return
            
            self.current_image = image.copy()
            self.processed_image = processed

            # Resize for display
            display_size = (250, 250)
            image_display = image.resize(display_size, Image.Resampling.LANCZOS)
            
            # Convert to CTkImage
            ctk_image = ctk.CTkImage(light_image=image_display, 
                                    dark_image=image_display,
                                    size=display_size)
            
            # Update label
            self.image_label.configure(image=ctk_image, text="")
            self.image_label.image = ctk_image  # prevent garbage collection

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")

    def process_image(self):
        """Process the image through both models"""
        if not self.teacher_model or not self.student_model:
            messagebox.showerror("Error", "Both models must be loaded first!")
            return
        
        if self.processed_image is None:
            messagebox.showerror("Error", "No image loaded!")
            return
        
        try:
            # Clear previous results
            self.teacher_pred_text.delete("1.0", "end")
            self.student_pred_text.delete("1.0", "end")
            self.metrics_text.delete("1.0", "end")
            
            # Teacher model prediction
            start_time = time.time()
            teacher_pred = self.teacher_model.predict(self.processed_image, verbose=0)
            teacher_time = time.time() - start_time
            
            # Student model prediction
            start_time = time.time()
            student_pred = self.student_model.predict(self.processed_image, verbose=0)
            student_time = time.time() - start_time
            
            # Get top 5 predictions for each model
            teacher_top5 = self.get_top_predictions(teacher_pred[0], 5)
            student_top5 = self.get_top_predictions(student_pred[0], 5)
            
            # Display teacher predictions
            self.teacher_pred_text.insert("1.0", "Top 5 Predictions:\n\n")
            for i, (idx, prob) in enumerate(teacher_top5, 1):
                self.teacher_pred_text.insert("end", 
                    f"{i}. {self.class_labels[idx]}: {prob:.2%}\n")
            
            # Display student predictions
            self.student_pred_text.insert("1.0", "Top 5 Predictions:\n\n")
            for i, (idx, prob) in enumerate(student_top5, 1):
                self.student_pred_text.insert("end", 
                    f"{i}. {self.class_labels[idx]}: {prob:.2%}\n")
            
            # Calculate and display metrics
            self.display_metrics(teacher_pred[0], student_pred[0], 
                               teacher_time, student_time)
            
            # Create visualization
            self.create_visualization(teacher_pred[0], student_pred[0])
            
        except Exception as e:
            messagebox.showerror("Error", f"Processing failed: {str(e)}")
    
    def get_top_predictions(self, probs, k=5):
        """Get top k predictions"""
        top_indices = np.argsort(probs)[-k:][::-1]
        return [(idx, probs[idx]) for idx in top_indices]
    
    def display_metrics(self, teacher_pred, student_pred, teacher_time, student_time):
        """Display comparison metrics"""
        metrics_text = "="*50 + "\n"
        metrics_text += "PERFORMANCE METRICS\n"
        metrics_text += "="*50 + "\n\n"
        
        # Inference time
        metrics_text += f"Inference Time:\n"
        metrics_text += f"  Teacher: {teacher_time*1000:.2f} ms\n"
        metrics_text += f"  Student: {student_time*1000:.2f} ms\n"
        metrics_text += f"  Speedup: {teacher_time/student_time:.2f}x\n\n"
        
        # Confidence metrics
        teacher_confidence = np.max(teacher_pred)
        student_confidence = np.max(student_pred)
        metrics_text += f"Max Confidence:\n"
        metrics_text += f"  Teacher: {teacher_confidence:.2%}\n"
        metrics_text += f"  Student: {student_confidence:.2%}\n\n"
        
        # Entropy (uncertainty)
        teacher_entropy = -np.sum(teacher_pred * np.log(teacher_pred + 1e-10))
        student_entropy = -np.sum(student_pred * np.log(student_pred + 1e-10))
        metrics_text += f"Prediction Entropy (lower = more certain):\n"
        metrics_text += f"  Teacher: {teacher_entropy:.4f}\n"
        metrics_text += f"  Student: {student_entropy:.4f}\n\n"
        
        # Agreement
        teacher_class = np.argmax(teacher_pred)
        student_class = np.argmax(student_pred)
        agreement = "Yes" if teacher_class == student_class else "No"
        metrics_text += f"Models Agree: {agreement}\n"
        if agreement == "Yes":
            metrics_text += f"  Predicted Class: {self.class_labels[teacher_class]}\n\n"
        else:
            metrics_text += f"  Teacher: {self.class_labels[teacher_class]}\n"
            metrics_text += f"  Student: {self.class_labels[student_class]}\n\n"
        
        # KL Divergence (similarity of distributions)
        kl_div = np.sum(teacher_pred * np.log(teacher_pred / (student_pred + 1e-10) + 1e-10))
        metrics_text += f"KL Divergence (Teacher||Student): {kl_div:.4f}\n"
        metrics_text += "  (lower = more similar predictions)\n\n"
        
        # Model size comparison (if available)
        try:
            teacher_params = self.teacher_model.count_params()
            student_params = self.student_model.count_params()
            metrics_text += f"Model Parameters:\n"
            metrics_text += f"  Teacher: {teacher_params:,}\n"
            metrics_text += f"  Student: {student_params:,}\n"
            metrics_text += f"  Compression Ratio: {teacher_params/student_params:.2f}x\n"
        except:
            pass
        
        self.metrics_text.insert("1.0", metrics_text)
    
    def create_visualization(self, teacher_pred, student_pred):
        """Create visualization comparing predictions"""
        # Clear previous plots
        for widget in self.viz_frame.winfo_children():
            widget.destroy()
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.patch.set_facecolor('#212121')
        
        # Teacher model bar chart
        ax1.bar(self.class_labels, teacher_pred, color='#3498db')
        ax1.set_title('Teacher Model Predictions', color='white', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Classes', color='white')
        ax1.set_ylabel('Probability', color='white')
        ax1.set_ylim([0, 1])
        ax1.tick_params(axis='x', rotation=45, colors='white')
        ax1.tick_params(axis='y', colors='white')
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#2b2b2b')
        
        # Student model bar chart
        ax2.bar(self.class_labels, student_pred, color='#2ecc71')
        ax2.set_title('Student Model Predictions', color='white', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Classes', color='white')
        ax2.set_ylabel('Probability', color='white')
        ax2.set_ylim([0, 1])
        ax2.tick_params(axis='x', rotation=45, colors='white')
        ax2.tick_params(axis='y', colors='white')
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#2b2b2b')
        
        plt.tight_layout()
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    # Note: Before running this app, save your trained models:
    # teacher.save('teacher_model.h5')
    # student.save('student_model.h5')
    
    app = ModelComparisonApp()
    app.mainloop()