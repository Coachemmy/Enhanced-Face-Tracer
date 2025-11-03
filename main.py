# main.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
from pathlib import Path
from models.vit_loader import load_vit_checkpoint

class AAMSoftmax(nn.Module):
    """Angular Additive Margin Softmax for identity enhancement"""
    def __init__(self, feat_dim=512, num_classes=1000, margin=0.3, scale=30):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.weight)
        
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, x, label=None):
        # Normalize features and weights
        x_norm = F.normalize(x, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        
        cosine = F.linear(x_norm, w_norm)
        
        if label is None:
            return cosine
        
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output

class FaceTracerSystem:
    def __init__(self, model_checkpoint_path, aam_checkpoint_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        print("Loading FaceTracer system...")
        
        # Load the main ViT backbone
        self.backbone = load_vit_checkpoint(model_checkpoint_path, device=device)
        
        # Load AAMSoftmax head
        if aam_checkpoint_path and os.path.exists(aam_checkpoint_path):
            self.aam_head = self.load_aamsoftmax(aam_checkpoint_path)
            print("✓ AAMSoftmax loaded for identity enhancement")
        else:
            self.aam_head = None
            print("⚠ AAMSoftmax not found, using base features only")
        
        print("✓ FaceTracer system loaded successfully!")
    
    def load_aamsoftmax(self, aam_path):
        """Load AAMSoftmax weights"""
        checkpoint = torch.load(aam_path, map_location=self.device)
        aam = AAMSoftmax(feat_dim=512, num_classes=1000)
        
        if 'weight' in checkpoint:
            aam.weight.data = checkpoint['weight']
        else:
            aam.load_state_dict(checkpoint)
        
        aam.eval()
        return aam
    
    def preprocess_face(self, face_image, size=112):
        """Preprocess face image for FaceTracer"""
        # Resize to standard face recognition size
        face_resized = cv2.resize(face_image, (size, size))
        
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        face_tensor = torch.from_numpy(face_rgb).float() / 255.0
        face_tensor = face_tensor.permute(2, 0, 1)  # HWC to CHW
        
        # Face recognition normalization (standard for face models)
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        face_tensor = (face_tensor - mean) / std
        
        return face_tensor.unsqueeze(0).to(self.device)
    
    def extract_identity_features(self, face_image):
        """
        Extract source identity features using the full FaceTracer pipeline
        Implements the three modules from the paper:
        1. Identity Information Extraction (ViT backbone)
        2. Identity Information Disentanglement (built into ViT training)
        3. Identity Information Enhancement (AAMSoftmax)
        """
        preprocessed = self.preprocess_face(face_image)
        
        with torch.no_grad():
            # 1. Identity Information Extraction - get features from ViT
            features = self.backbone(preprocessed)  # [1, num_patches, 512]
            
            # Global average pooling for compact identity representation
            identity_embedding = features.mean(dim=1)  # [1, 512]
            
            # 2. Identity Information Disentanglement is handled by the trained ViT
            
            # 3. Identity Information Enhancement with AAMSoftmax
            if self.aam_head is not None:
                # Get enhanced discriminative features
                enhanced_logits = self.aam_head(identity_embedding)
                # Use the features before the final classification
                enhanced_features = F.normalize(identity_embedding, p=2, dim=1)
                return identity_embedding, enhanced_features, enhanced_logits
            
            return identity_embedding, None, None
    
    def detect_faces(self, image):
        """Detect faces in image using OpenCV Haar Cascade"""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def process_swapped_content(self, image_path):
        """
        Process face-swapped content to trace source identity
        Returns the source identity features that can be matched against a database
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image: {image_path}")
            return None
        
        faces = self.detect_faces(image)
        print(f"Found {len(faces)} faces in {Path(image_path).name}")
        
        results = []
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face_img = image[y:y+h, x:x+w]
            
            # Get source identity features (this is the key step)
            base_embedding, enhanced_features, enhanced_logits = self.extract_identity_features(face_img)
            
            result = {
                'face_bbox': (x, y, w, h),
                'base_embedding': base_embedding.cpu().numpy(),
                'enhanced_features': enhanced_features.cpu().numpy() if enhanced_features is not None else None,
                'enhanced_logits': enhanced_logits.cpu().numpy() if enhanced_logits is not None else None
            }
            
            results.append(result)
            
            print(f"  Face {i+1}: Base features {base_embedding.shape}", end="")
            if enhanced_features is not None:
                print(f" | Enhanced features {enhanced_features.shape}")
            else:
                print()
        
        return results
    
    def compute_similarity(self, embedding1, embedding2):
        """Compute cosine similarity between two embeddings"""
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Convert to numpy if they're tensors
        if isinstance(embedding1, torch.Tensor):
            embedding1 = embedding1.cpu().numpy()
        if isinstance(embedding2, torch.Tensor):
            embedding2 = embedding2.cpu().numpy()
        
        # Flatten and normalize
        emb1_flat = embedding1.flatten()
        emb2_flat = embedding2.flatten()
        
        emb1_flat = emb1_flat / (np.linalg.norm(emb1_flat) + 1e-8)
        emb2_flat = emb2_flat / (np.linalg.norm(emb2_flat) + 1e-8)
        
        similarity = np.dot(emb1_flat, emb2_flat)
        return float(similarity)
    
    def build_identity_gallery(self, reference_images_dir):
        """Build a gallery of known identities for matching"""
        gallery = {}
        ref_dir = Path(reference_images_dir)
        
        if ref_dir.exists():
            for img_path in ref_dir.glob("*.*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    results = self.process_swapped_content(str(img_path))
                    if results:
                        # Use the first face found as the reference
                        gallery[img_path.stem] = results[0]['base_embedding']
                        print(f"Added {img_path.stem} to identity gallery")
        
        return gallery
    
    def trace_source_identity(self, query_embedding, identity_gallery, similarity_threshold=0.6):
        """
        Trace the source identity by matching against known identities
        Returns the best match and similarity score
        """
        best_match = None
        best_similarity = 0
        
        for identity_name, gallery_embedding in identity_gallery.items():
            similarity = self.compute_similarity(query_embedding, gallery_embedding)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = identity_name
        
        # Only return match if above threshold
        if best_similarity >= similarity_threshold:
            return best_match, best_similarity
        else:
            return None, best_similarity

def main():
    # Initialize FaceTracer with both models
    model_path = "./checkpoints/HiRes/net.pth.tar"
    aam_path = "./checkpoints/HiRes/aamsoftmax.pth.tar"
    
    facetracer = FaceTracerSystem(model_path, aam_path)
    
    # Process test directory
    input_dir = Path("./input_videos/test")
    output_dir = Path("./output_videos")
    output_dir.mkdir(exist_ok=True)
    
    print("=== FaceTracer: Source Identity Tracing ===")
    print("This system extracts the SOURCE person's identity from face-swapped content")
    print("It can trace back to the original fraudster behind the swap\n")
    
    # Process all test images
    image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpeg"))
    
    if not image_files:
        print("No test images found in input_videos/test/")
        print("Please add some face-swapped images to test the system")
        return
    
    print(f"Found {len(image_files)} test images")
    
    # Process each image and extract source identities
    all_results = {}
    for image_file in image_files:
        print(f"\n🔍 Analyzing: {image_file.name}")
        print("=" * 50)
        
        results = facetracer.process_swapped_content(str(image_file))
        
        if results:
            all_results[image_file.name] = results
            
            # For demonstration, show the extracted features
            for i, result in enumerate(results):
                print(f"  Face {i+1}:")
                print(f"    - Base embedding: {result['base_embedding'].shape}")
                if result['enhanced_features'] is not None:
                    print(f"    - Enhanced features: {result['enhanced_features'].shape}")
                if result['enhanced_logits'] is not None:
                    print(f"    - AAMSoftmax logits: {result['enhanced_logits'].shape}")
    
    print(f"\n🎉 FaceTracer analysis completed!")
    print(f"Processed {len(all_results)} images with face-swapped content")
    print("\nThe extracted source identity features can now be used to:")
    print("  1. Match against known fraudster databases")
    print("  2. Link multiple swapped videos to the same source")
    print("  3. Provide evidence for fraud prevention")
    
    # Save results for further analysis
    if all_results:
        results_path = output_dir / "source_identity_features.npy"
        np.save(str(results_path), all_results)
        print(f"\n📁 Results saved to: {results_path}")

if __name__ == "__main__":
    import math  # Added for AAMSoftmax
    main()






















# # main.py -CODE FOR LOADING MY MODEL
# import torch
# from models.vit_loader import load_vit_checkpoint

# checkpoint_path = "./checkpoints/HiRes/net.pth.tar"
# model = load_vit_checkpoint(checkpoint_path, device='cpu')
# print("Model loaded successfully!")

# # Test with the exact size that matches the positional embeddings (108x108)
# print("\n=== Testing with exact size ===")
# try:
#     dummy_input = torch.randn(1, 3, 108, 108)  # 12x12 patches of size 9
#     with torch.no_grad():
#         output = model(dummy_input)
#         print(f"Input 108x108 -> Output: {output.shape}")
#         print("✓ Success! Model is working correctly.")
# except Exception as e:
#     print(f"Error: {e}")
#     print("Let me check which parameters we have...")
    
#     # Debug: show available block parameters
#     block_params = [key for key in model.original_keys if 'blocks.0' in key]
#     print(f"Block 0 parameters: {block_params}")

# print("\n=== Model ready! ===")