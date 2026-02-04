import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io


class ImageClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None
        self._load_model()

    def _load_model(self):
        def load_model():
            model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
            model.eval()
            return model

        self.model = load_model()
        self.model.to(self.device)

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_bytes: bytes) -> dict:
        def process_image(img_bytes):
            image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
            input_tensor = self.preprocess(image)
            input_batch = input_tensor.unsqueeze(0)
            return input_batch.to(self.device)

        input_batch = process_image(image_bytes)

        with torch.no_grad():
            output = self.model(input_batch)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)

        with open('imagenet_classes.txt') as f:
            categories = [s.strip() for s in f.readlines()]

        top5_prob, top5_catid = torch.topk(probabilities, 5)

        results = []
        for i in range(top5_prob.size(0)):
            results.append({
                'class': categories[top5_catid[i]],
                'confidence': float(top5_prob[i])
            })

        return {
            'predictions': results,
            'top_prediction': {
                'class': categories[top5_catid[0]],
                'confidence': float(top5_prob[0])
            }
        }


_classifier_instance = None


def get_classifier() -> ImageClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ImageClassifier()
    return _classifier_instance
