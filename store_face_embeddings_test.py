from backend.detection.face_detector import FaceDetector
from backend.vector_store.store_manager import VectorStoreManager

def embed_and_store_faces(image_paths):
    detector = FaceDetector()
    vector_store = VectorStoreManager()

    detections = detector.detect(image_paths)
    if not detections:
        print("No faces detected.")
        return False

    stored = vector_store.store_faces(detections)
    print(f"Stored {len(detections)} face embeddings: {stored}")
    return stored

if __name__ == "__main__":
    images = [
        r"D:\BTP\new_ai_powered\new_repo_ai_powered\IMG-20250322-WA0044.jpg",
        r"D:\BTP\new_ai_powered\new_repo_ai_powered\IMG-20240921-WA0022.jpg",
        r"D:\BTP\new_ai_powered\new_repo_ai_powered\IMG-20240811-WA0030.jpg",
    ]
    embed_and_store_faces(images)
