"""
Quick trial script for object detection + search.

Usage:
    python object_search_trial.py --image path/to/image.jpg --label apple

Steps:
1) Detect objects in the image (YOLO if available, otherwise mock).
2) Generate embeddings for detected objects.
3) Store them into object FAISS.
4) Run object search by label (if provided) and by embedding of the first detection.
5) Print results for inspection.
"""

import argparse
import sys
from pathlib import Path

# Ensure backend is on path
ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from backend.detection.object_detector import ObjectDetector
from backend.embeddings.embedding_manager import EmbeddingManager
from backend.vector_store.store_manager import VectorStoreManager


def main():
    parser = argparse.ArgumentParser(description="Trial object detect + search.")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--label", help="Optional label query for search (e.g., apple)")
    parser.add_argument("--top-k", type=int, default=5, help="Top K results to return")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[ERR] Image not found: {image_path}")
        sys.exit(1)

    print("=== Init components ===")
    detector = ObjectDetector()
    embed_mgr = EmbeddingManager()
    store = VectorStoreManager()

    print(f"=== Detecting objects in: {image_path} ===")
    detections = detector.detect([str(image_path)])
    print(f"Detections: {len(detections)}")
    for det in detections:
        print(f" - label={det.get('label')} conf={det.get('confidence')} bbox={det.get('bbox')}")

    if not detections:
        print("No detections; exiting.")
        return

    print("=== Generating embeddings for detections ===")
    embeddings = embed_mgr.generate_object_embeddings(detections)
    print(f"Embeddings generated: {len(embeddings)}")

    print("=== Storing objects into FAISS ===")
    ok = store.store_objects(embeddings, detections)
    print(f"Store success: {ok}, total objects in index: {store.faiss_objects.get_count()}")

    print("=== Search by label (if provided) ===")
    if args.label:
        label_results = store.search_objects(label_query=args.label, top_k=args.top_k)
        print(f"Label search '{args.label}' results: {len(label_results)}")
        for r in label_results:
            print(f" - image={r.get('image_path')} label={r.get('label')} conf={r.get('confidence'):.4f}")
    else:
        print("No label provided; skipping label search.")

    print("=== Search by embedding (first detection) ===")
    emb_results = store.search_objects(embeddings[0], top_k=args.top_k)
    print(f"Embedding search results: {len(emb_results)}")
    for r in emb_results:
        print(f" - image={r.get('image_path')} label={r.get('label')} conf={r.get('confidence'):.4f}")


if __name__ == "__main__":
    main()
