from deepface import DeepFace
import cv2
from sklearn.metrics.pairwise import cosine_similarity

def generate_embedding(frame):
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        embedding = DeepFace.represent(img_path=rgb_frame, model_name='Facenet', enforce_detection=True, align=True)[0]["embedding"]
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def find_best_match(current_embedding, face_embeddings, threshold=0.7):
    max_similarity = -1
    matched_user_id = None

    for user_id, data in face_embeddings.items():
        stored_embedding = data['embedding']
        similarity = cosine_similarity([stored_embedding], [current_embedding])[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            matched_user_id = user_id

    if max_similarity > threshold:
        return matched_user_id, max_similarity
    return None, None
