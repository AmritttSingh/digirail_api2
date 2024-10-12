import os
import pickle
from utils.camera import capture_image
from utils.deepface_helpers import generate_embedding, find_best_match

def recognize_face():
    # Check if face embeddings exist
    if not os.path.exists('data/face_embeddings.pkl'):
        return {"message": "No faces have been registered yet.", "status": "error"}

    # Load embeddings from file
    try:
        with open('data/face_embeddings.pkl', 'rb') as f:
            face_embeddings = pickle.load(f)
    except Exception as e:
        return {"message": f"Error loading face embeddings: {e}", "status": "error"}

    # Capture image from the camera
    frame = capture_image()

    if frame is not None:
        # Generate face embedding for captured image
        current_embedding = generate_embedding(frame)
        if current_embedding is None:
            return {"message": "Failed to generate embedding for captured image.", "status": "error"}

        # Find best match in existing embeddings
        matched_user_id, similarity = find_best_match(current_embedding, face_embeddings)

        if matched_user_id:
            user_name = face_embeddings.get(matched_user_id, {}).get('name', 'Unknown User')
            return {"message": f"Match found! User: {user_name} (ID: {matched_user_id})", "similarity": similarity, "status": "success"}
        else:
            return {"message": "No match found.", "status": "error"}
    else:
        return {"message": "Failed to capture image for recognition.", "status": "error"}
