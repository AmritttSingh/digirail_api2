import os
import pickle
import numpy as np
from utils.camera import capture_image
from utils.deepface_helpers import generate_embedding

def register_face(user_id, user_name, num_images=5):
    embeddings = []

    for i in range(num_images):
        print(f"Capturing image {i+1} of {num_images}...")
        frame = capture_image()

        if frame is not None:
            embedding = generate_embedding(frame)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                print(f"Failed to generate embedding for image {i+1}.")
        else:
            print(f"Failed to capture image {i+1}")

    if len(embeddings) > 0:
        avg_embedding = np.mean(embeddings, axis=0)

        try:
            if os.path.exists('data/face_embeddings.pkl'):
                with open('data/face_embeddings.pkl', 'rb') as f:
                    face_embeddings = pickle.load(f)
            else:
                face_embeddings = {}
        except Exception as e:
            print(f"Error loading face embeddings: {e}")
            face_embeddings = {}

        face_embeddings[user_id] = {
            'name': user_name,
            'embedding': avg_embedding
        }

        try:
            with open('data/face_embeddings.pkl', 'wb') as f:
                pickle.dump(face_embeddings, f)
            print(f"Face registered successfully for user: {user_name} (ID: {user_id})")
        except Exception as e:
            print(f"Error saving face embeddings: {e}")
    else:
        print("Failed to register face. No valid embeddings captured.")

def delete_user(user_id):
    try:
        if not os.path.exists('data/face_embeddings.pkl'):
            return False

        with open('data/face_embeddings.pkl', 'rb') as f:
            face_embeddings = pickle.load(f)

        if user_id in face_embeddings:
            del face_embeddings[user_id]
            with open('data/face_embeddings.pkl', 'wb') as f:
                pickle.dump(face_embeddings, f)
            return True
    except Exception as e:
        print(f"Error deleting user: {e}")
    return False
