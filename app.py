from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import List
import models.face_recognition as fr
import models.face_registration as reg
import numpy as np
import cv2
import io
from PIL import Image

app = FastAPI()

# Data models for request body
class RegisterRequest(BaseModel):
    user_id: str
    user_name: str

@app.get("/")
def read_root():
    return {"message": "Welcome to DigiRail Facial Recognition API"}

# Helper function to convert uploaded images into numpy arrays
def convert_image_to_np(image_file: UploadFile):
    try:
        image = Image.open(io.BytesIO(image_file.file.read()))
        image_np = np.array(image)
        return image_np
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

# Endpoint to register a new user
@app.post("/register_face")
def register_face(request: RegisterRequest):
    try:
        reg.register_face(request.user_id, request.user_name)
        return {"message": f"User {request.user_name} (ID: {request.user_id}) has been registered."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to recognize a face
@app.post("/recognize_face")
def recognize_face():
    try:
        result = fr.recognize_face()
        if result['status'] == 'success':
            return {"message": f"User: {result['user_name']} recognized with similarity {result['similarity']:.2f}"}
        else:
            return {"message": result['message']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to recognize face from an array of 5 images
@app.post("/recognize_face_multiple")
async def recognize_face_multiple(images: List[UploadFile] = File(...)):
    if len(images) != 5:
        raise HTTPException(status_code=400, detail="Exactly 5 images are required.")
    
    embeddings = []
    try:
        # Convert and process each image
        for image_file in images:
            image_np = convert_image_to_np(image_file)
            embedding = fr.generate_embedding(image_np)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                raise HTTPException(status_code=500, detail="Failed to generate embedding for an image.")

        # Take the average of the embeddings for all 5 images
        avg_embedding = np.mean(embeddings, axis=0)

        # Check against the database for best match
        matched_user_id, similarity = fr.find_best_match(avg_embedding)

        if matched_user_id:
            return {"message": f"User recognized: {matched_user_id}, similarity: {similarity:.2f}"}
        else:
            return {"message": "No match found."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during recognition: {str(e)}")

# Endpoint to list all registered users
@app.get("/users")
def list_users():
    try:
        users = reg.list_registered_users()
        return {"users": users}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to delete a user by user_id
@app.delete("/delete_user/{user_id}")
def delete_user(user_id: str):
    try:
        result = reg.delete_user(user_id)
        if result:
            return {"message": f"User {user_id} deleted successfully."}
        else:
            raise HTTPException(status_code=404, detail="User not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/recognize_faces_array")
async def recognize_faces_array(images: List[UploadFile] = File(...)):
    # Process the images (same as recognition logic)
    # Save images temporarily if needed
    results = []
    for image in images:
        # Process each image, generate embeddings, and find matches
        # Add recognition results for each image
        results.append({"image_name": image.filename, "status": "processed"})

    return {"message": "Images processed", "results": results}
