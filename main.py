import models.face_recognition as fr
import models.face_registration as reg
import os
import pickle

def list_registered_users():
    if not os.path.exists('data/face_embeddings.pkl'):
        print("No faces have been registered yet.")
        return
    
    with open('data/face_embeddings.pkl', 'rb') as f:
        face_embeddings = pickle.load(f)
    
    if len(face_embeddings) == 0:
        print("No registered users found.")
    else:
        print(f"Total number of registered users: {len(face_embeddings)}")
        for user_id, data in face_embeddings.items():
            print(f"User ID: {user_id}, Name: {data['name']}")

def delete_user(user_id):
    confirm = input(f"Are you sure you want to delete user {user_id}? (y/n): ").strip().lower()
    if confirm == 'y':
        if reg.delete_user(user_id):
            print(f"User with ID {user_id} has been deleted.")
        else:
            print(f"No user found with ID {user_id}.")
    else:
        print("Delete operation canceled.")

def main():
    print("Welcome to DigiRail Face Recognition System")
    try:
        while True:
            print("\nOptions:")
            print("  'r' - Register a new face")
            print("  'c' - Check/Recognize a face")
            print("  'l' - List registered users")
            print("  'd' - Delete a user")
            print("  'q' - Quit the application")
            choice = input("Enter your choice: ").strip().lower()

            if choice == 'r':
                user_id = input("Enter User ID: ").strip()
                user_name = input("Enter User Name: ").strip()
                reg.register_face(user_id, user_name)
            elif choice == 'c':
                result = fr.recognize_face()
                print(result.get("message"))
            elif choice == 'l':
                list_registered_users()
            elif choice == 'd':
                user_id = input("Enter the User ID to delete: ").strip()
                delete_user(user_id)
            elif choice == 'q':
                print("Exiting DigiRail system. Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")
    except KeyboardInterrupt:
        print("\nExiting DigiRail system. Goodbye!")

if __name__ == "__main__":
    main()
