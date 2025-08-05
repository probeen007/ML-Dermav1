from data_analysis import analyze_dataset
from data_preprocessing import preprocess_data
from model_building import build_model
from model_training import train_model, fine_tune_model
from model_evaluation import evaluate_model

def main():
    # Define dataset paths
    train_dir = 'SkinDisease/Train'
    test_dir = 'SkinDisease/Test'

    # Step 1: Analyze dataset
    print("Analyzing dataset...")
    analyze_dataset(train_dir)

    # Step 2: Preprocess data
    print("Preprocessing data...")
    train_generator, validation_generator, test_generator = preprocess_data(train_dir, test_dir)

    # Step 3: Build the model
    print("Building the model...")
    model = build_model()

    # Step 4: Train the model
    print("Training the model...")
    history = train_model(model, train_generator, validation_generator)

    # Step 5: Fine-tune the model
    print("Fine-tuning the model...")
    history_fine = fine_tune_model(model, train_generator, validation_generator)

    # Step 6: Evaluate the model
    print("Evaluating the model...")
    evaluate_model(model, test_generator)

    # Step 7: Save the final model
    print("Saving the final model...")
    model.save('models/final_model.keras')
    print("Model saved successfully!")

if __name__ == "__main__":
    main()