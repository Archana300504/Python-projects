import os
import google.generativeai as genai
def setup_model(api_key, model_name):
    os.environ['API_KEY'] = api_key
    if api_key and model_name:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        if hasattr(model, 'generate_content'):
            return model
        else:
            print("Model initialization failed.")
            return None
    else:
        print("API key or model name is missing.")
        return None


def generate_travel_guide(model, destination, duration, interests):
    if model is None:
        return "Model not initialized."
    
    prompt = f"Create a travel guide for {destination}. The trip duration is {duration}. The traveler's interests include {interests}."
    
    if hasattr(model, 'generate_content'):
        response = model.generate_content(prompt)
        if hasattr(response, 'text'):
            return response.text
        else:
            return "Unexpected response format."
    else:
        return "Model does not support content generation."

def main():
    
    api_key = 'AIzaSyBv-Fk-s_KCS0dJhRspi-WyfxYkEtLGEb8'
    model_name = 'gemini-1.5-flash'
    model = setup_model(api_key, model_name)
    
    if model is None:
        print("Failed to initialize the model.")
        return

    
    destination = input("Enter the travel destination: ").strip()
    duration = input("Enter the trip duration (e.g., 7 days): ").strip()
    interests = input("Enter your interests (e.g., museums, hiking, food): ").strip()

    response = generate_travel_guide(model, destination, duration, interests)
    
    print("\nGenerated Travel Guide:")
    print(response)
    
    filename = f'travel_guide_{destination.replace(" ", "_")}.txt'
    
    if response and response != "Model not initialized." and response != "Unexpected response format.":
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(response)
        print(f"Travel guide saved to '{filename}'")
    else:
        print("No content to save.")

main()

