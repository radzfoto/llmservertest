import requests
import json

# Ollama server URL (default)
OLLAMA_SERVER_URL = "http://localhost:11434/api/v1/generate"

def generate_text_ollama(model, prompt, context="", max_tokens=1000, temperature=0.7, stream=False):  # Added model parameter
    """Sends a request to the Ollama server and returns the generated text."""

    payload = {
        "model": model,  # Specify the Ollama model
        "prompt": prompt,
        "context": context,  # Optional context
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream    # Whether to stream responses (usually False)
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(OLLAMA_SERVER_URL, data=json.dumps(payload), headers=headers, stream=stream)
        response.raise_for_status()

        if stream:  # Handle streaming responses (if stream=True)
            for chunk in response.iter_lines():
                if chunk:
                    decoded_chunk = chunk.decode('utf-8')  # Decode bytes to string
                    try:
                        json_chunk = json.loads(decoded_chunk[6:]) # remove "data: " prefix and parse
                        if 'response' in json_chunk:
                            yield json_chunk['response']
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON chunk: {e}. Chunk: {decoded_chunk}")
                        continue
        else:
            response_json = response.json()
            reply = response_json.get("response") # Ollama uses 'response' key
            return reply

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with the Ollama server: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}. Response text: {response.text}")
        return None
    except KeyError as e:
        print(f"Error extracting 'response' from JSON: {e}. Response JSON: {response_json}")
        return None


if __name__ == "__main__":
    model_name = "llama3.2-3b"  # Replace with your desired Ollama model
    prompt = "Write a short story about a robot learning to feel emotions."
    context = "The robot's name is Unit 734. It lives in a futuristic city."

    generated_text = generate_text_ollama(model_name, 
                                          prompt, 
                                          context
                                          max_tokens=1000, 
                                          temperature=0.7, 
                                          stream=False)

    if generated_text:
        print("Generated Text:\n", generated_text)

        story = generated_text

        # Example of saving to a file:
        # with open("robot_story.txt", "w") as f:
        #     f.write(story)

    # Example of streaming:
    # for chunk in generate_text_ollama(model_name, prompt, context, stream=True):
    #     print(chunk, end="", flush=True) # Print each chunk as it arrives
