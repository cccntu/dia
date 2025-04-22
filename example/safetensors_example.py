import soundfile as sf
import torch

from dia.model import Dia


def main():
    """
    Example script demonstrating how to use the BF16 safetensors version of Dia model.
    
    This example shows how to:
    1. Load the model from the safetensors repository with BF16 precision
    2. Generate audio for a simple dialogue
    3. Save the resulting audio to a file
    
    Benefits of safetensors format:
    - Faster loading times
    - Improved security
    - Better compatibility across PyTorch versions
    - Support for specific dtype loading (like BF16)
    - Reduced memory usage during loading
    """
    print("Loading Dia model from safetensors repository with BF16 precision...")
    
    # Load the model from the safetensors repository with BF16 precision
    # The dtype="bf16" parameter is important to load the BF16 version of the model
    model = Dia.from_pretrained(
        model_name="ttj/dia-1.6b-safetensors",
        dtype="bf16"
    )
    
    print(f"Model loaded successfully. Using device: {model.device}")
    
    # Example dialogue prompt with two speakers
    # [S1] and [S2] are speaker tags that help the model generate different voices
    text = "[S1] Have you heard about the new safetensors version of Dia? [S2] Yes, it's more efficient and supports BF16 precision! [S1] That's amazing! How does it compare to the original? [S2] It loads faster and uses less memory."
    
    print("Generating audio from text...")
    
    # Generate audio with customized parameters
    output = model.generate(
        text=text,
        temperature=1.2,    # Controls randomness (higher = more random)
        top_p=0.9,          # Controls diversity of word choices
        cfg_scale=3.0       # Controls how closely output follows the prompt
    )
    
    # Save the output to an MP3 file
    output_file = "safetensors_example.mp3"
    sf.write(output_file, output, 44100)
    
    print(f"Audio saved to {output_file}")
    
    # Note: You can also use voice cloning as shown below (commented out)
    """
    # Voice Cloning Example:
    # 1. Create a source audio with transcript
    # clone_from_text = "[S1] This is an example of my voice. [S2] And this is another voice."
    # clone_from_audio = "source_audio.mp3"  # Path to your source audio
    # 
    # # 2. Create new text to generate with the same voices
    # new_text = "[S1] I'm speaking with the same voice. [S2] Me too!"
    # 
    # # 3. Generate using the source audio as a prompt
    # cloned_output = model.generate(
    #     clone_from_text + new_text,
    #     audio_prompt_path=clone_from_audio
    # )
    # 
    # sf.write("cloned_voices.mp3", cloned_output, 44100)
    """


if __name__ == "__main__":
    main()