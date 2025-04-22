# Dia-1.6B: Text-to-Speech Model for Dialogue Generation

<div align="center">
  <img src="images/banner.png" width="600px"/>
</div>

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6+](https://img.shields.io/badge/pytorch-2.6+-blue.svg)](https://pytorch.org/get-started/locally/)

> **Note**: This is a fork of [nari-labs/Dia-1.6B](https://github.com/nari-labs/Dia-1.6B) with additional features including safetensors support and packaging improvements.

## About Dia

Dia is a 1.6 billion parameter text-to-speech model designed for generating realistic dialogue audio with multiple distinct voices. It was developed by Nari Labs and released as an open-source project, allowing complete control over scripts and voices.

### Key capabilities:

- Generate realistic dialogue between multiple speakers
- Control speaker identity with simple tags in the prompt text
- Voice cloning from sample audio
- Supports both original .pth model weights and safetensors format

## What's New in This Fork

This fork enhances the original Dia model with:

1. **Safetensors Support**: Added support for loading models in the safetensors format, which offers:
   - Faster loading times
   - Improved security
   - Better compatibility across PyTorch versions
   - Support for BF16 precision
   - Reduced memory usage during loading

2. **Packaging Improvements**:
   - Properly configured pyproject.toml for pip installation
   - Relaxed dependency constraints for better compatibility
   - Fixed path handling for DAC model loading

3. **New Examples**:
   - Added safetensors_example.py demonstrating BF16 precision model usage

## Installation

You can install this package directly from GitHub:

```bash
pip install git+https://github.com/ttj/dia-1.6b-safetensors.git
```

### Requirements

- Python 3.10+
- PyTorch 2.6+
- CUDA (optional, but recommended for faster inference)

## Usage

### Basic Example

```python
import soundfile as sf
from dia.model import Dia

# Load the original model
model = Dia.from_pretrained("nari-labs/Dia-1.6B")

# Create a prompt with speaker tags
text = "[S1] Hello, how are you? [S2] I'm doing well, thank you! [S1] That's great to hear."

# Generate audio
output = model.generate(text)

# Save to file
sf.write("output.mp3", output, 44100)
```

### Using Safetensors Model with BF16 Precision

```python
import soundfile as sf
from dia.model import Dia

# Load the safetensors model with BF16 precision
model = Dia.from_pretrained(
    model_name="ttj/dia-1.6b-safetensors",
    dtype="bf16"
)

# Create a prompt with speaker tags
text = "[S1] Have you heard about the new safetensors version of Dia? [S2] Yes, it's more efficient and supports BF16 precision!"

# Generate audio
output = model.generate(text)

# Save to file
sf.write("safetensors_output.mp3", output, 44100)
```

### Voice Cloning

```python
import soundfile as sf
from dia.model import Dia

model = Dia.from_pretrained("nari-labs/Dia-1.6B")

# Define the transcript of your reference audio
clone_from_text = "[S1] This is an example of my voice. [S2] And this is another voice."
clone_from_audio = "reference_audio.mp3"  # Path to your reference audio

# New text to generate with the same voices
new_text = "[S1] I'm speaking with the cloned voice. [S2] Me too!"

# Generate using voice cloning
output = model.generate(
    clone_from_text + new_text,
    audio_prompt_path=clone_from_audio
)

sf.write("voice_clone_output.mp3", output, 44100)
```

## Advanced Parameters

```python
output = model.generate(
    text=your_text,
    temperature=1.3,    # Controls randomness (higher = more random)
    top_p=0.95,         # Controls diversity of word choices 
    cfg_scale=3.0,      # Controls how closely output follows the prompt
    max_tokens=None     # Limit generation length (None = use default)
)
```

## Credits

This model was originally created by [Nari Labs](https://github.com/nari-labs) as [Dia-1.6B](https://github.com/nari-labs/Dia-1.6B).

This fork adds safetensors support and packaging improvements to make the model more accessible and easier to use.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.