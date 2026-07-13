"""OwnVoice: train a LoRA voice adapter for pocket-tts, own the resulting model.

OwnVoice wraps kyutai-labs/pocket-tts (MIT-licensed, CPU-capable local
text-to-speech) with a LoRA fine-tuning workflow. The output is an adapter
file you keep and run yourself, not an API subscription.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
