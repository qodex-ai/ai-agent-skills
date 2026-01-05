"""
Image Generation Agent
Generates images using diffusion models with prompt engineering and style control.
"""

from diffusers import StableDiffusionPipeline
import torch


class ImageGenerationAgent:
    """Generates images from text prompts using Stable Diffusion."""

    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)

    def generate_image(self, prompt, num_inference_steps=50, guidance_scale=7.5, style="realistic"):
        """Generate image from text prompt.

        Args:
            prompt: Text description of image
            num_inference_steps: Number of diffusion steps
            guidance_scale: Classifier-free guidance scale
            style: Artistic style to apply

        Returns:
            PIL.Image.Image: Generated image
        """
        # Enhance prompt with style
        enhanced_prompt = self.enhance_prompt(prompt, style)

        # Generate image
        image = self.pipeline(
            enhanced_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=512,
            width=512
        ).images[0]

        return image

    def enhance_prompt(self, prompt, style="realistic"):
        """Add style-specific enhancements to prompt.

        Args:
            prompt: Base prompt
            style: Artistic style

        Returns:
            str: Enhanced prompt
        """
        style_modifiers = {
            "realistic": "photorealistic, high quality, detailed, 8k",
            "anime": "anime style, manga art, beautiful animation",
            "oil_painting": "oil painting, renaissance, masterpiece",
            "cyberpunk": "cyberpunk, neon, futuristic, digital art",
            "watercolor": "watercolor painting, soft colors, artistic"
        }

        modifier = style_modifiers.get(style, "high quality")
        return f"{prompt}, {modifier}"

    def generate_variations(self, image, num_variations=4):
        """Generate variations of an existing image.

        Args:
            image: Seed image
            num_variations: Number of variations to create

        Returns:
            list: List of variation images
        """
        variations = []

        # Use image as seed
        seed_embedding = self.encode_image(image)

        for _ in range(num_variations):
            variation = self.pipeline(
                image=image,
                num_inference_steps=30,
                strength=0.7  # Strength of modification
            ).images[0]

            variations.append(variation)

        return variations

    def encode_image(self, image):
        """Encode image to latent space.

        Args:
            image: Image to encode

        Returns:
            torch.Tensor: Latent encoding
        """
        pass
