"""
Meme Generation Agent
Generates memes through image captioning, text-based formats, and style adaptation.
"""

from PIL import Image, ImageDraw, ImageFont
import textwrap


class MemeGenerationAgent:
    """Generates image-based memes with AI captions."""

    def __init__(self):
        self.image_generator = self.load_image_generator()
        self.caption_generator = self.load_caption_generator()

    def generate_meme(self, topic, meme_template="drake"):
        """Generate complete meme with caption and template.

        Args:
            topic: Topic for meme generation
            meme_template: Template name to use

        Returns:
            PIL.Image.Image: Generated meme
        """
        # Generate caption
        caption = self.generate_caption(topic)

        # Get meme template image
        template_image = self.get_template(meme_template)

        # Apply text to image
        meme = self.apply_caption_to_template(template_image, caption)

        return meme

    def generate_caption(self, topic, style="humorous"):
        """Generate meme caption for given topic.

        Args:
            topic: Topic for caption
            style: Style of humor

        Returns:
            list: List of caption lines
        """
        prompt = f"Generate a funny two-line meme caption about {topic} in {style} style"
        caption = self.caption_generator.generate(prompt, max_tokens=100)
        return caption.strip().split('\n')[:2]

    def get_template(self, template_name):
        """Get meme template image.

        Args:
            template_name: Name of template

        Returns:
            PIL.Image.Image: Template image
        """
        templates = {
            "drake": "templates/drake.jpg",
            "loss": "templates/loss.jpg",
            "expanding_brain": "templates/expanding_brain.jpg",
            "wojak": "templates/wojak.jpg"
        }
        return Image.open(templates.get(template_name, templates["drake"]))

    def apply_caption_to_template(self, image, captions):
        """Apply text captions to meme template.

        Args:
            image: Template image
            captions: List of caption texts

        Returns:
            PIL.Image.Image: Image with captions
        """
        draw = ImageDraw.Draw(image)
        font = self.get_font(size=40)

        positions = self.calculate_caption_positions(image, len(captions))

        for i, caption in enumerate(captions):
            text = self.wrap_text(caption, 40)
            self.draw_text_with_outline(draw, positions[i], text, font)

        return image

    def draw_text_with_outline(self, draw, position, text, font, outline_width=2):
        """Draw text with outline for visibility on images.

        Args:
            draw: PIL ImageDraw object
            position: Text position
            text: Text to draw
            font: Font to use
            outline_width: Width of outline
        """
        x, y = position

        # Draw outline
        for adj_x in range(-outline_width, outline_width + 1):
            for adj_y in range(-outline_width, outline_width + 1):
                draw.text((x + adj_x, y + adj_y), text, fill='black', font=font)

        # Draw text
        draw.text(position, text, fill='white', font=font)

    def wrap_text(self, text, width):
        """Wrap text to specified width.

        Args:
            text: Text to wrap
            width: Maximum width

        Returns:
            str: Wrapped text
        """
        return '\n'.join(textwrap.wrap(text, width=width))

    def calculate_caption_positions(self, image, num_captions):
        """Calculate text positions for captions.

        Args:
            image: Image to position text on
            num_captions: Number of captions

        Returns:
            list: List of (x, y) positions
        """
        width, height = image.size
        positions = []

        if num_captions == 1:
            positions.append((20, 20))
        elif num_captions == 2:
            positions.append((20, 20))
            positions.append((20, height - 100))

        return positions

    def get_font(self, size=40):
        """Get font for text rendering.

        Args:
            size: Font size

        Returns:
            PIL.ImageFont: Font object
        """
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except Exception:
            return ImageFont.load_default()

    def load_image_generator(self):
        """Load image generation model."""
        pass

    def load_caption_generator(self):
        """Load caption generation model."""
        pass


class TextMemeGenerator:
    """Generates text-based memes in various formats."""

    def __init__(self):
        self.joke_engine = self.load_joke_engine()
        self.format_templates = self.load_format_templates()

    def generate_text_meme(self, topic, format_type="joke"):
        """Generate text-only meme in specified format.

        Args:
            topic: Topic for meme
            format_type: Type of meme format

        Returns:
            str: Generated meme text
        """
        if format_type == "joke":
            return self.generate_joke_meme(topic)
        elif format_type == "unexpected":
            return self.generate_unexpected_meme(topic)
        elif format_type == "deep":
            return self.generate_deep_meme(topic)

    def generate_joke_meme(self, topic):
        """Generate joke-style meme.

        Args:
            topic: Topic for joke

        Returns:
            str: Joke meme
        """
        setup = self.joke_engine.generate_setup(topic)
        punchline = self.joke_engine.generate_punchline(setup)

        return f"Setup: {setup}\nPunchline: {punchline}"

    def generate_unexpected_meme(self, topic):
        """Generate meme with unexpected twist.

        Args:
            topic: Topic for meme

        Returns:
            str: Unexpected meme
        """
        prompt = f"Create a meme with an unexpected twist about {topic}"
        meme_text = self.joke_engine.generate(prompt)
        return meme_text

    def generate_deep_meme(self, topic):
        """Generate deep/philosophical meme.

        Args:
            topic: Topic for meme

        Returns:
            str: Deep meme
        """
        prompt = f"Generate a deep, philosophical meme about {topic}"
        meme_text = self.joke_engine.generate(prompt)
        return meme_text

    def load_joke_engine(self):
        """Load joke/text generation engine."""
        pass

    def load_format_templates(self):
        """Load meme format templates."""
        pass
