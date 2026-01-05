# Creative Generation Agent - Code Organization

This directory contains the refactored Creative Generation Agent skill with Python code extracted into modular, reusable files.

## Directory Structure

```
creative-generation-agent/
├── README.md                      # This file
├── SKILL.md                       # Main skill documentation
├── examples/                      # Example implementations
│   ├── music_generation.py        # Music composition and audio synthesis
│   ├── meme_generator.py          # Image and text-based meme generation
│   ├── podcast_producer.py        # Podcast scripting and audio production
│   ├── image_generation.py        # Diffusion model image generation
│   └── style_transfer.py          # Neural style transfer
└── scripts/                       # Utility modules
    ├── creative_quality_assessment.py  # Content quality evaluation
    ├── audio_effects.py           # Audio effect processing
    └── content_moderation.py      # Safety and compliance filtering
```

## Quick Start Guide

### Installation

```bash
# Install required dependencies
pip install music21 numpy scipy pillow diffusers torch opencv-python
```

### Music Generation

Generate melodies and synthesize audio:

```python
from examples.music_generation import MusicGenerationAgent, AudioSynthesisAgent

# Create melody from seed notes
agent = MusicGenerationAgent()
melody = agent.generate_melody(
    seed_notes=[("C4", 1), ("E4", 1), ("G4", 1)],
    length=32,
    temperature=0.8
)

# Synthesize audio
synth = AudioSynthesisAgent()
audio = synth.synthesize_from_midi(midi_data)
audio = synth.add_effects(audio, effect_type="reverb")
synth.save_audio(audio, "output.wav")
```

### Meme Generation

Create memes with captions and text:

```python
from examples.meme_generator import MemeGenerationAgent, TextMemeGenerator

# Image-based meme
meme_agent = MemeGenerationAgent()
meme = meme_agent.generate_meme(topic="AI agents", meme_template="drake")
meme.save("meme.png")

# Text-based meme
text_gen = TextMemeGenerator()
joke = text_gen.generate_text_meme(topic="Python", format_type="joke")
print(joke)
```

### Podcast Generation

Create podcast scripts and produce audio:

```python
from examples.podcast_producer import PodcastScriptGenerator, PodcastAudioProducer

# Generate script
generator = PodcastScriptGenerator()
episode = generator.generate_episode(
    topic="The Future of AI",
    duration_minutes=30,
    num_hosts=2
)

# Produce audio
producer = PodcastAudioProducer()
audio = producer.produce_podcast(episode["script"])
```

### Image Generation

Generate images from text prompts:

```python
from examples.image_generation import ImageGenerationAgent

agent = ImageGenerationAgent()
image = agent.generate_image(
    prompt="A futuristic city with neon lights",
    style="cyberpunk"
)
image.save("generated.png")

# Create variations
variations = agent.generate_variations(image, num_variations=4)
```

### Style Transfer

Apply artistic styles to images:

```python
from examples.style_transfer import StyleTransferAgent

agent = StyleTransferAgent()
stylized = agent.transfer_style(
    content_image="photo.jpg",
    style_image="monet.jpg"
)
```

## Utilities

### Quality Assessment

Evaluate content quality:

```python
from scripts.creative_quality_assessment import CreativeQualityAssessor

assessor = CreativeQualityAssessor()

# Assess different content types
music_score = assessor.assess_content_quality(audio, content_type="music")
meme_score = assessor.assess_content_quality(meme, content_type="meme")
image_score = assessor.assess_content_quality(image, content_type="image")

print(f"Overall score: {music_score['overall_score']}")
print(f"Metrics: {music_score['metrics']}")
```

### Audio Effects

Apply professional audio effects:

```python
from scripts.audio_effects import AudioEffects

effects = AudioEffects(sample_rate=44100)

# Apply various effects
audio = effects.reverb(audio, room_size=0.5, delay_ms=50)
audio = effects.compression(audio, threshold=0.5, ratio=4)
audio = effects.fade_in(audio, duration_ms=1000)

# Mix multiple tracks
mixed = effects.mix(track1, track2, track3, volumes=[1.0, 0.5, 0.3])
```

### Content Moderation

Filter and validate generated content:

```python
from scripts.content_moderation import ContentModerator

moderator = ContentModerator()

# Moderate different content types
text_result = moderator.moderate_content(text, content_type="text", strict=False)
image_result = moderator.moderate_content(image, content_type="image")
meme_result = moderator.moderate_content(meme, content_type="meme", strict=True)

if text_result["passed"]:
    print("Content approved")
else:
    print(f"Issues: {text_result['issues']}")

# Generate moderation report
report = moderator.get_moderation_report(text_result, format="text")
print(report)
```

## Module Overview

### examples/music_generation.py

**Classes:**
- `MusicGenerationAgent` - Symbolic music generation using continuation
- `AudioSynthesisAgent` - Audio waveform synthesis with effects

**Key Methods:**
- `generate_melody()` - Generate melody continuation
- `generate_full_composition()` - Create complete musical piece
- `synthesize_from_midi()` - Convert MIDI to audio
- `add_effects()` - Apply audio effects (reverb, chorus, delay)

### examples/meme_generator.py

**Classes:**
- `MemeGenerationAgent` - Image-based meme generation
- `TextMemeGenerator` - Text-only meme generation

**Key Methods:**
- `generate_meme()` - Create meme with caption
- `generate_caption()` - Generate funny captions
- `apply_caption_to_template()` - Add text to image
- `generate_text_meme()` - Create text memes in various formats

### examples/podcast_producer.py

**Classes:**
- `PodcastScriptGenerator` - Generate podcast scripts
- `PodcastAudioProducer` - Produce audio from scripts

**Key Methods:**
- `generate_episode()` - Create complete episode
- `generate_script()` - Generate script with structure
- `generate_content_segments()` - Create discussion segments
- `produce_podcast()` - Convert script to audio

### examples/image_generation.py

**Classes:**
- `ImageGenerationAgent` - Text-to-image generation using diffusion models

**Key Methods:**
- `generate_image()` - Create image from prompt
- `enhance_prompt()` - Add style modifiers to prompts
- `generate_variations()` - Create image variations

### examples/style_transfer.py

**Classes:**
- `StyleTransferAgent` - Neural style transfer

**Key Methods:**
- `transfer_style()` - Apply style from one image to another
- `preprocess_image()` - Prepare image for processing
- `postprocess_image()` - Convert output to displayable format

### scripts/creative_quality_assessment.py

**Classes:**
- `CreativeQualityAssessor` - Quality evaluation for all content types

**Quality Metrics:**
- Music: coherence, variety, rhythm, harmony
- Meme: humor, clarity, originality, relatability
- Image: clarity, coherence, aesthetic, technical

### scripts/audio_effects.py

**Classes:**
- `AudioEffects` - Professional audio effect processing

**Available Effects:**
- Reverb - Spatial depth and ambience
- Echo - Repetition with decay
- Chorus - Pitch modulation
- Delay - Time-based effect
- Compression - Dynamic range control
- Normalization - Level adjustment
- Fade In/Out - Smooth transitions
- Equalization - Frequency adjustment
- Mix - Combine multiple tracks

### scripts/content_moderation.py

**Classes:**
- `ContentModerator` - Content safety and compliance checking

**Moderation Features:**
- Inappropriate language detection
- Bias detection
- Factual accuracy verification
- NSFW content detection
- Violence detection
- Copyright checking
- Hate speech detection
- Offensive content detection

## Best Practices

### Temperature Control

For creative content:
- **High temperature (0.7-0.9)**: More creative and diverse outputs
- **Medium temperature (0.5-0.7)**: Balanced creativity and coherence
- **Low temperature (0.1-0.3)**: More consistent and predictable

### Audio Processing

1. **Levels**: Keep peaks below -3dB for headroom
2. **Effects Chain**: Apply compression before EQ
3. **Mixing**: Use compression and limiting on master bus
4. **Normalization**: Normalize after all effects

### Content Generation

1. Start with specific prompts
2. Refine through iteration
3. Assess quality systematically
4. Moderate safety-critical content
5. Version successful parameters

### Quality Metrics

- **Music**: Aim for coherence > 0.7 and variety > 0.6
- **Memes**: Humor score > 0.6, clarity > 0.7
- **Images**: Clarity > 0.8, aesthetic > 0.7

## Integration Examples

### Complete Music Production Pipeline

```python
from examples.music_generation import MusicGenerationAgent, AudioSynthesisAgent
from scripts.audio_effects import AudioEffects
from scripts.creative_quality_assessment import CreativeQualityAssessor

# Generate music
music_agent = MusicGenerationAgent()
composition = music_agent.generate_full_composition()

# Synthesize to audio
synth = AudioSynthesisAgent()
audio = synth.synthesize_from_midi(composition)

# Apply effects
effects = AudioEffects()
audio = effects.reverb(audio)
audio = effects.compression(audio)
audio = effects.normalize(audio)

# Assess quality
assessor = CreativeQualityAssessor()
quality = assessor.assess_content_quality(audio, content_type="music")

synth.save_audio(audio, "final_track.wav")
```

### Safe Content Generation

```python
from examples.meme_generator import MemeGenerationAgent
from scripts.content_moderation import ContentModerator

# Generate meme
agent = MemeGenerationAgent()
meme = agent.generate_meme(topic="AI")

# Moderate content
moderator = ContentModerator()
result = moderator.moderate_content(meme, content_type="meme", strict=True)

if result["passed"]:
    meme.save("approved_meme.png")
else:
    print(f"Content rejected: {result['issues']}")
```

## Performance Considerations

- Music generation: 30-60 seconds for 32-bar composition
- Image generation: 30-120 seconds depending on inference steps
- Audio synthesis: Real-time processing possible
- Quality assessment: < 1 second per item

## Extending the Code

To add new features:

1. **Add effects**: Extend `AudioEffects` class
2. **Add quality metrics**: Extend `CreativeQualityAssessor`
3. **Add generation models**: Create new classes in `examples/`
4. **Add moderation rules**: Extend `ContentModerator`

## Requirements

- Python 3.8+
- numpy
- scipy
- pillow
- music21
- diffusers
- torch
- opencv-python (cv2)

## References

- [Music21 Documentation](https://web.mit.edu/music21/)
- [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- [Audio Effects Guide](https://en.wikipedia.org/wiki/Audio_signal_processing)

## License

Part of the AI Agent Skills project
