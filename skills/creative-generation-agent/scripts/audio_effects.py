"""
Audio Effects Utilities
Provides various audio effect implementations for sound processing and enhancement.
"""

import numpy as np


class AudioEffects:
    """Provides common audio effects for music and podcast production."""

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def apply_effect(self, audio, effect_type="reverb", **kwargs):
        """Apply audio effect to signal.

        Args:
            audio: Audio samples
            effect_type: Type of effect to apply
            **kwargs: Effect-specific parameters

        Returns:
            np.ndarray: Processed audio
        """
        effects = {
            "reverb": self.reverb,
            "echo": self.echo,
            "chorus": self.chorus,
            "delay": self.delay,
            "compression": self.compression,
            "normalization": self.normalize,
            "fade_in": self.fade_in,
            "fade_out": self.fade_out,
        }

        if effect_type not in effects:
            raise ValueError(f"Unknown effect: {effect_type}")

        return effects[effect_type](audio, **kwargs)

    def reverb(self, audio, room_size=0.5, delay_ms=50, decay=0.5):
        """Add reverb effect to audio.

        Args:
            audio: Input audio
            room_size: Room size parameter (0-1)
            delay_ms: Delay time in milliseconds
            decay: Decay amount (0-1)

        Returns:
            np.ndarray: Audio with reverb
        """
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        reverb = audio.copy()

        for i in range(1, 5):
            delayed = np.zeros_like(audio)
            delayed[delay_samples * i:] = audio[:-delay_samples * i]
            reverb += delayed * (decay ** i)

        return reverb / np.max(np.abs(reverb))

    def echo(self, audio, delay_ms=500, decay=0.6, repeats=2):
        """Add echo/repetition effect.

        Args:
            audio: Input audio
            delay_ms: Delay time in milliseconds
            decay: Decay per repetition
            repeats: Number of repetitions

        Returns:
            np.ndarray: Audio with echo
        """
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        output = audio.copy()

        for i in range(repeats):
            delayed = np.zeros_like(audio)
            shift = delay_samples * (i + 1)
            delayed[shift:] = audio[:-shift] * (decay ** (i + 1))
            output += delayed

        return output / np.max(np.abs(output))

    def chorus(self, audio, rate_hz=1.5, depth=0.02):
        """Add chorus effect (pitch modulation).

        Args:
            audio: Input audio
            rate_hz: LFO rate in Hz
            depth: Modulation depth

        Returns:
            np.ndarray: Audio with chorus effect
        """
        samples = np.arange(len(audio))
        delay_samples = 10  # Base delay
        lfo = depth * np.sin(2 * np.pi * rate_hz * samples / self.sample_rate)
        varied_delay = delay_samples * (1 + lfo)

        chorus_output = np.interp(
            samples - varied_delay,
            np.arange(len(audio)),
            audio
        )

        return (audio + chorus_output) / 2

    def delay(self, audio, delay_ms=200, feedback=0.6, num_repeats=1):
        """Add delay effect.

        Args:
            audio: Input audio
            delay_ms: Delay time in milliseconds
            feedback: Feedback amount (0-1)
            num_repeats: Number of repeats

        Returns:
            np.ndarray: Audio with delay
        """
        delay_samples = int(delay_ms * self.sample_rate / 1000)
        output = audio.copy()

        for _ in range(num_repeats):
            delayed = np.zeros_like(audio)
            delayed[delay_samples:] = output[:-delay_samples] * feedback
            output = output + delayed

        return output / np.max(np.abs(output))

    def compression(self, audio, threshold=0.5, ratio=4, attack_ms=5, release_ms=50):
        """Apply dynamic range compression.

        Args:
            audio: Input audio
            threshold: Compression threshold
            ratio: Compression ratio
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds

        Returns:
            np.ndarray: Compressed audio
        """
        attack_samples = int(attack_ms * self.sample_rate / 1000)
        release_samples = int(release_ms * self.sample_rate / 1000)

        output = audio.copy()
        envelope = np.zeros_like(audio)

        for i in range(len(audio)):
            if i == 0:
                envelope[i] = abs(audio[i])
            else:
                level = abs(audio[i])
                if level > envelope[i - 1]:
                    # Attack
                    envelope[i] = envelope[i - 1] + (level - envelope[i - 1]) / attack_samples
                else:
                    # Release
                    envelope[i] = envelope[i - 1] - (envelope[i - 1] - level) / release_samples

        # Apply compression
        gain = np.ones_like(audio)
        above_threshold = envelope > threshold

        if np.any(above_threshold):
            gain[above_threshold] = (threshold + (envelope[above_threshold] - threshold) / ratio) / envelope[above_threshold]

        return output * gain

    def normalize(self, audio, target_level=0.9):
        """Normalize audio to target level.

        Args:
            audio: Input audio
            target_level: Target amplitude level (0-1)

        Returns:
            np.ndarray: Normalized audio
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio * (target_level / max_val)
        return audio

    def fade_in(self, audio, duration_ms=1000):
        """Apply fade-in effect.

        Args:
            audio: Input audio
            duration_ms: Fade duration in milliseconds

        Returns:
            np.ndarray: Audio with fade-in
        """
        fade_samples = int(duration_ms * self.sample_rate / 1000)
        fade_samples = min(fade_samples, len(audio))

        output = audio.copy()
        envelope = np.ones(len(audio))
        envelope[:fade_samples] = np.linspace(0, 1, fade_samples)

        return output * envelope

    def fade_out(self, audio, duration_ms=1000):
        """Apply fade-out effect.

        Args:
            audio: Input audio
            duration_ms: Fade duration in milliseconds

        Returns:
            np.ndarray: Audio with fade-out
        """
        fade_samples = int(duration_ms * self.sample_rate / 1000)
        fade_samples = min(fade_samples, len(audio))

        output = audio.copy()
        envelope = np.ones(len(audio))
        envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)

        return output * envelope

    def equalize(self, audio, gain_low=0, gain_mid=0, gain_high=0, frequencies=None):
        """Apply parametric equalization.

        Args:
            audio: Input audio
            gain_low: Gain adjustment for low frequencies (dB)
            gain_mid: Gain adjustment for mid frequencies (dB)
            gain_high: Gain adjustment for high frequencies (dB)
            frequencies: Custom frequency settings

        Returns:
            np.ndarray: Equalized audio
        """
        # Simplified EQ implementation
        from scipy.signal import butter, sosfilt

        output = audio.copy()

        # Low shelf (< 200 Hz)
        if gain_low != 0:
            sos = butter(2, 200, 'low', fs=self.sample_rate, output='sos')
            low_band = sosfilt(sos, audio)
            gain_factor = 10 ** (gain_low / 20)
            output += (low_band - audio) * (gain_factor - 1)

        return output

    def mix(self, *audio_tracks, volumes=None):
        """Mix multiple audio tracks.

        Args:
            *audio_tracks: Audio arrays to mix
            volumes: Volume levels for each track (0-1)

        Returns:
            np.ndarray: Mixed audio
        """
        if volumes is None:
            volumes = [1.0] * len(audio_tracks)

        # Pad to same length
        max_length = max(len(track) for track in audio_tracks)
        mixed = np.zeros(max_length)

        for track, volume in zip(audio_tracks, volumes):
            padded = np.zeros(max_length)
            padded[:len(track)] = track
            mixed += padded * volume

        # Normalize to prevent clipping
        max_val = np.max(np.abs(mixed))
        if max_val > 1.0:
            mixed = mixed / max_val

        return mixed
