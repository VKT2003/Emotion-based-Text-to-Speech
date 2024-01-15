import gradio as gr
from transformers import pipeline

# Emotion Text Classification
title_emotion = "Classify text according to emotion"
description_emotion = "Emotion text classification by Vishal Tiwari "
examples_emotion = [
    ["Remember before Twitter when you took a photo of food, got the film developed, then drove around showing everyone the pic? No? Me neither."],
    ['''"We are all here because we're committed to the biggest question of all: What's out there?" Take your first steps toward answering that question by watching our Gameplay Reveal from the #XboxBethesda Showcase. '''],
    ["A STUNNER IN KNOXVILLE! 😱 Notre Dame takes down No. 1 Tennessee for its first trip to Omaha in 20 years‼️"],
    ["you and I best moment is yet to come 💜 #BTS9thAnniversary"]
]

interface_emotion = gr.Interface.load(
    "huggingface/bhadresh-savani/bert-base-go-emotion",
    title=title_emotion,
    description=description_emotion,
    examples=examples_emotion
)

# Text to Speech Translation
title_tts = "Text to Speech Translation"
examples_tts = [
    "I love learning machine learning",
    "How do you do?",
]

interface_tts = gr.Interface.load(
    "huggingface/facebook/fastspeech2-en-ljspeech",
    title=title_tts,
    examples=examples_tts,
    description="Give me something to say!",
)

# Launching both interfaces with tabs
demo = gr.TabbedInterface([interface_emotion, interface_tts], ["Emotion Classification", "Text to Speech"])

if __name__ == "__main__":
    demo.launch()
    