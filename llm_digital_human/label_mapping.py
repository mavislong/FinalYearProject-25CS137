    

GoEmotion_Labels = [
    "admiration", "amusement", "anger", "annoyance", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism",
    "pride", "realization", "relief", "remorse", "sadness", "surprise",
    "neutral",
]


Group = [
    "neutral",               # 0
    "positive",        # 1
    "warm",           # 2
    "curious",    # 3
    "desire",                # 4
    "angry",         # 5
    "disgust",               # 6
    "sad",            # 7
    "anxious",       # 8
    "surprised",   # 9
]

Label_Group = {
    "neutral": "neutral",
    "joy": "positive",
    "amusement": "positive",
    "excitement": "positive",
    "gratitude": "positive",
    "love": "positive",
    "optimism": "positive",
    "relief": "positive",
    "pride": "positive",
    "admiration": "positive",
    "approval": "positive",
    "caring": "warm",
    "curiosity": "curious",
    "realization": "curious",
    "desire": "desire",
    "anger": "angry",
    "annoyance": "angry",
    "disapproval": "angry",
    "disgust": "disgust",
    "sadness": "sad",
    "disappointment": "sad",
    "embarrassment": "sad",
    "grief": "sad",
    "remorse": "sad",
    "fear": "anxious",
    "nervousness": "anxious",
    "surprise": "surprised",
    "confusion": "surprised",
}

Group_ID = {g: i for i, g in enumerate(Group)}
ID_Group = {i: g for g, i in Group_ID.items()}
