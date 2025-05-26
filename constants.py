# constants.py

# Define constants
BATCH_SIZE = 32
MODEL_NAME = "openai/clip-vit-large-patch14"
TEST_DIR = "/kaggle/working/RAFDB/test"  # Path to your test folder
TRAIN_DIR = "/kaggle/working/RAFDB/train"  # Path to your train folder
BOTTLENECK_DIM = 64  # Dimensionality of the bottleneck layer
LEARNING_RATE = 3e-4
NUM_EPOCHS = 5
ALPHA = 0.2  # Residual ratio for visual branch (can be tuned)
BETA = 0.2   # Residual ratio for text branch (can be tuned)

# RAFDB emotion labels
EMOTIONS = [
    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
]

# Detailed emotion descriptions for zero-shot classification
def get_emotion_descriptions():
    """Create detailed descriptions for each emotion category"""

    descriptions = {
        "angry": [
            "the image of an angry facial emotion with furrowed brows and clenched teeth",
            "a person expressing anger with narrowed eyes and tightened jaw",
            "a face showing intense frustration and hostility",
            "an irritated facial expression with a glaring stare",
            "a person displaying rage with tensed facial muscles"
        ],
        "disgust": [
            "the image of a disgusted facial emotion with wrinkled nose and raised upper lip",
            "a person expressing revulsion with a grimace and squinted eyes",
            "a face showing strong aversion with curled lip",
            "a nauseated facial expression with furrowed brows",
            "a person displaying distaste with pulled back lips"
        ],
        "fear": [
            "the image of a fearful facial emotion with widened eyes and raised eyebrows",
            "a person expressing terror with a dropped jaw and pulled-back lips",
            "a face showing panic with tense mouth and dilated pupils",
            "a frightened facial expression with raised upper eyelids",
            "a person displaying anxiety with frozen stare and pale complexion"
        ],
        "happy": [
            "the image of a happy facial emotion with upturned mouth corners and crinkled eyes",
            "a person expressing joy with a broad smile and relaxed face",
            "a face showing delight with raised cheeks and visible teeth",
            "a cheerful facial expression with beaming smile and bright eyes",
            "a person displaying pleasure with dimples and lifted cheeks"
        ],
        "neutral": [
            "the image of a neutral facial emotion with relaxed features and natural expression",
            "a person with an emotionless face showing no particular feeling",
            "a face with a balanced expression, neither positive nor negative",
            "a composed facial expression with resting features",
            "a person displaying a calm and unemotional demeanor"
        ],
        "sad": [
            "the image of a sad facial emotion with downturned mouth and drooping eyelids",
            "a person expressing sorrow with furrowed brows and quivering lips",
            "a face showing grief with lowered gaze and compressed lips",
            "a melancholic facial expression with sunken cheeks",
            "a person displaying unhappiness with glazed or teary eyes"
        ],
        "surprise": [
            "the image of a surprised facial emotion with raised eyebrows and widened eyes",
            "a person expressing astonishment with an open mouth and stretched skin",
            "a face showing shock with expanded pupils and heightened alertness",
            "a startled facial expression with dropped jaw and gasping mouth",
            "a person displaying amazement with rounded eyes and lifted brows"
        ]
    }

    return descriptions