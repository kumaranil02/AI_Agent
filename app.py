from smolagents import (
    CodeAgent,
    FinalAnswerTool,
    InferenceClientModel,
    GradioUI,
    tool
)

from smolagents.agent_types import AgentImage
from huggingface_hub import InferenceClient

import datetime
import pytz
import yaml
import os

from dotenv import load_dotenv

# =====================================================
# LOAD ENV
# =====================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN_")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN_ not found in .env file"
    )

# =====================================================
# TIME TOOL
# =====================================================

@tool
def get_current_time_in_timezone(timezone: str) -> str:
    """
    Get current time in a timezone.

    Args:
        timezone: Valid timezone such as Asia/Kolkata
    """

    try:
        tz = pytz.timezone(timezone)

        current_time = datetime.datetime.now(tz)

        return current_time.strftime("%Y-%m-%d %H:%M:%S")

    except Exception as e:
        return str(e)

# =====================================================
# IMAGE TOOL
# =====================================================

@tool
def generate_image(prompt: str) -> AgentImage:
    """
    Generate an image from a text prompt.

    Args:
        prompt: Detailed image description.
    """

    client = InferenceClient(
        token=HF_TOKEN
    )

    image = client.text_to_image(
        prompt=prompt,
        model="black-forest-labs/FLUX.1-schnell"
    )

    return AgentImage(image)

# =====================================================
# FINAL ANSWER TOOL
# =====================================================

final_answer = FinalAnswerTool()

# =====================================================
# MODEL
# =====================================================

model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    token=HF_TOKEN,
    max_tokens=2048,
    temperature=0.5,
)

# =====================================================
# PROMPTS
# =====================================================

try:
    with open("prompts.yaml", "r", encoding="utf-8") as f:
        prompt_templates = yaml.safe_load(f)
except FileNotFoundError:
    prompt_templates = None

# =====================================================
# AGENT
# =====================================================

agent = CodeAgent(
    model=model,
    tools=[
        generate_image,
        get_current_time_in_timezone,
        final_answer,
    ],
    max_steps=6,
    verbosity_level=2,
    prompt_templates=prompt_templates,
)

# =====================================================
# LAUNCH
# =====================================================

GradioUI(agent).launch()