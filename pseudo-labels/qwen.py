# qwen.py
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)
from qwen_vl_utils import process_vision_info


# -----------------------------
# 3-step CoT prompt
# -----------------------------

RATIONALE_TEMPLATE = r"""
The detected object is a {category} and is located {bbox_description} in the image. It is in the {is_foreground}.
"""

PROMPT_TEMPLATE = r"""
You are provided with an image, and you are a vision-language model with advanced three-step chain-of-thought reasoning.

Follow these steps carefully:

Step 1) Determine whether no specific object is explicitly identifiable.
- “Yes”: if and only if a specific object is clearly detected. Continue to Step 2.
- “No”: if no specific object is detected. Stop here at Step 1.
- “Unsure”: if you are not almost 100 percent certain whether a specific object is detected. Stop here at Step 1.

 [Examples]
    Example 1:  
    Image: (A color photo of a dog standing clearly in focus)  
    Answer: Yes

    Example 2:  
    Image: (A grayscale image with blurred outlines and no clear shapes)  
    Answer: No

    Example 3:  
    Image: (An image where a part of an object might be present, but it is not fully visible or too unclear) 
    Answer: Unsure
    
Step 2) Determine the detected specific object category in 1-2 words.
- Focus only on the colored region; the grayscale and blurred areas must be completely ignored.
- If Step 1 answer is either “No” or “Unsure”, return an empty string "".

 [Examples]
    Example 1:
    Image: (A focused image of a grayscale and blurred person riding a colored skateboard)  
    Answer: skateboard

    Example 2:
    Image: (A colored image of a zebra walking in grayscale and blurry grass)  
    Answer: zebra

    Example 3:  
    Image: (A colored bowl containing grayscale and blurry contents)
    Answer: bowl
        
Step 3) Based on general common sense, is the detected object typically considered a foreground object (Yes) or a background element (No)?
- Focus only on the colored region; the grayscale and blurred areas must be completely ignored.
- Answer either “Yes” or “No”.
- If Step 1 answer is either “No” or “Unsure”, return an empty string "".

 [Examples]
    Example 1:  
    Object: dog
    Image: (A colored dog is standing in sharp focus in front of a grayscale and blurry park)  
    Answer: Yes

    Example 2:  
    Object: sky
    Image: (A grayscale and blurry person is standing in front of a bright blue sky)  
    Answer: No

    Example 3:  
    Object: tree
    Image: (A grayscale and blurry person in front, with colored trees in the back) 
    Answer: No


IMPORTANT:
Return your final answer as JSON only, with this schema:
{
  "presence": "<Step 1 Answer>",
  "class_name": "<Step 2 Answer>",
  "is_foreground": "<Step 3 Answer>",
  "description": "<Describe the reasoning behind your answers, only focusing on colored cues (ignoring grayscale and blurry areas). The format is: A photo of a {reasoning}.>" 
}

 [Examples]
    Example 1:  
    Image: (A color photo of a cow that has black and white spots and is grazing in a green field)  
    Final Answer:
    {
      "presence": "Yes",
      "class_name": "cow",
      "is_foreground": "Yes",
      "description": "A photo of a cow that has black and white spots and is grazing in a green field."
    }

    Example 2:  
    Image: (A color photo of a concrete apartment building)
    Final Answer:
    {
      "presence": "Yes",
      "class_name": "building",
      "is_foreground": "No",
      "description": "A photo of a concrete apartment building."
    }
"""

# sample_class_names = [
#     "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
#     "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
#     "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
#     "backpack", "umbrella", "handbag", "tie", "suitcase",
#     "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
#     "skateboard", "surfboard", "tennis racket",
#     "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
#     "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
#     "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
#     "keyboard", "cell phone",
#     "microwave", "oven", "toaster", "sink", "refrigerator",
#     "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
# ]

# PROMPT_TEMPLATE2 = r"""
# You are provided with an image and an original class name, and you are a vision-language model with advanced two-step chain-of-thought reasoning.
# - original class name: {class_name}

# Follow these steps carefully:

# Step 1) Determine whether the original class name of an object in the image can be reasonably mapped to any class name below.
# - 80-class list: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

# Step 2) Match to the closest class name from the 80-class list.
# - Return the matched class name that exists in the 80-class list; if there is no match, return the original class name {class_name}.

# [Example]
# Matched output format: teddy bear

# Unmatched output format: ""
# """


# -----------------------------
# Output validation / sanitation
# -----------------------------

_ALLOWED_PRESENCE = {"Yes", "No", "Unsure"}
_ALLOWED_LABEL = {"Yes", "No"}


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    js = m.group(0)
    try:
        return json.loads(js)
    except Exception:
        return None


def validate_and_postprocess(
    obj: Dict[str, Any],
) -> Tuple[Optional[Dict[str, str]], str]:

    presence = obj.get("presence")
    class_name = obj.get("class_name")
    is_foreground = obj.get("is_foreground")
    description = obj.get("description")

    if presence not in _ALLOWED_PRESENCE:
        return None, f"Invalid presence: {obj.get('presence')}"

    if presence == "No":
        # Sanity Check 1
        if class_name != "" or is_foreground != "":
            return None, "Presence is No but class_name/is_foreground are not empty strings."
        # Label must be empty, OK
        return {
            "presence": "No",
            "class_name": "",
            "is_foreground": "",
            "description": description if description else "Nothing identifiable is visible in the ROI.",
        }, ""

    # presence is Yes or Unsure => class_name/is_foreground must be non-empty (Sanity Check 1)
    if class_name == "" or is_foreground == "":
        return None, "Presence is Yes/Unsure but class_name/is_foreground is empty."

    if is_foreground not in _ALLOWED_LABEL:
        return None, f"Invalid is_foreground: {obj.get('is_foreground')}"

    return {
        "presence": presence,
        "class_name": class_name,
        "is_foreground": is_foreground,
        "description": description if description else "Decision based on the clearest visible cues inside the ROI.",
    }, ""


# -----------------------------
# Qwen wrapper
# -----------------------------

@dataclass
class QwenResult:
    presence: str
    class_name: str
    is_foreground: str
    description: str


class Qwen:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        device: str = "cuda",
        max_new_tokens: int = 384,
    ):
        self.device = device
        self.max_new_tokens = int(max_new_tokens)

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",#torch.bfloat16,
            device_map="auto",
            # attn_implementation="sdpa"
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name)

    def forward(self, image, bbox) -> QwenResult:

        prompt = PROMPT_TEMPLATE
        raw = self._generate_json_text(image=image, prompt=prompt)

        # sanity check 1
        obj = _extract_json(raw)
        if obj is None:
            err = "Could not parse JSON."
            return QwenResult(
                presence="Unsure",
                class_name="",
                is_foreground="",
                description="Could not parse JSON from model output.",
            )
        
        # sanity check 2
        final_obj, err = validate_and_postprocess(obj)
        if final_obj is None:
            return QwenResult(
                presence="Unsure",
                class_name="",
                is_foreground="",
                description=f"Model output was invalid: {err}",
            )
        
        class_name = final_obj["class_name"]
        # if final_obj["presence"] == "Yes":
        #     if final_obj["class_name"] not in sample_class_names:
        #         prompt = PROMPT_TEMPLATE2.format(class_name=class_name)
        #         matched_class_name = self._generate_json_text(image=image, prompt=prompt)
        #         print("@@", class_name, matched_class_name)
        #         if matched_class_name in sample_class_names:
        #             class_name = matched_class_name

        return QwenResult(
            presence=final_obj["presence"],
            class_name=class_name,
            is_foreground=final_obj["is_foreground"],
            description=final_obj["description"],
        )

    def _generate_json_text(self, image: Image.Image, prompt: str) -> str:
        content = [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        out_text = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        return out_text
