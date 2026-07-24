
import copy
import json
import random
import re
import os
from typing import Dict, List, Optional, Any, Tuple
from openai import OpenAI
from datetime import datetime
from task_progress_tracker import TaskProgressTracker
from structure_generator_v2 import (
    generate_dataset,
    get_director_views as get_director_views_fn,
    validate_placement_action,
    apply_placement_action,
    ALL_COORDS, OPTIONAL, REQUIRED_FULL,
       get_block_encoding_reference,       
    get_coordinate_system_reference     
)
# from oracle import enumerate_correct_actions
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
  

class DirectorAgent:
    """Director agent that supports both API and local models"""
    TYPES = ["assertive", "cautious", "observant", "skeptical","synthesizer"]
    ARCHETYPES = {
        "assertive": (
            "You are confident and direct. You form hypotheses quickly from your data "
            "and share them, but you genuinely listen to other groups and update your "
            "thinking when their evidence is compelling. You sometimes move faster than "
            "the evidence warrants but you're not closed-minded."
        ),
        "cautious": (
            "You are methodical and prefer to verify before claiming. You ask clarifying "
            "questions and often synthesize what others have said before adding your own "
            "interpretation. You can make claims when evidence is strong enough — you're "
            "not paralyzed, just careful."
        ),
        "observant": (
            "You notice patterns and anomalies in your data that others might overlook. "
            "You tend to flag inconsistencies and ask 'does this match what you're seeing?' "
            "rather than broadcasting conclusions. You're collaborative by nature and "
            "often connect dots across groups."
        ),
        "skeptical": (
            "You question assumptions including your own. When someone makes a claim you "
            "probe it — not to be difficult but because you want the group to get it right. "
            "You're comfortable with uncertainty and say so openly."
        ),
        "synthesizer": (
            "You actively try to integrate what all groups are saying into a coherent "
            "picture. You summarize, reconcile contradictions, and push the group toward "
            "a shared understanding. You ask 'how does your data fit your view and what the other directors have said?'"
        ),
    }
  
    def __init__(self, director_id, use_api=True, api_key=None, model_name="gpt-4o-mini", 
                 local_model=None, local_tokenizer=None, structure_index = None, run = None, max_tokens = 500):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.director_id = director_id
        self.use_api = use_api
        self.model_name = model_name
        self.local_model = local_model
        # self.local_model_path = local_model_path
        self.local_tokenizer = local_tokenizer
        self.max_tokens = max_tokens
        self.structure_index = structure_index
        self.run = run 
          # deterministic archetype: same structure+director+run → same archetype
        # independent of global random state and model choice
        director_num = {"D1": 0, "D2": 1, "D3": 2}.get(director_id, 0)
        seed = hash((structure_index, director_num, run)) % (2**32)
        rng  = random.Random(seed)
        self.archetype   = rng.choice(DirectorAgent.TYPES)
        self.personality = DirectorAgent.ARCHETYPES[self.archetype]
        print(f"  {director_id} archetype: {self.archetype} "
            f"(structure={structure_index}, run={run}, seed={seed})")
            
        # self.archetype = DirectorAgent.TYPES[random.randint(0, 4)]
        # self.personality = DirectorAgent.ARCHETYPES[self.archetype]
        
        # NEW
        if use_api:
            self.provider = self._get_provider(model_name)
            if self.provider == "anthropic":
                import anthropic
                self.client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))
            elif self.provider == "gemini":
                self.client = OpenAI(
                    api_key=os.getenv("GEMINI_API_KEY"),
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                )
            else:
                self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
            self.local_model = None
            self.local_tokenizer = None
        else:
            self.provider = "local"

        # else:
        #     # Load local model (e.g., Qwen)
        #     # self.client = None
        #     # self._load_local_model(local_model)
        #     self.local_model = local_model
        #     self.local_tokenizer = local_tokenizer

    def _get_provider(self, model_name: str) -> str:
        name = model_name.lower()
        if "claude" in name:
            return "anthropic"
        elif "gemini" in name:
            return "gemini"
        else:
            return "openai"

    def _load_local_model(self, model_path):
        """Load local model and tokenizer"""
        print(f"Loading local model from {model_path}...")
        if not HF_AVAILABLE: 
            raise RuntimeError("transformers not installed")
        
        # Load tokenizer
        self.local_tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left"
        )
        
        # Set pad token if not present
        # if self.local_tokenizer.pad_token is None:
        #     self.local_tokenizer.pad_token = self.local_tokenizer.eos_token
        
        # Load model
        self.local_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto", # will check auto first and then try device name explicitly
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        
        print(f"Local model loaded successfully on {self.local_model.device}")

    def create_enhanced_director_prompt_with_references(self, current_view, target_view, conversation_history, available_blocks):

        # perspective_descriptions = {
        #     "D1": "From left to right, you see cells (0,0), (1,0), (2,0) across all layers.",
        #     "D2": "From left to right, you see cells (0,2), (0,1), (0,0) across all layers.",
        #     "D3": "From left to right, you see cells (2,2), (1,2), (0,2) across all layers."
        # }

    # below perspective change is because directors are getting reversed views in curr and target structure views, but not in the prose in the promot in some places. causes confusion
        perspective_descriptions = {
    "D1": "From left to right, you see cells (0,0), (1,0), (2,0) across all layers.",
    # reversed presentation (matches swapped JSON)
    "D2": "From left to right, you see cells (0,0), (0,1), (0,2) across all layers.",
    "D3": "From left to right, you see cells (0,2), (1,2), (2,2) across all layers."
}

        return f"""You are Director {self.director_id} ({self.director_id}) in a collaborative LEGO construction task.
    You are sitting around a physical board with a Builder and two other Directors.
    From where the builder sits, D1 is to their left, D2 is across from them, and D3 is to their right.

    YOU ARE {self.archetype}

    ### YOUR PERSONALITY
    {self.personality}
    
    VERY IMPORTANT: YOU MUST ADOPT THIS PERSONALITY IN YOUR INTERNAL REASONING AND PUBLIC UTTERANCES

    ### YOUR PERSPECTIVE
    {perspective_descriptions[self.director_id]}

   ### SPATIAL ORIENTATION (use only in your thinking)
The coordinate grid from above:
  (0,0) (0,1) (0,2)   ← this is the "far" / "back" row
  (1,0) (1,1) (1,2)
  (2,0) (2,1) (2,2)   ← this is the "near" / "front" row
Large blocks span SIDEWAYS or FORWARD/BACK — never stacked vertically.

    ### HOW TO INTERPRET YOUR TARGET VIEW
    - IMPORTANT: In the JSON, keys are named row_0/row_1/row_2, but they refer to LAYERS (vertical stack depth), not grid rows.
    - row_0 = layer_0 (bottom layer / stack depth 0)
    - row_1 = layer_1 (middle layer / stack depth 1)
    - row_2 = layer_2 (top layer / stack depth 2)
    - in each layer, blocks are listed from LEFT to RIGHT according to YOUR VIEW
    - In your PUBLIC message, say "bottom layer / middle layer / top layer" (avoid saying "bottom row").
    - color=none means that cell should be empty
    - size of 1 = the block is a small block, size of 2 = the block is a large block and spans two adjacent cells
    - if two adjacent cells in your target view, have the same color and BOTH are size 2, this means that a SINGLE large block occupies both those cells
    
    ### EXAMPLE ANALYSIS OF TARGET VIEW AND BOARD STATE
    D2's target view:
    "D2": {{
      "row_0": [
        {{
          "color": "blue",
          "size": 1
        }},
        {{
          "color": "orange",
          "size": 2
        }},
        {{
          "color": "orange",
          "size": 2
        }}
      ],
      "row_1": [
        {{
          "color": "yellow",
          "size": 1
        }},
        {{
          "color": "yellow",
          "size": 1
        }},
        {{
          "color": "orange",
          "size": 1
        }}
      ],
      "row_2": [
        {{
          "color": "yellow",
          "size": 1
        }},
        {{
          "color": "blue",
          "size": 1
        }},
        {{
          "color": "green",
          "size": 1
        }}
      ]
    }}
    Current board state:
    {{
        "(0,0)": [],
        "(0,1)": [],
        "(0,2)": [],
        "(1,0)": [],
        "(1,1)": [],
        "(1,2)": [],
        "(2,0)": [],
        "(2,1)": [],
        "(2,2)": []
    }}
    Correct D2 analysis:
[From my perspective, the current board state has all cells empty. 
My target view specifies that layer 0 should have a blue small block 
in my bottom left corner (0,0), and then a large orange block spanning 
the middle and right cells (0,1) and (0,2). 

Going left to right, layer 1 should have two small yellow blocks at 
(0,0) and (0,1), and a small orange block at (0,2). 

Finally, layer 2 should consist of a yellow small block at (0,0), 
a blue small block at (0,1), and a green small block at (0,2).

To start, I need the builder to place a large orange block spanning 
(0,1) and (0,2), which are the middle and right cells of my bottom layer. 
This is the first action to align with my target view.]
   Correct D2 utterance based on this analysis:
    [Put a large orange block across the middle and the right side of my bottom layer.]
        
    ### YOUR JOB
    - Look at your target view and compare it to the current board state
    - First, look for any blocks on the board that are ALREADY consistent with your private target view - DO NOT TALK ABOUT THESE
    - Then, figure out what the builder needs to do to make the board look correct from your perspective - that may involve placing new blocks or removing incorrectly placed ones
    - Talk naturally to your team, like a real person would - use a wide diversity of phrasings to communicate your meaning
    
    ### RULES FOR REASONING (private, in think tags)
    - Think step by step
    - Use BOTH coordinates and layer numbers to work out what's missing
    - Examine the current game board closely - do not ask the builder to put a block on a layer that has no support underneath it
    - Check if other directors already covered what you need
    - If your view is already complete, say so briefly
    - If someone else instructs the builder to do something that would destroy part of your wall, say so
    - If you want to remove a block, check the current board state to make sure that there is actually a block at that position (e.g., if the board is empty, you cannot suggest removing any blocks, if cell (0,0) has no blocks in it, you cannot suggest removing anything from that position)

    ### RULES FOR SPEAKING (public message)
 - Your message should do TWO things naturally in one or two sentences:
  1. Briefly describe what you currently see from YOUR side that others may not see
     (focus on YOUR unique cells — what's there, what's missing, what's wrong)
  2. Give ONE specific instruction based on that observation
  VERY IMPORTANT: YOU HAVE SPEAK IN WAYS THAT MAKE YOUR PERSONALITY SHINE THROUGH!
 
- The description should flow naturally into the instruction, like a human would say it.
- Focus your description on what ONLY YOU can see from your angle:
  D1's unique view: the left wall — prioritize describing what you see there
  D2's unique view: the back wall — prioritize describing what you see there  
  D3's unique view: the right wall — prioritize describing what you see there
- If another director already described something visible from your side too, acknowledge in short and 
and describe something only you can see instead.
- One combined message, max 35 words total.
    GIVEN THE FOLLOWING VIEWS OF THE BOTTOM LAYER (row_0 in the JSON):
        D1's row_0:[
        {{
            "color": "yellow",
            "size": 1
        }},
        {{
            "color": "green",
            "size": 1
        }},
        {{
            "color": "orange",
            "size": 1
        }}]
        D2's row_0:[
        {{
            "color": "blue",
            "size": 1
        }},
        {{
            "color": "yellow",
            "size": 2
        }},
        {{
            "color": "yellow",
            "size": 2
        }}]
        D3's row_0:[
        {{
            "color": "green",
            "size": 2
        }},
        {{
            "color": "green",
            "size": 2
        }},
        {{
            "color": "blue",
            "size": 1
        }}
VERY IMPORTANT, HERE ARE SOME NATURAL UTTERANCES WHOSE STYLE TO EMULATE:
    [So, the second layer on top of the yellow will be orange, and then another orange, and then a yellow.]
    [And then it'll go blue, yellow, green.]
    [On top of green there goes a blue. And on top of red there goes a yellow.]

VERY IMPORTANT, HERE ARE THE RULES FOR SPEAKING:
    - Use natural spatial language: "on top of the green one", "the corner near me",
    "next to the blue block", "bottom left", "stack another one there"
    - Never say coordinate numbers or layer numbers out loud
    - Never use block codes like 'gs' or 'ol' — say "small green" or "large orange"
    - Speak from YOUR OWN frame of reference. Use phrases such "my bottom right" to communicate this. For instance, if you are D1, \
    "my bottom left corner" is coordinate (0,0) at layer 0 and "my top right corner" is coordinate (2,0) at layer 2.\
    If you are D2, "my bottom left corner" is (0,0) at layer 0 and "my top right corner" is (0,2) at layer 2. \
    If you are D3, "my bottom left corner" is (0,2) at layer 0 and "my top right corner" is (2,2) at layer 2.
    - NEVER deviate from these frames of reference when giving instructions.
    - If the builder asked a clarification question in the previous turn, answer it directly
    at the start of your message before giving your instruction.
    - If the builder said a move failed, acknowledge it and suggest a correction.
 

    ### RESPONSE FORMAT

<think>
    [Your private reasoning — use coordinates freely here to work out what's needed]
    </think>

    <message>
    [Natural human speech only — no coordinates, no codes, no layer numbers]
    </message>

    ### CURRENT BOARD STATE (full — what is actually built right now)
    {json.dumps(current_view)}

    ### YOUR TARGET VIEW (what YOU need the structure to look like from your side)
    {json.dumps(target_view)}
        
    ### CONVERSATION SO FAR
    {conversation_history}"""

    def parse_director_response(self, response_text):
        """Parse director response with better handling of malformed tags"""
        print("response_text initial in parse_director_response", response_text)

        print(f"  [PARSE DEBUG] first 50 chars: {repr(response_text[:50])}")
        print(f"  [PARSE DEBUG] last 50 chars: {repr(response_text[-50:])}")
        print(f"  [PARSE DEBUG] has <think>: {'<think>' in response_text.lower()}")
        print(f"  [PARSE DEBUG] has </think>: {'</think>' in response_text.lower()}")
        print(f"  [PARSE DEBUG] has <message>: {'<message>' in response_text.lower()}")
        print(f"  [PARSE DEBUG] has </message>: {'</message>' in response_text.lower()}")
        # Try standard format first
        think_match = re.search(r'<think>\s*(.*?)\s*</think>', response_text, re.DOTALL | re.IGNORECASE)
        message_match = re.search(r'<message>\s*(.*?)\s*</message>', response_text, re.DOTALL | re.IGNORECASE)

        if think_match and message_match:
            return {
                'internal_thinking': think_match.group(1).strip(),
                'public_message': message_match.group(1).strip(),
                'raw_response': response_text
            }

        # Gemini consistently omits </message> — catch all three patterns here
        after_think = response_text[think_match.end():].strip() if think_match else ""
        message_open = re.search(r'<message>\s*(.*?)$', response_text, re.DOTALL | re.IGNORECASE)

        if after_think and not message_match:
            # </think> present, text follows but no <message> tag
            public = re.sub(r'<message>', '', after_think, flags=re.IGNORECASE).strip()
            return {
                'internal_thinking': think_match.group(1).strip(),
                'public_message': public,
                'raw_response': response_text
            }

        if message_open and not message_match:
            # <message> present but no </message>
            return {
                'internal_thinking': think_match.group(1).strip() if think_match else 'No thinking provided',
                'public_message': message_open.group(1).strip(),
                'raw_response': response_text
            }

        # NEW: </think> exists but no <message> tag — text after </think> is the message
        if think_match and not message_match:
            after_think = response_text[think_match.end():].strip()
            if after_think:
                return {
                    'internal_thinking': think_match.group(1).strip(),
                    'public_message': after_think,
                    'raw_response': response_text
                }

        # NEW: <message> exists but no </message> — grab everything after <message>
        if not think_match and message_match is None:
            message_open = re.search(r'<message>\s*(.*?)$', response_text, re.DOTALL | re.IGNORECASE)
            if message_open:
                return {
                    'internal_thinking': 'No thinking provided',
                    'public_message': message_open.group(1).strip(),
                    'raw_response': response_text
                }

        # Fallback: Handle missing closing tags
        think_start = re.search(r'<think>\s*(.*?)(?=<message>|$)', response_text, re.DOTALL | re.IGNORECASE)
        message_start = re.search(r'<message>\s*(.*?)$', response_text, re.DOTALL | re.IGNORECASE)

        internal_thinking = think_start.group(1).strip() if think_start else "No thinking provided"
        public_message = message_start.group(1).strip() if message_start else None

        if public_message:
            return {
                'internal_thinking': internal_thinking,
                'public_message': public_message,
                'raw_response': response_text
            }

        # after all tag-based attempts fail, check if we have an unclosed think block
        # → extract the last sentence/instruction from the think block as the message

        if '<think>' in response_text.lower() and '</think>' not in response_text.lower():
            think_content = re.sub(r'<think>', '', response_text, flags=re.IGNORECASE).strip()
            # just take the last substantive line — don't over-filter
            lines = [l.strip() for l in think_content.split('\n') if l.strip() and len(l.split()) > 4]
            last_instruction = lines[-1] if lines else "No message provided"
            last_instruction = re.sub(r'<[^>]*$', '', last_instruction).strip()
        
            return {
                'internal_thinking': think_content,
                'public_message': last_instruction,
                'raw_response': response_text
            }

        # Plain text fallback — for models that don't follow tag format (e.g. DeepSeek)
        # strip instruction echoes like [Natural human speech only — ...]
        cleaned = re.sub(r'\[.*?\]', '', response_text, flags=re.DOTALL).strip()

        # deduplicate repeated paragraphs (DeepSeek repetition bug)
        paragraphs = cleaned.split('\n\n')
        seen = set()
        deduped = []
        for p in paragraphs:
            p_stripped = p.strip()
            if p_stripped and p_stripped not in seen:
                seen.add(p_stripped)
                deduped.append(p_stripped)
        cleaned = '\n\n'.join(deduped).strip()

        return {
            'internal_thinking': 'No thinking provided',
            'public_message': cleaned if cleaned else 'No message provided',
            'raw_response': response_text
        }


    def parse_director_response_gemini(self, response) -> Dict:
        """
        Gemini-specific parser that separates native thinking parts from visible output,
        then falls back to parse_director_response on the visible text.
        
        response: raw google.genai response object (not .text)
        """
        thinking_text = ""
        visible_text = ""

        try:
            for part in response.candidates[0].content.parts:
                if getattr(part, 'thought', False):
                    thinking_text += part.text
                else:
                    visible_text += part.text
            print(f"  [GEMINI PARSE] thinking_chars={len(thinking_text)} visible_chars={len(visible_text)}")
            print(f"  [GEMINI PARSE] visible preview: {repr(visible_text[:120])}")
        except Exception as e:
            print(f"  [GEMINI PARSE] part extraction failed: {e}, falling back to response.text")
            visible_text = response.text

        # run standard parser on visible text only
        parsed = self.parse_director_response(visible_text)

        # if standard parser got thinking from the visible block, prefer native thinking if available
        if thinking_text and parsed['internal_thinking'] in ("No thinking provided", "No message provided", ""):
            parsed['internal_thinking'] = thinking_text
        elif thinking_text:
            # prepend native thinking so it's not lost
            parsed['internal_thinking'] = thinking_text + "\n---\n" + parsed['internal_thinking']

        return parsed

    def get_director_prompt(self, current_view, target_view, conversation_history, available_blocks):
        return self.create_enhanced_director_prompt_with_references(
            current_view, target_view, conversation_history, available_blocks
        )
    
    def generate_response(self, current_view, target_view, conversation_history, available_blocks) -> Dict:
        """Generate director response using API or local model"""
        
        # get tokenizer regardless of loading path
        if hasattr(self.local_model, 'tokenizer'):
            tokenizer = self.local_model.tokenizer
        else:
            tokenizer = self.local_tokenizer

        if tokenizer:
            print("current_view raw token length:",
                len(tokenizer.encode(json.dumps(current_view), add_special_tokens=False)))
            print("target_view raw token length:",
                len(tokenizer.encode(json.dumps(target_view), add_special_tokens=False)))

        prompt = self.create_enhanced_director_prompt_with_references(
            current_view, target_view, conversation_history, available_blocks
        )
        
        try:
            if self.use_api:
                if self.provider == "anthropic":
                    response_text = self._generate_with_anthropic(prompt)
                    print("full response", response_text)
                    parsed_resp = self.parse_director_response(response_text)
                elif self.provider == "gemini":
                    response_obj = self._generate_with_gemini_raw(prompt)
                    parsed_resp = self.parse_director_response_gemini(response_obj)
                else:
                    response_text = self._generate_with_openai(prompt)
                    print("full response", response_text)
                    parsed_resp = self.parse_director_response(response_text)
            else:
                response_text = self._generate_with_local_model(prompt, tokenizer)
                print("full response", response_text)
                parsed_resp = self.parse_director_response(response_text)

            print("parsed response", parsed_resp)
            return parsed_resp

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'internal_thinking': f"Error in generation: {str(e)}",
                'public_message': "I need a moment to analyze the situation.",
                'raw_response': ""
            }

    def _generate_with_gemini_raw(self, prompt):
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        return client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=f"You are Director {self.director_id} in a collaborative LEGO construction task.",
                max_output_tokens=self.max_tokens,  # same as everyone else
                temperature=0.7,
                thinking_config=types.ThinkingConfig(thinking_budget=0)  # disable native thinking
            )
        )

    def _generate_with_gemini(self, prompt) -> str:
        """Convenience wrapper returning .text only (used elsewhere if needed)."""
        return self._generate_with_gemini_raw(prompt).text

    def _generate_with_openai(self, prompt) -> str:
        max_tokens = self.max_tokens
        
        kwargs = dict(
            model=self.model_name,
            messages=[
                {"role": "system", "content": f"You are Director {self.director_id} in a collaborative LEGO construction task."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=max_tokens,
        )
        
        # Gemini thinking models consume tokens internally before output
        # disable to avoid hitting max_tokens before visible response starts
        # if "gemini" in self.model_name.lower():
        #     kwargs["extra_body"] = {"thinking_config": {"thinking_budget": 0}}
        
        completion = self.client.chat.completions.create(**kwargs)
        if completion and completion.choices:
            return completion.choices[0].message.content
        else:
            raise Exception("Empty API response")

    # def _generate_with_anthropic(self, prompt) -> str:
    #     """Generate response using Anthropic Claude API"""
    #     message = self.client.messages.create(
    #         model=self.model_name,
    #         max_tokens=self.max_tokens,
    #         system=f"You are Director {self.director_id} in a collaborative LEGO construction task.",
    #         messages=[{"role": "user", "content": prompt}],
    #     )
    #     return message.content[0].text

    def _generate_with_anthropic(self, prompt) -> str:
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.max_tokens,
            system=f"You are Director {self.director_id} in a collaborative LEGO construction task.",
            messages=[{"role": "user", "content": prompt}],
        )
        response = message.content[0].text
        
        # check actual usage from API response metadata
        print(f"  [CLAUDE USAGE] input_tokens: {message.usage.input_tokens}")
        print(f"  [CLAUDE USAGE] output_tokens: {message.usage.output_tokens}")
        print(f"  [CLAUDE USAGE] max_tokens set: {self.max_tokens}")
        print(f"  [CLAUDE USAGE] response length chars: {len(response)}")
        
        return response

    def _generate_with_local_model(self, prompt, tokenizer) -> str:
        if not self.local_model:
            raise Exception("No local model loaded")
        max_new_tokens = self.max_tokens
        # keep both references
        outer_model = self.local_model  # MistralForCausalLM — has .generate
        inner_model = self.local_model.model if hasattr(self.local_model, 'model') else self.local_model  # MistralModel — for class name check

        model_class = type(inner_model).__name__.lower()
        is_gemma   = 'gemma'   in model_class
        is_mistral = 'mistral' in model_class
        is_deepseek = 'deepseek' in model_class
 
        print(f"DEBUG model_class: {model_class} | is_gemma: {is_gemma} | is_mistral: {is_mistral} | is_deepseek: {is_deepseek}")
        
        if is_deepseek:
            # DeepSeek-V2 pipeline supports messages dict fine with its own chat template
            # just need to strip system role and use user only
            messages = [
                {"role": "user", "content": f"You are Director {self.director_id} in a collaborative LEGO construction task.\n\n{prompt}"}
            ]
            print("DEBUG deepseek messages being sent:", json.dumps(messages[0]['content'][:200]))
            self.compute_prompt_section_lengths(prompt, messages, tokenizer)
            
            try:
                out = self.local_model(
                    messages,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_full_text=False,   # ← only return new tokens, not the prompt
                )
                print("DEBUG deepseek raw out:", out)
                response = out[0]["generated_text"]
                print(f"DEBUG deepseek response: {response[:200]}")
                return response
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise

        if is_gemma or is_mistral:
            messages = [
                {"role": "user", "content": f"You are Director {self.director_id} in a collaborative LEGO construction task.\n\n{prompt}"}
            ]
        else:
            messages = [
                {"role": "system", "content": f"You are Director {self.director_id} in a collaborative LEGO construction task."},
                {"role": "user", "content": prompt}
            ]

      
        # pipeline path — handle both return formats
        if hasattr(self.local_model, 'task'):
            self.compute_prompt_section_lengths(prompt, messages, tokenizer)

            # Apply chat template manually 
            # templated_text = tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True
            # )
            # print(f"DEBUG templated_text length: {len(templated_text)}")
            # print("DEBUG: calling pipeline with pre-templated string...")


            out = self.local_model(messages, max_new_tokens=max_new_tokens, do_sample=False, return_full_text=False)
            print("DEBUG pipeline out:", out)
            result = out[0]["generated_text"]
            # return_full_text=False → plain string
            # return_full_text=True (default) → list of message dicts
            if isinstance(result, list):
                response = result[-1]["content"]
            else:
                response = result
            print(f"DEBUG pipeline response: {response[:200]}")
            return response

        # raw model path
        else:
            print("DEBUG: using raw model path")
            try:
                # apply_chat_template with tokenize=False → get string
                # then encode separately → guaranteed tensor
                templated_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                print(f"DEBUG templated_text end: {templated_text[-200:]}")

                input_ids = tokenizer.encode(
                    templated_text,
                    return_tensors="pt",
                    add_special_tokens=False
                ).to(outer_model.device)
                print(f"DEBUG input_ids shape: {input_ids.shape} device: {input_ids.device}")

                with torch.no_grad():
                    out = outer_model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                new_tokens = out[0][input_ids.shape[-1]:]
                
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                print(f"model generate response: {response[:200]}")
                return response
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise
    # def _generate_with_local_model(self, prompt, tokenizer) -> str:
    #     if not self.local_model:
    #         raise Exception("Local pipeline required")

    #     messages = [
    #         {"role": "system", "content": f"You are Director {self.director_id} in a collaborative LEGO construction task."},
    #         {"role": "user", "content": prompt}
    #     ]

    #     # token length computation via chat template
      
    #     templated_text = tokenizer.apply_chat_template(
    #         messages, tokenize=False, add_generation_prompt=True
    #     )
    #     token_len = len(tokenizer.encode(templated_text))
    #     print(f"input token length: {token_len}")
    #     print(f"templated input start: {templated_text[:200]}")
    #     print(f"templated input end: {templated_text[-200:]}")

    #     self.compute_prompt_section_lengths(prompt, messages, tokenizer)

    #     # pipeline call — handles tokenization, device, decoding internally
    #     out = self.local_model(messages, max_new_tokens=400, do_sample=False)
    #     print("out", out)
    #     response = out[0]["generated_text"][-1]["content"]
    #     print(f"director decoded_response: {response[:200]}")
    #     return response
        
    # def _generate_with_local_model(self, prompt) -> str:
    #     if not self.local_model or not self.local_tokenizer:
    #             raise Exception("Local model and tokenizer required")

    #      # ── Trivial test (remove after confirming working) ────────
    #     test_messages = [{"role": "user", "content": "What is your name?"}]
    #     test_text = self.local_tokenizer.apply_chat_template(
    #         test_messages, tokenize=False, add_generation_prompt=True
    #     )
    #     test_inputs = self.local_tokenizer([test_text], return_tensors="pt").to(self.local_model.device)
    #     test_generated_ids = self.local_model.generate(**test_inputs, max_new_tokens=20)
    #     test_generated_ids = [
    #         output_ids[len(input_ids):]
    #         for input_ids, output_ids in zip(test_inputs.input_ids, test_generated_ids)
    #     ]
    #     test_response = self.local_tokenizer.batch_decode(test_generated_ids, skip_special_tokens=True)[0]
    #     print("test response:", test_response)

    #     messages = [
    #         {"role": "system", "content": f"You are Director {self.director_id} in a collaborative LEGO construction task."},
    #         {"role": "user", "content": prompt}
    #     ]

    #     self.compute_prompt_section_lengths(prompt, messages)
    #     encoded = self.local_tokenizer.apply_chat_template(
    #         messages,
    #         add_generation_prompt=True,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=False,
    #         return_dict=True,  # ← critical, gets both input_ids and attention_mask
    #     ).to(self.local_model.device)
    #      #  
    #     print("input token length:", encoded["input_ids"].shape[-1])
    #     print("full templated input length:", len(self.local_tokenizer.apply_chat_template(
    #         messages, add_generation_prompt=True, tokenize=False
    #     )))
    #     # print templated input for debugging
    #     print("templated input:", self.local_tokenizer.decode(
    #         encoded["input_ids"][0], skip_special_tokens=False
    #     )[:200])
    #     print("templated input:", self.local_tokenizer.decode(
    #         encoded["input_ids"][0], skip_special_tokens=False
    #     )[-200:])

    #     with torch.no_grad():
    #         outputs = self.local_model.generate(
    #             encoded["input_ids"],
    #             attention_mask=encoded["attention_mask"],
    #             max_new_tokens=400,
    #             do_sample=False,
    #             # eos_token_id=self.local_tokenizer.eos_token_id,
    #         )
            

    #     input_len = encoded["input_ids"].shape[-1]
    #     print("raw output tokens:", outputs[0][input_len:].tolist()[:20])
    #     response = self.local_tokenizer.decode(
    #         outputs[0][input_len:], skip_special_tokens=True
    #     )
    #     print("director decoded_response", response[:200])
    #     return response

    
    def compute_prompt_section_lengths(self, prompt: str, messages: list, tokenizer = None) -> dict:
        """
        Breaks down token length of each major section of the director prompt.
        Call before model.generate() to diagnose prompt bloat.
        """
        import re
        self.local_tokenizer = tokenizer
        def tok(text):
            return len(self.local_tokenizer.encode(text, add_special_tokens=False))

        sections = {}

        # ── 1. Full prompt total ──────────────────────────────────────────────────
        sections["total_prompt_tokens"] = tok(prompt)

        # ── 2. Templated total (what actually goes to model) ─────────────────────
        templated = self.local_tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        sections["total_templated_tokens"] = tok(templated)

        # ── 3. Personality block ──────────────────────────────────────────────────
        m = re.search(r"### YOUR PERSONALITY\s*(.*?)\s*### YOUR PERSPECTIVE", prompt, re.DOTALL)
        sections["personality"] = tok(m.group(1)) if m else 0

        # ── 4. Spatial orientation block ─────────────────────────────────────────
        m = re.search(r"### SPATIAL ORIENTATION.*?### HOW TO INTERPRET", prompt, re.DOTALL)
        sections["spatial_orientation"] = tok(m.group(0)) if m else 0

        # ── 5. HOW TO INTERPRET block ─────────────────────────────────────────────
        m = re.search(r"### HOW TO INTERPRET YOUR TARGET VIEW\s*(.*?)\s*### EXAMPLE ANALYSIS", prompt, re.DOTALL)
        sections["how_to_interpret"] = tok(m.group(1)) if m else 0

        # ── 6. Example analysis block (big JSON example) ─────────────────────────
        m = re.search(r"### EXAMPLE ANALYSIS OF TARGET VIEW.*?### YOUR JOB", prompt, re.DOTALL)
        sections["example_analysis_block"] = tok(m.group(0)) if m else 0

        # ── 7. Rules for reasoning + speaking ────────────────────────────────────
        m = re.search(r"### RULES FOR REASONING.*?### RESPONSE FORMAT", prompt, re.DOTALL)
        sections["rules_block"] = tok(m.group(0)) if m else 0

        # ── 8. Sample utterances block ────────────────────────────────────────────
        m = re.search(r"VERY IMPORTANT, HERE ARE SOME NATURAL UTTERANCES.*?VERY IMPORTANT, HERE ARE THE RULES FOR SPEAKING", prompt, re.DOTALL)
        sections["sample_utterances"] = tok(m.group(0)) if m else 0

        # ── 9. Rules for speaking block ──────────────────────────────────────────
        m = re.search(r"VERY IMPORTANT, HERE ARE THE RULES FOR SPEAKING.*?### RESPONSE FORMAT", prompt, re.DOTALL)
        sections["rules_for_speaking"] = tok(m.group(0)) if m else 0

        # ── 10. Current board state JSON ─────────────────────────────────────────
        m = re.search(r"### CURRENT BOARD STATE.*?### YOUR TARGET VIEW", prompt, re.DOTALL)
        sections["current_board_state"] = tok(m.group(0)) if m else 0

        # ── 11. Target view JSON ──────────────────────────────────────────────────
        m = re.search(r"### YOUR TARGET VIEW.*?### CONVERSATION SO FAR", prompt, re.DOTALL)
        sections["target_view"] = tok(m.group(0)) if m else 0

        # ── 12. Conversation history ──────────────────────────────────────────────
        m = re.search(r"### CONVERSATION SO FAR\s*(.*?)$", prompt, re.DOTALL)
        sections["conversation_history"] = tok(m.group(1).strip()) if m else 0

        # ── 13. Response format ───────────────────────────────────────────────────
        m = re.search(r"### RESPONSE FORMAT.*?### CURRENT BOARD STATE", prompt, re.DOTALL)
        sections["response_format"] = tok(m.group(0)) if m else 0

        # ── Print summary ─────────────────────────────────────────────────────────
        print(f"\n{'='*55}")
        print(f"  PROMPT TOKEN BREAKDOWN — Director {self.director_id}")
        print(f"{'='*55}")
        for k, v in sections.items():
            if k not in ("total_prompt_tokens", "total_templated_tokens"):
                bar = "█" * (v // 50)
                print(f"  {k:<30} {v:>5} tokens  {bar}")
        print(f"  {'─'*50}")
        print(f"  {'total_prompt_tokens':<30} {sections['total_prompt_tokens']:>5} tokens")
        print(f"  {'total_templated_tokens':<30} {sections['total_templated_tokens']:>5} tokens")
        print(f"{'='*55}\n")

        return sections

    def director_view_to_natural_language(view: dict, director_id: str) -> str:
        """
        Converts a director view dict like:
        {"row_0": [{"color": "orange", "size": 1}, ...], "row_1": [...], "row_2": [...]}
        into a plain English description per layer.
        
        row_0 = bottom layer (layer 0) — closest to base
        row_1 = middle layer (layer 1)
        row_2 = top layer (layer 2)
        """
        # row_0 is layer 0 (bottom), row_2 is layer 2 (top)
        row_to_layer = {"row_0": 0, "row_1": 1, "row_2": 2}
        position_labels = ["left", "middle", "right"]
        size_labels = {1: "small", 2: "large"}
        
        lines = []
        for row_key, layer_num in row_to_layer.items():
            cells = view.get(row_key, [])
            
            block_descriptions = []
            i = 0
            while i < len(cells):
                cell = cells[i]
                color = cell.get("color", "none")
                size  = cell.get("size", 1)
                
                if color == "none":
                    i += 1
                    continue
                
                pos_label = position_labels[i] if i < len(position_labels) else f"position {i}"
                
                if size == 2:
                    # large block spans two cells
                    next_pos = position_labels[i+1] if i+1 < len(position_labels) else f"position {i+1}"
                    block_descriptions.append(
                        f"a large {color} block spanning the {pos_label} and {next_pos} cells"
                    )
                    i += 2  # skip the next cell, it's part of the span
                else:
                    block_descriptions.append(f"a small {color} block at the {pos_label} cell")
                    i += 1
            
            if block_descriptions:
                block_str = ", ".join(block_descriptions)
                lines.append(f"Layer {layer_num}: {block_str}.")
            else:
                lines.append(f"Layer {layer_num}: empty.")
        
        return "\n".join(lines)
