from openai import OpenAI
import time
import json
import os

class AIDebate:
    def __init__(self, api_key, base_url):
        """
        Initialize AI conversation handler (Direct Human Interaction).
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    
    def simulate_conversation(self, 
                            role_a_name: str, 
                            role_a_description: str, 
                            rounds: int = 3,
                            model: str = "Qwen/Qwen2.5-72B-Instruct"):
        """
        Facilitates a conversation between the AI and the Human User via CLI.
        """
        role_a_history = []
        
        print(f"\n{'='*20} INTERACTIVE SESSION START {'='*20}")
        print(f"Task: Follow the instructions provided to refine the quantization.")
        print(f"{'='*55}\n")

        # Start with a trigger for the AI to present the initial state or analysis
        current_user_input = "Please analyze the sentence and the initial quantization, then propose adjustments."
        
        for round_num in range(rounds):
            print(f"\n--- Round {round_num + 1}/{rounds} ---")
            
            # 1. Get AI Response
            role_a_response = self._get_ai_response(
                system_prompt=role_a_description,
                user_message=current_user_input,
                conversation_history=role_a_history,
                model=model
            )
            
            # Display AI Output
            print(f"\n[{role_a_name}]: {role_a_response}")
            
            # Update history with AI's message
            role_a_history.append({"role": "assistant", "content": role_a_response})

            # 2. Check for final round to extract JSON result
            if round_num == rounds - 1:
                print("\n[System]: Final round reached. Extracting structured result...")
                extract_prompt = read_md_file("sota/LLM_api/extract_ans.md")
                if not extract_prompt: 
                    extract_prompt = "Based on the discussion above, output the final list of lists [[x1, y1], ...] in JSON format ONLY."
                
                final_ans = self._get_ai_response(
                    system_prompt="You are a data extraction assistant.",
                    user_message=f"{extract_prompt}\nLast response: {role_a_response}",
                    conversation_history=role_a_history, 
                    model=model
                )
                return final_ans

            # 3. Get Human Feedback
            print(f"\n[Your Input]: ", end="")
            current_user_input = input()
            
            # Update history with human message
            role_a_history.append({"role": "user", "content": current_user_input})

    def _get_ai_response(self, system_prompt, user_message, conversation_history, model):
        messages = [{"role": "system", "content": system_prompt}]
        # Context window management
        for msg in conversation_history[-10:]:
            messages.append(msg)
        messages.append({"role": "user", "content": user_message})
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1000,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API call error: {e}")
            return "Error: AI failed to respond."

def read_md_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except:
        return ""

def LLM_api(obj, model_name="Qwen/Qwen2.5-72B-Instruct"): 
    API_KEY = ""
    BASE_URL = ""

    sentence = obj["sentence"]
    list1 = obj["label"]
    
    ai_handler = AIDebate(api_key=API_KEY, base_url=BASE_URL)
    
    # System Instruction for the AI
    role_a_name = "Bot"
    role_a_description = (
        f"{read_md_file('sota/LLM_api/bot.md')}\n"
        f"Context:\nSentence: {sentence}\n"
        f"Initial Labeling: {list1}\n"
        "Goal: Work with the user to refine these values into a professional quantization form."
    )
    
    # Simulate conversation (User acts as Role B internally)
    final_ans_raw = ai_handler.simulate_conversation(
        role_a_name=role_a_name,
        role_a_description=role_a_description,
        rounds=3, 
        model=model_name
    )

    try:
        # Extract JSON list from AI response
        start = final_ans_raw.find('[')
        end = final_ans_raw.rfind(']') + 1
        res = json.loads(final_ans_raw[start:end])
        return res
    except:
        print("\n[Warning]: Failed to parse JSON result. Returning initial label.")
        return obj["label"]

if __name__ == "__main__":
    obj = {
        "sentence" : "ReqView Desktop shall start in less than 10s.\n",
        "label" : [[10.0, 1.0], [11.0, 0.0]],
        "prefer_label" : [[10.5, 1.0],  [10.8, 0.6], [11.2, 0.0]]
    }
    # prefer_label is kept in obj for compatibility but no longer used for automated roleplay
    final_ans = LLM_api(obj)
    print(f"\nFinal Result: {final_ans}")