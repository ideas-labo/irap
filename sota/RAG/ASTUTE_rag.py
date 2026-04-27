from IRAPE.Anal_reason.retrieval import Retriever
import pickle
import copy
import json
from openai import OpenAI
from utils.utils import read_md_file, to_str

class AstuteRag:
    def __init__(self, api_key, base_url, model_name, database_path):

        """
        Initialize model API
        
        Args:
            api_key (str): SiliconFlow API key
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model_name = model_name
        self.retriever = Retriever(database_path)
        self.inner_prompt = read_md_file('sota/RAG/prompt/gen_inner_msg.md')
        self.extern_prompt = read_md_file('sota/RAG/prompt/integrate_extern_msg.md')
    
    def get_response(self, messages):
        """
        Get AI response (SiliconFlow version)
        
        Args:
            messages (list): List of conversation messages in OpenAI format
        """        
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,  # Increase creativity
                max_tokens=500,   # Limit response length
                stream=True       # Use streaming output
            )
            
            # Collect complete response
            full_response = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    full_response += content
            
            return full_response
            
        except Exception as e:
            print(f"API call error: {e}")
            return "Sorry, AI cannot respond temporarily."

    def gen_ans(self, obj, k=1):
        # Extract internal knowledge
        prompt = copy.deepcopy(self.inner_prompt)
        prompt = prompt + to_str(obj, False)
        # print(prompt)
        messages = [{"role": "user", "content": prompt}]
        ans = self.get_response(messages)
        # print(ans)
        messages.append({"role" : "assistant", "content" : ans})

        # Integrate retrieved knowledge
        matchs = self.retriever.retrieval(obj, k=k) # Retrieve 5 similar tasks
        prompt = copy.deepcopy(self.extern_prompt)
        for smp in matchs:
            prompt = to_str(smp, True) + '\n' + prompt 
        prompt = prompt + to_str(obj, False) 
        # print(prompt)
        messages.append({"role": "user", "content": prompt})
        ans = self.get_response(messages)
        # print(ans)

        try:
            res = json.loads(ans)
            return res
        except:
            res = obj["label"] # Return unadjusted result if answer format is invalid
            return res



if __name__ == '__main__':
    model_name=""
    API_KEY = ""
    BASE_URL = ""
    database_path = 'dataset/pkl/final_synthetic_dataset.pkl'
    task_data_path = 'dataset/pkl/final_Promise.pkl'
    tasks = []
    with open(task_data_path, 'rb') as f:
        tasks = pickle.load(f)

    astute_rag = AstuteRag(api_key=API_KEY, base_url=BASE_URL, model_name=model_name, database_path=database_path)

    for task in tasks:
        astute_rag.gen_ans(task, k = 5)
        break