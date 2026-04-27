from sota.RAG.hybrid_retrieval import HybridRetriever
import pickle
import copy
import json
from openai import OpenAI
from utils.utils import read_md_file, to_str


class HybridRetrievalRag:
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
        self.retriever = HybridRetriever(database_path)
        self.base_prompt = read_md_file('sota/RAG/prompt/native_prompt.md')
        self.task_prompt = read_md_file('sota/RAG/prompt/task_prompt.md')
        self.extract_prompt = read_md_file('sota/RAG/prompt/hybrid_retrieval_prompt.md')
    
    def get_response(self, prompt):
        """
        Get AI response (SiliconFlow version)
        
        Args:
            prompt (str): User prompt for AI generation
        """
        # Build message list
        messages = [{"role": "user", "content": prompt}]
        
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

    def extract_keywords(self, sentence):
        prompt = self.extract_prompt + sentence
        resp = self.get_response(prompt)
        try:
            res = json.loads(resp)
            return res
        except:
            return []

    def gen_ans(self, obj, k=1):
        keywords = self.extract_keywords(obj["sentence"])
        matchs = self.retriever.hybrid_retrieval(obj, keywords, k) # Retrieve k similar tasks
        prompt = copy.deepcopy(self.base_prompt)
        for smp in matchs:
            prompt = prompt + to_str(smp, True) + '\n'
        prompt = prompt + self.task_prompt + to_str(obj, False)
        ans = self.get_response(prompt)

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

    native_rag = HybridRetrievalRag(api_key=API_KEY, model_name=model_name, database_path=database_path)

    for task in tasks:
        print(native_rag.gen_ans(task, k=1))
        break