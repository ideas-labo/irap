from IRAPE.Anal_reason.retrieval import Retriever
from IRAPE.Anal_reason.reasoning import Reasoner
from IRAPE.base_op import BaseOp
import copy
import pickle
import json

class Transfer :
    def __init__(self, database_path):
        self.retriever = Retriever(database_path)

    def make_obj(self, sentence, list1) -> dict:
        res = {}
        res["sentence"] = sentence
        res["label"] = list1
        res["vector"] = self.retriever.embedding(sentence)
        return res

       
    def trans(self, obj) -> list:
        A = obj["label"]
        best_match = self.retriever.retrieval(obj)[0]
        A_k = best_match["label"]
        A_k_prime = best_match["prefer_label"]
        points = copy.deepcopy(A)
        ops = Reasoner.diff_reasoning(A_k, A_k_prime)
        points = BaseOp.batch_exec(points, ops)
        return points
    

if __name__ == '__main__':
    task_data_path = 'dataset/pkl/final_Promise.pkl'
    database_path = 'dataset/pkl/final_synthetic_dataset.pkl'
    task = []
    with open(task_data_path, 'rb') as f:
        task = pickle.load(f)
    
    transfer = Transfer(database_path)
    for data in task:
        origin_points = data["label"]
        real_points = data["prefer_label"]
        pred_points = transfer.trans(data)
        print("origin points : ", origin_points)
        print("final points : ", real_points)
        print("predicte points : ", pred_points)
        print("----------------\n")