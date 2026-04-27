from IRAPE.Base_gen.base_generator import BaseGenerator
from IRAPE.Anal_reason.diff_trans import Transfer
from IRAPE.Query.interaction_simulate import query

if __name__ == '__main__':
    database_path = 'dataset/pkl/final_synthetic_dataset.pkl'
    req_sentence = "The search results shall be returned no later 30 seconds  after the user has entered the search criteria."
    prefer_state = [[30.0, 0.99], [31.5, 0.495], [33.0, 0.0]]
    
    state1 = BaseGenerator.base_gen(req_sentence)
    
    transfer = Transfer(database_path)
    obj = transfer.make_obj(req_sentence, state1)
    state2 = transfer.trans(obj)

    ops, state3 = query(state2)

    print(state3)


