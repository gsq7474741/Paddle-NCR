from main import main

argdict = {'train': 0, 'load': 1, 'model_path': '"saved_model/NCR/0.3653_0.4254_0.5287_0.7144best_test.model"',
           'dataset': "'Electronics01-1-5'", 'metric': "'ndcg@5,ndcg@10,hit@5,hit@10'", 'max_his': 5, 'test_neg_n': 100,
           'l2': 1e-4,
           'r_weight': 0.1, 'random_seed': 1, 'gpu': '"0"'}

# dist.spawn(main,args=(argdict))
main(argdict)
