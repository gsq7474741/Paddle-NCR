"""python ./main.py --rank 1 --model_name NCR --optimizer Adam --lr 0.001 --dataset ml100k01-1-5 --metric ndcg@5,ndcg@10,hit@5,hit@10 --test_neg_n 100 --max_his 5 --r_weight 0.1 --random_seed 2021 --gpu 0"""
"""python ./main.py --rank 1 --model_name NCR --optimizer Adam --lr 0.001 --dataset 5MoviesTV01-1-5 --metric ndcg@5,ndcg@10,hit@5,hit@10 --test_neg_n 100 --max_his 5 --r_weight 0.1 --random_seed 2021 --gpu 0"""
"""python ./main.py --rank 1 --model_name NCR --optimizer Adam --lr 0.001 --dataset Electronics01-1-5 --metric ndcg@5,ndcg@10,hit@5,hit@10 --test_neg_n 100 --max_his 5 --r_weight 0.1 --random_seed 2019 --gpu 0"""
"""x2paddle --convert_torch_project --project_dir=/Users/songqi/PycharmProjects/NCR --save_dir=/Users/songqi/PycharmProjects/NCR-paddle"""

"""python ./main.py -- load --rank 1 --model_name NCR --optimizer Adam --lr 0.001 --dataset ml100k01-1-5 --metric ndcg@5,ndcg@10,hit@5,hit@10 --test_neg_n 100 --max_his 5 --r_weight 0.1 --random_seed 2021 --gpu 0"""
