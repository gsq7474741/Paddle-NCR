import argparse
import logging
import os
import sys

import numpy as np
import paddle

from configs import cfg
from utils import utils


# init_psr=None,psr=None,model_n=None
def main(kwargs=None):
    init_parser = argparse.ArgumentParser(description='Model')
    init_parser.add_argument('--rank', type=int, default=1, help='1=ranking, 0=rating/click')
    init_parser.add_argument('--data_loader', type=str, default='DataLoader', help='Choose data_loader')
    init_parser.add_argument('--model_name', type=str, default='NCR', help='Choose saved_model to run.')
    init_parser.add_argument('--runner', type=str, default='BaseRunner', help='Choose runner')
    init_parser.add_argument('--data_processor', type=str, default='DataProcessor', help='Choose runner')
    # if init_psr is not None:
    #     init_parser=init_psr
    init_args, init_extras = init_parser.parse_known_args()

    data_loader_name = eval(init_args.data_loader)
    model_name = eval(init_args.model_name)
    if init_args.model_name in ['NCR']:
        init_args.runner_name = 'ProLogicRunner'
    else:
        init_args.runner_name = 'BaseRunner'
    runner_name = eval(init_args.runner_name)
    if init_args.model_name in ['SVDPP']:
        init_args.data_processor = 'HisDataProcessor'
    elif init_args.model_name in ['NCR', 'RNNModel', 'CompareModel', 'GRU4Rec', 'STAMP']:
        init_args.data_processor = 'ProLogicRecDP'
    data_processor_name = eval(init_args.data_processor)
    parser = argparse.ArgumentParser(description='')
    parser = utils.parse_global_args(parser)
    parser = data_loader_name.parse_data_args(parser)
    parser = model_name.parse_model_args(parser, model_name=init_args.model_name)
    parser = runner_name.parse_runner_args(parser)
    parser = data_processor_name.parse_dp_args(parser)
    # if psr is not None:
    #     parser=psr
    args, extras = parser.parse_known_args()

    if kwargs is not None:
        for k in kwargs.items():
            exec(f'args.{k[0]} = {k[1]}')



    log_file_name = [str(init_args.rank), init_args.model_name, args.dataset,
                     str(args.random_seed), 'optimizer=' + args.optimizer,
                     'lr=' + str(args.lr), 'l2=' + str(args.l2), 'dropout=' + str(args.dropout),
                     'batch_size=' + str(args.batch_size)]
    log_file_name = '__'.join(log_file_name).replace(' ', '__')
    if args.log_file == cfg.DEFAULT_LOG:
        args.log_file = './log/%s.txt' % log_file_name
    if args.result_file == './result/result.npy':
        args.result_file = './result/%s.npy' % log_file_name
    if args.model_path == './saved_model/%s/%s.pdiparams' % (init_args.model_name, init_args.model_name):
        args.model_path = './saved_model/%s/%s.pdiparams' % (init_args.model_name, log_file_name)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=args.log_file, level=args.verbose)
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(vars(init_args))
    logging.info(vars(args))
    logging.info('DataLoader: ' + init_args.data_loader)
    logging.info('Model: ' + init_args.model_name)
    logging.info('Runner: ' + init_args.runner_name)
    logging.info('DataProcessor: ' + init_args.data_processor)
    paddle.seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    logging.info('# cuda device: %s' % paddle.device.get_device())
    data_loader = data_loader_name(path=args.path, dataset=args.dataset, label=args.label, sep=args.sep)
    features, feature_dims, feature_min, feature_max = (
        data_loader.feature_info(include_id=model_name.include_id,
                                 include_item_features=model_name.include_item_features,
                                 include_user_features=model_name.include_user_features))
    if init_args.model_name in ['BaseModel']:
        model = model_name(label_min=data_loader.label_min,
                           label_max=data_loader.label_max, feature_num=len(features),
                           random_seed=args.random_seed, model_path=args.model_path)
    # elif init_args.model_name in ['RecModel', 'BiasedMF', 'SVDPP']:
    #     model = model_name(label_min=data_loader.label_min,
    #                        label_max=data_loader.label_max, feature_num=0, user_num=data_loader.
    #                        user_num, item_num=data_loader.item_num, u_vector_size=args.
    #                        u_vector_size, i_vector_size=args.i_vector_size, random_seed=args.random_seed,
    #                        model_path=args.model_path)
    # elif init_args.model_name in ['GRU4Rec']:
    #     model = model_name(neg_emb=args.neg_emb, neg_layer=args.neg_layer,
    #                        hidden_size=args.hidden_size, num_layers=args.num_layers,
    #                        p_layers=args.p_layers, label_min=data_loader.label_min,
    #                        label_max=data_loader.label_max, feature_num=0, user_num=data_loader.user_num,
    #                        item_num=data_loader.item_num,
    #                        u_vector_size=args.u_vector_size, i_vector_size=args.
    #                        i_vector_size, random_seed=args.random_seed, model_path=args.
    #                        model_path)
    # elif init_args.model_name in ['STAMP']:
    #     model = model_name(neg_emb=args.neg_emb, neg_layer=args.neg_layer,
    #                        hidden_size=args.hidden_size, num_layers=args.num_layers,
    #                        p_layers=args.p_layers, label_min=data_loader.label_min,
    #                        label_max=data_loader.label_max, feature_num=0, user_num=data_loader.user_num,
    #                        item_num=data_loader.item_num,
    #                        u_vector_size=args.u_vector_size, i_vector_size=args.
    #                        i_vector_size, random_seed=args.random_seed, model_path=args.
    #                        model_path, attention_size=args.attention_size)
    elif init_args.model_name in ['NCR', 'CompareModel']:
        model = model_name(label_min=data_loader.label_min, label_max=data_loader.label_max, feature_num=0,
                           user_num=data_loader.user_num, item_num=data_loader.item_num,
                           u_vector_size=args.u_vector_size, i_vector_size=args.i_vector_size, r_weight=args.r_weight,
                           ppl_weight=args.ppl_weight, pos_weight=args.pos_weight, random_seed=args.random_seed,
                           model_path=args.model_path)
    # elif init_args.model_name in ['RNNModel']:
    #     model = model_name(label_min=data_loader.label_min, label_max=data_loader.label_max, feature_num=0,
    #                        user_num=data_loader.
    #                        user_num, item_num=data_loader.item_num, u_vector_size=args.
    #                        u_vector_size, i_vector_size=args.i_vector_size, random_seed=args.random_seed,
    #                        model_path=args.model_path)
    else:
        logging.error('Unknown Model: ' + init_args.model_name)
        return
    model.apply(model.init_paras)
    if paddle.device.get_device() != 'cpu':
        model = model
    if init_args.model_name in ['NCR', 'RNNModel', 'CompareModel', 'GRU4Rec', 'STAMP']:
        data_loader.append_his(last_n=args.max_his, supply=False, neg=True, neg_column=False)
    if init_args.rank == 1:
        data_loader.drop_neg()
    if init_args.data_processor in ['ProLogicRecDP']:
        data_processor = data_processor_name(data_loader, model, rank=init_args.rank, test_neg_n=args.test_neg_n,
                                             max_his=args.
                                             max_his, sup_his=0, sparse_his=0)
    elif init_args.data_processor in ['HisDataProcessor']:
        data_processor = data_processor_name(data_loader, model, rank=init_args.rank, test_neg_n=args.test_neg_n,
                                             sup_his=args.
                                             sup_his, max_his=args.max_his, sparse_his=args.sparse_his)
    else:
        data_processor = data_processor_name(data_loader, model, rank=init_args.rank, test_neg_n=args.test_neg_n)
    if init_args.runner_name in ['BaseRunner', 'ProLogicRunner']:
        runner = runner_name(optimizer=args.optimizer, learning_rate=args.
                             lr, epoch=args.epoch, batch_size=args.batch_size,
                             eval_batch_size=args.eval_batch_size, dropout=args.dropout, l2=args.l2,
                             metrics=args.metric, check_epoch=args.check_epoch,
                             early_stop=args.early_stop)
    else:
        logging.error('Unknown Runner: ' + init_args.runner_name)
        return
    logging.info('Test Before Training = ' + utils.format_metric(
        runner.evaluate(model, data_processor.get_test_data(), data_processor)) + ' ' + ','.join(runner.metrics))

    if args.load > 0:
        model.load_model()
    if args.train > 0:
        # dist.init_parallel_env()
        # saved_model=paddle.DataParallel(saved_model)
        runner.train(model, data_processor, skip_eval=args.skip_eval)

    logging.info('Test After Training = ' + utils.format_metric(
        runner.evaluate(model, data_processor.get_test_data(), data_processor)) + ' ' + ','.join(runner.metrics))

    np.save(args.result_file, runner.predict(model, data_processor.get_test_data(), data_processor))

    logging.info('Save Test Results to ' + args.result_file)
    logging.debug(runner.evaluate(model, data_processor.get_test_data(), data_processor))
    logging.debug(runner.evaluate(model, data_processor.get_test_data(), data_processor))
    return


# paddle.device.get_device()
if __name__ == '__main__':
    # paddle.enable_static()
    main()
