'''
use tensorboard
from your local machine, run
ssh -N -f -L localhost:16006:localhost:6006 <user@remote>

on the remote machine, run tensorboard --logdir <path>
'''
import argparse
import torch
import numpy as np
import time
import os
import logging
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from utils.minibatch_loader import minibatch_loader
from utils.load_word2vec import load_word2vec_embeddings
from model.AttSum import AttSum
from tqdm import trange
from tensorboardX import SummaryWriter


USE_CUDA = torch.cuda.is_available()

def to_var(inputs, use_cuda, evaluate=False):
    if use_cuda:
        return Variable(torch.from_numpy(inputs).cuda(), volatile=evaluate)
    else:
        return Variable(torch.from_numpy(inputs), volatile=evaluate)

def to_vars(inputs, use_cuda, evaluate=False):
    return [to_var(inputs_, use_cuda, evaluate) for inputs_ in inputs]

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')

def get_args():
    parser = argparse.ArgumentParser(
        description='Attention Sum Reader for \
        Lambada dataset Using PyTorch')
    parser.register('type', 'bool', str2bool)

    parser.add_argument('-data', type=str, default='./data/train_data',
                        help='data directory containing input data')

    # word embedding
    parser.add_argument('-train_emb', type='bool', default=True,
                        help='whether to train word embed')
    parser.add_argument('-vocab_size', type=int, default=None,
                        help='size of vocabulary')

    # model file and log file
    parser.add_argument('-log_file', type=str, default=None,
                        help='log file')
    parser.add_argument('-restore_model', type=str, default=None)
    parser.add_argument('-save_model', type=str, default=None)
    parser.add_argument('-save_mode', type=str, choices=['all', 'best', 'series'], default='series')

    # model related
    parser.add_argument('-n_layers', type=int, default=1,
                        help='number of layers of the model')
    parser.add_argument('-gru_size', type=int, default=256,
                        help='size of word GRU hidden state')
    parser.add_argument('-grad_clip', type=float, default=10,
                        help='clip gradients at this value')

    # training related
    # batch size = 2
    parser.add_argument('-embed_dim', type=int, default=128, 
                        help='embedding dimension')
    parser.add_argument('-batch_size', type=int, default=1, 
                        help='mini-batch size')
    parser.add_argument('-epoch', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('-init_learning_rate', type=float, default=5e-4,
                        help='initial learning rate')
    parser.add_argument('-seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('-drop_out', type=float, default=0.1,
                        help='dropout rate')
    parser.add_argument('-punc', type='bool', default=True, 
                        help='filtered punctation in the answer')

    # print, visualize
    parser.add_argument('-print_every', type=int, default=1,
                        help='print frequency in batch')
    parser.add_argument('-use_board', default=False,
                        help='use tensor board')
    parser.add_argument('-board_name', default=None,
                        help='use tensor board')

    args = parser.parse_args()
    return args

def evaluate(model, batch_loader, use_cuda, is_dev):
    acc = loss = n_examples = 0

    for docs, anss, docs_mask, cand_position in batch_loader:
        batch_size = docs.shape[0]
        n_examples += batch_size

        docs, anss, docs_mask, cand_position = \
            to_vars([docs, anss, docs_mask, cand_position], 
                use_cuda=use_cuda, evaluate=True)

        loss_, acc_ = model(docs, anss, docs_mask, cand_position)

        _loss = loss_.cpu().data.numpy()[0]
        _acc = acc_.cpu().data.numpy()[0]

        loss += _loss
        acc += _acc

    if is_dev:
        return loss / len(batch_loader), acc / (n_examples + 830)
    else:
        return loss / len(batch_loader), acc / n_examples

def train(args, word_idx, train_data, valid_data, dev_data, writer = None):
	
	# build minibatch loader
    train_batch_loader = minibatch_loader(
        train_data, args.batch_size, sample=1.0, punc=args.punc)

    valid_batch_loader = minibatch_loader(
        valid_data, args.batch_size, shuffle=False, punc=args.punc)

    dev_batch_loader = minibatch_loader(
        dev_data, args.batch_size, shuffle=False, punc=args.punc)

    # training phase
    if args.restore_model != None:
        logging.info("restore from previous training...")

        _, embed_dim = load_word2vec_embeddings(word_idx, args.embed_file, args.embed_dim, False)

        model = AttSum(args.n_layers, args.vocab_size, args.drop_out, 
            args.gru_size, None, embed_dim, args.train_emb)

        checkpoint = torch.load(args.restore_model + '.chkpt')
        
        opt = torch.optim.Adam(
            params=filter(
                lambda p: p.requires_grad, model.parameters()
            ),
            lr=args.init_learning_rate)

        model.load_state_dict(checkpoint)
        '''
        model.load_state_dict(checkpoint['state_dict'])
        opt.load_state_dict(checkpoint['optimizer'])
        '''

    else:
        embed_init, embed_dim = load_word2vec_embeddings(word_idx, args.embed_file, args.embed_dim, True)

        logging.info("embedding dim: {}".format(embed_dim))
        logging.info("initialize model ...")
        model = AttSum(args.n_layers, args.vocab_size, args.drop_out, 
            args.gru_size, embed_init, embed_dim, args.train_emb)
        opt = torch.optim.Adam(
            params=filter(
                lambda p: p.requires_grad, model.parameters()
            ),
            lr=args.init_learning_rate)

    if USE_CUDA:
        model.cuda()
    logging.info("Running on cuda: {}".format(USE_CUDA))



    logging.info('-' * 50)
    logging.info("Start training ...")

    best_valid_acc = best_dev_acc = 0

    for epoch in range(args.epoch):
        '''
        if epoch >= 2:
            for param_group in opt.param_groups:
                param_group['lr'] /= 2
        '''

        model.train()
        train_acc = acc = train_loss = loss = n_examples = train_examples = it = 0
        start = time.time()



        for docs, anss, docs_mask, \
            cand_position in train_batch_loader:

            train_examples += docs.shape[0]
            n_examples += docs.shape[0]
        
            docs, anss, docs_mask, \
                cand_position = to_vars([docs, anss, docs_mask, \
                    cand_position], use_cuda=USE_CUDA)

            opt.zero_grad()

            loss_, acc_ = model(docs, anss, docs_mask, cand_position)
            
            train_loss += loss_.cpu().data.numpy()[0]
            loss += loss_.cpu().data.numpy()[0]
            train_acc += acc_.cpu().data.numpy()[0]
            acc += acc_.cpu().data.numpy()[0]
            it += 1
            
            loss_.backward()
            clip_grad_norm(
                parameters=filter(
                    lambda p: p.requires_grad, model.parameters()
                ),
                max_norm=args.grad_clip)
            opt.step()

            if (it % args.print_every == 0):
                # on training
                spend = (time.time() - start) / 60
                statement = "it: {} (max: {}), "\
                    .format(it, len(train_batch_loader))
                statement += "train loss: {:.3f}, acc: {:.3f}, time: {:.1f}(m)"\
                    .format(loss / float(args.print_every), acc / n_examples, spend)
                logging.info(statement)
                

                # on valid
                model.eval()
                start = time.time()
                valid_loss, valid_acc = evaluate(model, valid_batch_loader, USE_CUDA, False)
                spend = (time.time() - start) / 60
                
                logging.info("Valid loss: {:.3f}, acc: {:.3f}, time: {:.1f}(m)"
                    .format(valid_loss, valid_acc, spend))
                if best_valid_acc < valid_acc:
                    best_valid_acc = valid_acc
                    logging.info("Best valid acc: {:.3f}".format(best_valid_acc))

                # on lambada dev   
                start = time.time()             
                dev_loss, dev_acc = evaluate(model, dev_batch_loader, USE_CUDA, True)
                spend = (time.time() - start) / 60
                
                logging.info("dev loss: {:.3f}, acc: {:.3f}, time: {:.1f}(m)"
                    .format(dev_loss, dev_acc, spend))
                if best_dev_acc < dev_acc:
                    best_dev_acc = dev_acc
                    logging.info("Best dev acc: {:.3f}".format(best_dev_acc))
                    if args.save_mode == 'best':
                        model_name = args.save_model + '.chkpt'
                        torch.save(model.state_dict(), model_name)
                        logging.info('    - [Info] The checkpoint file has been updated [best].')

                if writer != None:
                    it_w = it / args.print_every
                    writer.add_scalar('data/train_loss', loss / float(args.print_every), it_w)
                    writer.add_scalar('data/train_acc', acc / n_examples, it_w)
                    writer.add_scalar('data/valid_loss', valid_loss, it_w)
                    writer.add_scalar('data/valid_acc', valid_acc, it_w)
                    writer.add_scalar('data/valid_loss', dev_loss, it_w)
                    writer.add_scalar('data/valid_acc', dev_acc, it_w)

                model.train()
                start = time.time()
                acc = loss = n_examples = 0
        
        logging.info("End: train loss: {:.3f}, acc: {:.3f}, time: {:.1f}(m)"
            .format(train_loss / len(train_batch_loader), train_acc / train_examples, spend))

        # on valid
        start = time.time()
        model.eval()
        valid_loss, valid_acc = evaluate(model, valid_batch_loader, USE_CUDA, False)
        spend = (time.time() - start) / 60
        
        logging.info("End: Valid loss: {:.3f}, acc: {:.3f}, time: {:.1f}(m)"
            .format(valid_loss, valid_acc, spend))
        if best_valid_acc < valid_acc:
            best_valid_acc = valid_acc
            logging.info("Best valid acc: {:.3f}".format(best_valid_acc))

        # on lambada dev
        start = time.time()
        dev_loss, dev_acc = evaluate(model, dev_batch_loader, USE_CUDA, True)
        spend = (time.time() - start) / 60
        
        logging.info("End: dev loss: {:.3f}, acc: {:.3f}, time: {:.1f}(m)"
            .format(dev_loss, dev_acc, spend))
        if best_dev_acc < dev_acc:
            best_dev_acc = dev_acc
            logging.info("Best dev acc: {:.3f}".format(best_dev_acc))

    #save checkpoint
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': opt.state_dict()}
    if args.save_model:
        if args.save_mode == 'series':
            model_name = args.save_model + '.chkpt'
            torch.save(model.state_dict(), model_name)
            #torch.save(checkpoint, model_name)
            logging.info('    - [Info] The checkpoint file has been updated [series].')
        elif args.save_mode == 'all':
            model_name = args.save_model + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_accu)
            torch.save(checkpoint, model_name)
            logging.info('    - [Info] The checkpoint file has been updated [all].')
        '''
        elif args.save_mode == 'best':
            model_name = args.save_model + '.chkpt'
            if valid_accu >= max(valid_accus):
                torch.save(checkpoint, model_name)
                print('    - [Info] The checkpoint file has been updated.')        
        '''

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    #========= Loading Dataset =========#
    data = torch.load(args.data)
    word_idx = data['dict']['all']
    train_data = data['train']
    valid_data = data['valid']
    dev_data = data['dev']
    
    #opt.max_token_seq_len = data['settings'].max_token_seq_len
    args.embed_file = data['settings'].vocab
    args.vocab_size = len(word_idx)

    if args.log_file is None:
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(message)s',
                            datefmt='%m-%d %H:%M')
    else:
        logging.basicConfig(filename=args.log_file,
                            filemode='w', level=logging.DEBUG,
                            format='%(asctime)s %(message)s',
                            datefmt='%m-%d %H:%M')
    logging.info(args)
    
    if args.use_board == 'True':
        if args.board_name != None:
            writer = SummaryWriter(args.board_name)
        writer = SummaryWriter()
        train(args, word_idx, train_data, valid_data, dev_data, writer)
        # export scalar data to JSON for external processing
        writer.export_scalars_to_json("./test.json")
        writer.close()
    else:
        train(args, word_idx, train_data, valid_data, dev_data, None)
    
if __name__ == '__main__':
    main()