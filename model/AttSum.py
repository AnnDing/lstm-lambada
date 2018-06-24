import torch
import torch.nn as nn
from utils.model_helper import *

class AttSum(nn.Module):
    def __init__(self, n_layers, vocab_size, dropout,
                 gru_size, embed_init, embed_dim, train_emb):
        super(AttSum, self).__init__()

        self.gru_size = gru_size
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.train_emb = train_emb
        self.n_vocab = vocab_size

        self.dropout = nn.Dropout(dropout)
        self.embed = nn.Embedding(vocab_size, embed_dim)

        if embed_init is not None:
            self.embed.weight.data.copy_(torch.from_numpy(embed_init))

        if not train_emb:
            self.embed.weight.requires_grad = False

        self.main_doc_layers = nn.ModuleList()

        main_input_feat = embed_dim

        for layer in range(n_layers):
            layer_doc = nn.GRU(
                input_size=main_input_feat,
                hidden_size=self.gru_size,
                batch_first=True,
                bidirectional=True)
            self.main_doc_layers.append(layer_doc)

        # final layer
        self.final_doc_layer = nn.GRU(
            input_size=main_input_feat,
            hidden_size=self.gru_size,
            batch_first=True,
            bidirectional=True)

        self.final_attention = nn.Linear(self.gru_size * 2, 1, bias=False)

    #(docs, quess, anss, docs_mask, quess_mask, cand_position, cands_mask)
    def forward(self, docs, anss, docs_mask, cand_position, test=False):
        doc_embed = self.embed(docs.long())

        for layer in range(self.n_layers - 1):
            doc_embed, _ = gru(
                doc_embed, docs_mask, self.main_doc_layers[layer])
        
        doc_embed, _ = gru(
            doc_embed, docs_mask, self.final_doc_layer)
        doc_embed = self.dropout(doc_embed)

        scores = self.final_attention(doc_embed)

        docs_mask = docs_mask.data.eq(Constants.PAD).unsqueeze(1)

        scores.data.masked_fill_(docs_mask, -float('inf'))

        attention = torch.nn.functional.softmax(scores, dim = 1)

        pred = attention_sum(attention, cand_position)

        loss = torch.mean(crossentropy(pred, anss.long())) 

        _, pred_ans = torch.max(pred, dim=1)

        acc = torch.sum(torch.eq(pred_ans, anss.long()))

        if test:
            anss_index = anss.cpu().data.numpy()[0]
            anss_pos = (cand_position[0, anss_index] == 1).nonzero()

            pred = pred.view(pred.size(1)).cpu().data.numpy()
            top_10_can = pred.argsort()[-10:][::-1]
            top_10_score = []
            top_10_pos = []
            for i in top_10_can:
                top_10_score += [pred[i]]
                top_10_pos += [(cand_position[0, i] == 1).nonzero().cpu().data.numpy().tolist()]
            
            pred_ans = pred_ans.cpu().data.numpy()[0]

            n_top2_correct = 0
            n_top5_correct = 0
            n_top10_correct = 0

            if anss_index in top_10_can[:2]:
                n_top2_correct = 1
            if anss_index in top_10_can[:5]:
                n_top5_correct = 1
            if anss_index in top_10_can:
                n_top10_correct = 1

            return top_10_score, top_10_pos, loss, acc,\
                n_top2_correct, n_top5_correct, n_top10_correct 
        else:
            return loss, acc
