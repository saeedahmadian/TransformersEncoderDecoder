# refrence : http://peterbloem.nl/blog/transformers

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self,k,head):
        super(SelfAttention,self).__init__()
        self.k= k
        self.head=head
        self.W_query= nn.Linear(k,head*k)
        self.W_key = nn.Linear(k,head*k)
        self.W_value= nn.Linear(k,head*k)
        self.unifyHeads = nn.Linear(head*k,k)

    def forward(self,sequence):
        b,t,k= sequence.size()
        h= self.head
        ## query is what we consider as the effect of xt (with k dimenstion) on yt (with k dimension)
        query = self.W_query(sequence)
        ## key is what we consider as the effect of xt on yt' (other outputs)
        key = self.W_key(sequence)
        ## value is what we consider as real value of x that we calculated weighted mean of x as y
        ## yi= sum(wij*xj) j belongs to 1,2,.., len(seq)
        value= self.W_value(sequence)
        # convert b,t,h*k to b,t,h,k

        query_ = query.view(b,t,h,k)
        key_ = key.view(b,t,h,k)
        value_ = value.view(b,t,h,k)
        # move h to batch side so that we can apply matrix operation
        # you can use contiguous or reshape doesn't matter
        query_= query_.transpose(1,2).contiguous().view(b*h,t,k)/(k**(1/4))
        key_ = key_.transpose(1,2).reshape(b*h,t,k)/(k**(1/4))
        value_ = value_.transpose(1,2).contiguous().view(b*h,t,k)
        ## calcualte wieghts over input sequence and remove dimension of k
        ## raw weights would look like b*h, t, t becuase => b*h,t,k dot b*h,k,t => b*h,t,t
        raw_weights = torch.bmm(query_,key_.transpose(1,2))
        # probabilities would be calculated from raw weights over columns
        # probs dimension is also b*h,t,t which each row sum would be 1
        # because we softmaxed over columns
        probs = F.softmax(raw_weights,dim=2)

        ## dot probs by value
        ## b*h,t,t dot b*h,t,k => b*h,t,k=> b,h,t,k
        value_prob= torch.bmm(probs,value_).view(b,h,t,k)
        # doesn't matter swap 1 to 2 or 2 to 1
        value_prob = value_prob.transpose(2,1).contiguous().view(b,t,h*k)

        out = self.unifyHeads(value_prob)

        return out



class TransformerBlock(nn.Module):
    def __init__(self,k,heads=8):
        super(TransformerBlock,self).__init__()
        self.attention = SelfAttention(k,heads)
        self.normLayer1= nn.LayerNorm(k)
        self.normLayer2 = nn.LayerNorm(k)
        self.feedforward=nn.Sequential(
            nn.Linear(k,4*k),
            nn.ReLU(),
            nn.Linear(4*k,k)
        )
    def forward(self,sequence):


        self_attended = self.attention(sequence)
        x = self.norm1(self_attended + sequence)

        fedforward = self.ff(x)
        return self.norm2(fedforward + x)

## this is for classification

class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
        super().__init__()

        self.num_tokens = num_tokens
        self.token_emb = nn.Embedding(num_tokens, k)
        self.pos_emb = nn.Embedding(seq_length, k)

        # The sequence of transformer blocks that does all the
        # heavy lifting
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)

        # Maps the final output sequence to class logits
        self.toprobs = nn.Linear(k, num_classes)

    def forward(self, x):
        """
        :param x: A (b, t) tensor of integer values representing
                  words (in some predetermined vocabulary).
        :return: A (b, c) tensor of log-probabilities over the
                 classes (where c is the nr. of classes).
        """
        # generate token embeddings
        tokens = self.token_emb(x)
        b, t, k = tokens.size()

        # generate position embeddings
        positions = torch.arange(t)

        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

        x = tokens + positions
        x = self.tblocks(x)

        # Average-pool over the t dimension and project to class
        # probabilities
        x = self.toprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)





seq_length = 100
batch_size= 32
embeding_dim= 512

sentence = torch.ones((batch_size,seq_length,embeding_dim))
attention= SelfAttention(k=embeding_dim,head=8)

embeded_seq = attention(sentence)

a=1











