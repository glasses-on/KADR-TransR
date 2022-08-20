import sys
import torch
import random
import numpy as np
import copy

import torch
import torch.nn.functional as F


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Aggregator(torch.nn.Module):
    '''
    Aggregator class
    Mode in ['sum', 'concat', 'neighbor']
    '''

    def __init__(self, batch_size, dim, aggregator):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.dim = dim
        if aggregator == 'concat':
            self.weights = torch.nn.Linear(2 * dim, dim, bias=True)
        else:
            self.weights = torch.nn.Linear(dim, dim, bias=True)
        self.aggregator = aggregator

    def forward(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings, act):
        batch_size = user_embeddings.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        if self.aggregator == 'sum':
            output = (self_vectors + neighbors_agg).view((-1, self.dim))

        elif self.aggregator == 'concat':
            output = torch.cat((self_vectors, neighbors_agg), dim=-1)
            output = output.view((-1, 2 * self.dim))

        else:
            output = neighbors_agg.view((-1, self.dim))

        output = self.weights(output)
        return act(output.view((self.batch_size, -1, self.dim)))

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, user_embeddings):
        '''
        This aims to aggregate neighbor vectors
        '''
        # [batch_size, 1, dim] -> [batch_size, 1, 1, dim]
        user_embeddings = user_embeddings.view((self.batch_size, 1, 1, self.dim))

        # [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, n_neighbor]
        user_relation_scores = (user_embeddings * neighbor_relations).sum(dim=-1)
        user_relation_scores_normalized = F.softmax(user_relation_scores, dim=-1)

        # [batch_size, -1, n_neighbor] -> [batch_size, -1, n_neighbor, 1]
        user_relation_scores_normalized = user_relation_scores_normalized.unsqueeze(dim=-1)

        # [batch_size, -1, n_neighbor, 1] * [batch_size, -1, n_neighbor, dim] -> [batch_size, -1, dim]
        neighbors_aggregated = (user_relation_scores_normalized * neighbor_vectors).sum(dim=2)

        return neighbors_aggregated


class KGCN(torch.nn.Module):
    def __init__(self, num_user, num_ent, num_rel, kg, args, device):
        super(KGCN, self).__init__()
        self.num_user = num_user
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size

        self.dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.cf_l2loss_lambda = args.cf_l2loss_lambda

        self.n_neighbor = args.neighbor_sample_size
        self.kg = kg
        self.device = device
        self.aggregator = Aggregator(self.batch_size, self.dim, args.aggregator)

        self._gen_adj()

        self.rel = torch.nn.Embedding(num_rel, args.relation_dim)

        # Contains Users and Entities embedding, user-ids are shifted by num-entities count
        self.entity_user_embed = torch.nn.Embedding(self.num_ent + self.num_user, self.dim)
        self.trans_M = torch.nn.Parameter(torch.Tensor(self.num_rel, self.dim, self.relation_dim))

        torch.nn.init.xavier_uniform_(self.entity_user_embed.weight)
        torch.nn.init.xavier_uniform_(self.rel.weight)
        torch.nn.init.xavier_uniform_(self.trans_M)

    def _gen_adj(self):
        '''
        Generate adjacency matrix for entities and relations
        Only cares about fixed number of samples
        '''
        self.adj_ent = torch.empty(self.num_ent + self.num_user, self.n_neighbor, dtype=torch.long)
        self.adj_rel = torch.empty(self.num_ent + self.num_user, self.n_neighbor, dtype=torch.long)

        for e in self.kg:
            if len(self.kg[e]) >= self.n_neighbor:
                neighbors = random.sample(self.kg[e], self.n_neighbor)
            else:
                neighbors = random.choices(self.kg[e], k=self.n_neighbor)

            self.adj_ent[e] = torch.LongTensor([ent for ent, _ in neighbors])
            self.adj_rel[e] = torch.LongTensor([rel for _, rel in neighbors])

    def calc_cf_loss(self, u, v):
        '''
        input: u, v are batch sized indices for users and items
        u: [batch_size]
        v: [batch_size]
        '''
        batch_size = u.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        # change to [batch_size, 1]
        u = u.view((-1, 1))
        v = v.view((-1, 1))

        # [batch_size, dim]
        user_embeddings = self.entity_user_embed(u).squeeze(dim=1)

        entities, relations = self._get_neighbors(v)

        item_embeddings = self._aggregate(user_embeddings, entities, relations)

        scores = (user_embeddings * item_embeddings).sum(dim=1)

        return torch.sigmoid(scores)

    def _get_neighbors(self, v):
        '''
        v is batch sized indices for items
        v: [batch_size, 1]
        '''
        entities = [v]
        relations = []

        for h in range(self.n_iter):
            neighbor_entities = torch.LongTensor(self.adj_ent[entities[h]]).view((self.batch_size, -1)).to(self.device)
            neighbor_relations = torch.LongTensor(self.adj_rel[entities[h]]).view((self.batch_size, -1)).to(self.device)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)

        return entities, relations

    def calc_kg_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """

        r_embed = self.rel(r)  # (kg_batch_size, relation_dim)
        W_r = self.trans_M[r]  # (kg_batch_size, embed_dim, relation_dim)

        h_embed = self.entity_user_embed(h)  # (kg_batch_size, embed_dim)
        pos_t_embed = self.entity_user_embed(pos_t)  # (kg_batch_size, embed_dim)
        neg_t_embed = self.entity_user_embed(neg_t)  # (kg_batch_size, embed_dim)

        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)

        # Trans R

        # Equation (1)
        pos_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)  # (kg_batch_size)

        # Equation (2)
        # kg_loss = F.softplus(pos_score - neg_score)
        kg_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        kg_loss = torch.mean(kg_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(
            r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def _aggregate(self, user_embeddings, entities, relations):
        '''
        Make item embeddings by aggregating neighbor vectors
        '''
        entity_vectors = [self.entity_user_embed(entity) for entity in entities]
        relation_vectors = [self.rel(relation) for relation in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                act = torch.tanh
            else:
                act = torch.sigmoid

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                vector = self.aggregator(
                    self_vectors=entity_vectors[hop],
                    neighbor_vectors=entity_vectors[hop + 1].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    neighbor_relations=relation_vectors[hop].view((self.batch_size, -1, self.n_neighbor, self.dim)),
                    user_embeddings=user_embeddings,
                    act=act)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        return entity_vectors[0].view((self.batch_size, self.dim))

    def calc_score(self, u, v):
        """
        user_ids:  (n_users)
        item_ids:  (n_items)
        """
        '''
        input: u, v are batch sized indices for users and items
        u: [batch_size]
        v: [batch_size]
        '''
        batch_size = u.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        # change to [batch_size, 1]
        u = u.view((-1, 1))
        v = v.view((-1, 1))

        # [batch_size, dim]
        user_embeddings = self.entity_user_embed(u).squeeze(dim=1)

        entities, relations = self._get_neighbors(v)

        item_embeddings = self._aggregate(user_embeddings, entities, relations)

        # print("user_embeddings")
        # print(user_embeddings.size(), item_embeddings.size())
        # Equation (12)
        cf_score = torch.matmul(user_embeddings, item_embeddings.transpose(0, 1))  # (n_users, n_items)
        return cf_score

    def forward(self, *input, mode):
        if mode == 'train_cf':
            return self.calc_cf_loss(*input)
        if mode == 'train_kg':
            return self.calc_kg_loss(*input)
        if mode == 'predict':
            return self.calc_score(*input)
