import collections
import random

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from data_loader.loader_base import DataLoaderBase


class KGCNDataLoader(DataLoaderBase):
    def __init__(self, data_set, args, logging):
        super().__init__(args, logging)

        self.r_list = None
        self.t_list = None
        self.h_list = None
        self.train_kg_dict = None
        self.n_users_entities = None
        self.n_entities = None
        self.n_relations = None

        self.test_batch_size = args.test_batch_size
        self.kg_batch_size = args.kg_batch_size

        kg_data = self.load_kg(self.kg_file)
        print("construct_data Started")
        self.construct_data(kg_data)
        print("construct_data Finished")

        print("_build_dataset Started")
        self.df_dataset = self._build_dataset()
        print("_build_dataset Finished")

    def construct_data(self, kg_data):
        # add inverse kg data
        n_relations = max(kg_data['r']) + 1
        inverse_kg_data = kg_data.copy()
        inverse_kg_data = inverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        inverse_kg_data['r'] += n_relations
        kg_data = pd.concat([kg_data, inverse_kg_data], axis=0, ignore_index=True, sort=False)

        # re-map user id
        kg_data['r'] += 2
        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        self.n_users_entities = self.n_users + self.n_entities

        self.cf_train_data = (
            np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32),
            self.cf_train_data[1].astype(np.int32))
        self.cf_test_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32),
                             self.cf_test_data[1].astype(np.int32))

        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in
                                self.train_user_dict.items()}
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in
                               self.test_user_dict.items()}

        # add interactions to kg data
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]

        inverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        inverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        inverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, inverse_cf2kg_train_data], ignore_index=True)
        self.n_kg_train = len(self.kg_train_data)

        # construct kg dict
        h_list = []
        t_list = []
        r_list = []

        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for row in self.kg_train_data.iterrows():
            h, r, t = row[1]
            h_list.append(h)
            t_list.append(t)
            r_list.append(r)

            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)

    def load_kg_kgcn(self):
        return self.train_kg_dict

    def load_dataset(self):
        return self.df_dataset

    def _build_dataset(self):
        '''
        Build dataset for training (rating data)
        It contains negative sampling process
        '''
        print('Build dataset dataframe ...', end=' ')
        df_dataset_list = []

        user_to_item_map = collections.defaultdict(list)
        for user_id, pos_item_id_list in self.train_user_dict.items():  # This is the updated shifted user-id
            for this_pos_item_id in pos_item_id_list:
                df_dataset_list.append({
                    "userID": user_id,
                    "itemID": this_pos_item_id,
                    "label": 1
                })
                user_to_item_map[user_id].append(this_pos_item_id)

        for user_id, pos_item_id_list in self.test_user_dict.items():
            for this_pos_item_id in pos_item_id_list:
                df_dataset_list.append({
                    "userID": user_id,
                    "itemID": this_pos_item_id,
                    "label": 1
                })
                user_to_item_map[user_id].append(this_pos_item_id)

        # Add negative sampling of existing users
        def sample_neg_items_for_u(pos_items, user_id, n_sample_neg_items):
            sample_neg_items = []
            while True:
                if len(sample_neg_items) == n_sample_neg_items:
                    break

                neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
                    sample_neg_items.append(neg_item_id)
            return sample_neg_items

        for this_user in user_to_item_map.keys():
            pos_items_list = user_to_item_map[this_user]

            # we have len(pos_items_list) positive items for this user, we should have these many negative items too
            neg_item_id_list = sample_neg_items_for_u(pos_items_list, this_user, len(pos_items_list))
            if neg_item_id_list:
                for this_neg_item_id in neg_item_id_list:  # There exists N neg item in this list
                    df_dataset_list.append({
                        "userID": this_user,
                        "itemID": this_neg_item_id,
                        "label": 0
                    })

        df_dataset = pd.DataFrame(data=df_dataset_list, columns=["userID", "itemID", "label"])
        df_dataset = df_dataset.sample(frac=1, random_state=42).reset_index(drop=True)
        return df_dataset

    def get_num(self):
        return self.n_users, self.n_entities, self.n_relations


class KGCNDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        user_id = np.array(self.df.iloc[idx]['userID'])
        item_id = np.array(self.df.iloc[idx]['itemID'])
        label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
        return user_id, item_id, label
