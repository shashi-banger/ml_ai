import json
import re
import random
import numpy as np
from collections import Counter
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

class EmbedModel(nn.Module):
    E_DIM = 30
    def __init__(self, device, num_links, num_movies):
        super().__init__()
        self.num_movies = num_movies
        self.device = device
        self.embed_links = nn.Embedding(num_links, EmbedModel.E_DIM)
        self.embed_movies = nn.Embedding(num_movies, EmbedModel.E_DIM)

    def forward(self, movie_batch, link_batch):
        m = self.embed_movies(movie_batch)
        l = self.embed_links(link_batch)
        m = m.div(m.norm(dim=1, keepdim=True))
        l = l.div(l.norm(dim=1, keepdim=True))
        res = torch.bmm(m.view(-1, 1, EmbedModel.E_DIM), l.view(-1, EmbedModel.E_DIM, 1))
        return res.view(-1)

    def print_embedding_example(self):
        input = torch.LongTensor([1]).to(self.device)
        print(self.embed_movies(input))


    def nearest_neighbors(self, index):
        """
        :param index: list of indices of movies
        """
        input = self.embed_movies(torch.LongTensor(index).to(self.device))
        input = input.div(input.norm(dim=1, keepdim=True))
        embedding = self.embed_movies(torch.LongTensor(
                        list(range(0, self.num_movies))).to(self.device))
        embedding = embedding.div(embedding.norm(dim=1, keepdim=True))

        print(input.shape, embedding.shape)
        res = torch.mm(input, embedding.transpose(0,1))
        nearest_movie_indices = res.argsort(dim=1)[:, -30:]
        return res, nearest_movie_indices


class MovieEmbeddings:
    def batchifier(self, positive_samples=10,negative_ratio=5):
        batch_size = positive_samples * (1 + negative_ratio)
        batch = np.zeros((batch_size, 3))
        while True:
            for idx, (link_id, movie_id) in enumerate(random.sample(self.pairs_set, positive_samples)):
                batch[idx, :] = (link_id, movie_id, 1)
            idx = positive_samples
            while idx < batch_size:
                movie_id = random.randrange(len(self.movie_to_idx))
                link_id = random.randrange(len(self.top_links))
                if not (link_id, movie_id) in self.pairs_set:
                    batch[idx, :] = (link_id, movie_id, -1)
                    idx += 1
            np.random.shuffle(batch)
            yield {'link': torch.tensor(batch[:, 0],
                                        dtype=torch.int64).to(self.device), 
                   'movie': torch.tensor(batch[:, 1],
                                         dtype=torch.int64).to(self.device)}, \
                            torch.tensor(batch[:,2], dtype=torch.float).to(self.device)
        
    def __init__(self, data_file):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.movies = []
        self.loss_func = F.mse_loss
        with open(data_file) as fd:
            for l in fd:
                try:
                    movie = json.loads(l)
                    if re.match('eng', movie[1]['language'].lower()):
                        self.movies.append(movie)
                except:
                    continue
            print(f"Total number of movies= {len(self.movies)}")

        link_counts = Counter()
        for movie in self.movies:
            link_counts.update(movie[2])
        self.top_links = [link for link, c in link_counts.items() if c >= 7]
        print(f"Number of top links = {len(self.top_links)}")
        self.link_to_idx = {link: idx for idx, link in enumerate(self.top_links)}
        self.movie_to_idx = {movie[0]: idx for idx, movie in enumerate(self.movies)}
        pairs = []
        for movie in self.movies:
            pairs.extend([(self.link_to_idx[link], self.movie_to_idx[movie[0]]) 
                for link in movie[2] if link in self.link_to_idx])
        self.pairs_set = set(pairs)
        print(f"Number of (movie, link) pairs = {len(self.pairs_set)}")
        with open("movies.txt", "w") as fd:
            for m, i in self.movie_to_idx.items():
                fd.write(f"{m}, {i}\n") 

        with open("links.txt", "w") as fd:
            for m, i in self.link_to_idx.items():
                fd.write(f"{m}, {i}\n") 

        with open("m_l_pairs.txt", "w") as fd:
            for l, m in self.pairs_set:
                fd.write(f"{l}, {m}\n") 


    def train(self):
        model = EmbedModel(self.device, len(self.top_links), len(self.movies))
        model.to(self.device)
        opt = optim.Adam(model.parameters(), lr=0.1)

        max_loop = len(self.movies)
        b = self.batchifier(positive_samples=2048, negative_ratio=10)
        for xb, yb in b:
            pred = model(xb['movie'], xb['link'])
            loss = self.loss_func(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()
            if (max_loop % 100) == 0:
                test_indices = [self.movie_to_idx['The Shawshank Redemption'],
                                self.movie_to_idx['Rear Window'],
                                self.movie_to_idx['Star Trek: First Contact']
                               ]
                nearest_dist, nearest_movies = model.nearest_neighbors(test_indices)
                nearest_dist = nearest_dist.cpu()
                nearest_movies = nearest_movies.cpu()
                for i, t in enumerate(test_indices):
                    print(f"\n\nMovies similar to {self.movies[t][0]}")
                    nm_np = nearest_movies[i].numpy()
                    for j in nm_np[::-1]:
                        print(f"{self.movies[j][0]}, {nearest_dist[i][j]}")

            max_loop -= 1

    


if __name__ == "__main__":
    import sys
    mov_embed = MovieEmbeddings(sys.argv[1])
    mov_embed.train()
