import torch
import json
import tqdm
import csv

class ArticleNetwork:
    def __init__(self, laws_path="./data/database/laws.csv", law_link_path = "./data/database/law_link_20240930_duplicate_eliminate.jsonl", mini_laws=None, mini_seed=42):
        # Dictionary to store the network (adjacency list)
        self.article_network = {}
        # Dictionary to store the article key to index mapping
        self.article_key_to_idx = {}
        # List to store all article keys
        self.all_article_keys = []
        self.mini_laws = mini_laws
        self.mini_seed = mini_seed
        
        # Load all articles from laws_html.jsonl
        self._load_nodes(laws_path)
        self._load_edges(law_link_path)

    def _load_nodes(self, laws_path):
        # Early-load: hitung degree dulu (jika mini_laws), lalu streaming laws.csv hanya keep.
        # Tidak pernah menyimpan 79k lalu slice; hanya simpan yang dibutuhkan.
        import random as _random
        import collections as _collections
        # 1) hitung degree early jika diminta
        deg = None
        keep = None
        n_hub = 0
        if self.mini_laws is not None:
            try:
                deg = _collections.Counter()
                import json as _json
                with open("./data/database/law_link_20240930_duplicate_eliminate.jsonl", 'r', encoding='utf-8') as _f:
                    for _line in _f:
                        _d = _json.loads(_line)
                        deg[_d['source_key']] += 1
                        deg[_d['target_key']] += 1
            except Exception as e:
                deg = None
                print(f"[MINI] degree early-load failed {e}, fallback random")

        # 2) streaming laws.csv
        if self.mini_laws is not None and deg is not None:
            # kumpulkan semua keys dulu untuk tentukan hubs (79k str ~5MB, masih O(N) tapi sekali)
            all_keys_tmp = []
            with open(laws_path, encoding='utf-8', errors='ignore') as _f:
                reader = csv.DictReader(_f)
                for row in reader:
                    article_key = row['article_title'].replace("·", "ㆍ")
                    all_keys_tmp.append(article_key)
            total = len(all_keys_tmp)
            if self.mini_laws < total:
                sorted_keys = sorted(all_keys_tmp, key=lambda k: deg.get(k, 0), reverse=True)
                n_hub = min(self.mini_laws // 2, len(sorted_keys))
                hubs = sorted_keys[:n_hub]
                remaining = [k for k in all_keys_tmp if k not in set(hubs)]
                rnd = _random.Random(self.mini_seed)
                rnd.shuffle(remaining)
                keep = set(hubs + remaining[: self.mini_laws - n_hub])
                self.all_article_keys = [k for k in all_keys_tmp if k in keep]
                print(f"[MINI] laws early-load sampled -> {len(self.all_article_keys)} (hubs {n_hub} + random {self.mini_laws - n_hub}) seed={self.mini_seed} target {self.mini_laws} total {total}")
            else:
                self.all_article_keys = all_keys_tmp
        elif self.mini_laws is not None:
            # fallback random early-load tanpa degree
            all_keys_tmp = []
            with open(laws_path, encoding='utf-8', errors='ignore') as _f:
                reader = csv.DictReader(_f)
                for row in reader:
                    article_key = row['article_title'].replace("·", "ㆍ")
                    all_keys_tmp.append(article_key)
            rnd = _random.Random(self.mini_seed)
            rnd.shuffle(all_keys_tmp)
            self.all_article_keys = all_keys_tmp[: self.mini_laws]
            print(f"[MINI] laws random early-load -> {len(self.all_article_keys)} target {self.mini_laws}")
        else:
            # full mode: streaming tanpa sampling
            with open(laws_path, encoding='utf-8', errors='ignore') as _f:
                reader = csv.DictReader(_f)
                for row in reader:
                    article_key = row['article_title'].replace("·", "ㆍ")
                    self.all_article_keys.append(article_key)

        # Create a mapping from article key to index
        self.article_key_to_idx = {key: idx for idx, key in enumerate(self.all_article_keys)}
        # print(f"Loaded {len(self.all_article_keys)} articles as nodes.")
    
    def add_article_node(self, article):
        from src.utils.utils import article_key_function
        article_key = article_key_function(article)
        article_idx = len(self.all_article_keys)
        self.all_article_keys.append(article_key)

        # 맨 마지막에 넣음
        self.article_key_to_idx[article_key]=article_idx
        return article_idx
    
    def _load_edges(self, law_link_path):
        # """Load edges (connections) from law_link JSONL file"""
        # print("Loading article network edges...")
        skipped_edge_cnt = 0
        with open(law_link_path, 'r', encoding='utf-8') as f:
            for line in tqdm.tqdm(f):
                link_data = json.loads(line)
                source = link_data['source_key']
                target = link_data['target_key']
                
                if (source not in self.article_key_to_idx.keys()) or (target not in self.article_key_to_idx.keys()):
                    skipped_edge_cnt = skipped_edge_cnt +1
                    continue

                # Add source -> target connection
                if source not in self.article_network:
                    self.article_network[source] = []
                self.article_network[source].append(target)
        # print(f"Loaded edges for {len(self.article_network)} articles.")
        # print(f"{skipped_edge_cnt} edges are skipped.")

    def find_connected_articles(self, article_key):
        """Return the list of articles connected to a given article key"""
        return self.article_network.get(article_key, [])

    def create_edge_index(self):
        """Create edge index from the article network"""
        edge_index = [[], []]  # List to store the source and target of edges


        no_found_cnt = 0
        # Ensure all edges in the article network are added to the edge_index
        for source, targets in self.article_network.items():
            if source in self.article_key_to_idx:
                for target in targets:
                    if target in self.article_key_to_idx:
                        # Add source -> target
                        # 실험해볼 만한 것 역순으로 넣어야 전파가 된다.
                        edge_index[1].append(self.article_key_to_idx[source])
                        edge_index[0].append(self.article_key_to_idx[target])
                        
                        edge_index[0].append(self.article_key_to_idx[source])
                        edge_index[1].append(self.article_key_to_idx[target])                        
                    else:

                        no_found_cnt += 1
            else:
                no_found_cnt += 1


        # Convert to a tensor
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        return edge_index_tensor

if __name__ == "__main__":
    # Initialize the ArticleNetwork class
    article_network = ArticleNetwork()
        
    # Find connected articles for a specific key
    print(article_network.find_connected_articles("특정범죄 가중처벌 등에 관한 법률 제2조"))
    
    # Create edge index and print it
    print(article_network.create_edge_index())