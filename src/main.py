import os
import chromadb
from matplotlib.pylab import choice
from tqdm import tqdm
import torch
from src.utils.retrieval_methods.gold import article_dictionary_list, gold_retriever
from src.utils.utils import article_key_function, LACD_DATASET_PATH
from transformers import AutoTokenizer
from src.methods.LawGNN.article_network.article_network import ArticleNetwork
from src.encoders.cross_encoder.retriever.cross_retriever import cross_retriever

import polars as pl
import openai
import numpy as np
import math

class RetrievalContext:
    def __init__(self, args, reranker, reranker_tokenizer, article_network, chroma_collection, conflicts, chroma_db_name, laws_df=None):
        self.args = args
        self.reranker = reranker
        self.reranker_tokenizer = reranker_tokenizer
        self.article_network = article_network
        self.chroma_collection = chroma_collection
        self.conflicts = conflicts
        self.chroma_db_name = chroma_db_name
        # Load laws.csv once here (not per query): ~40MB parsed a single time
        # and shared by reference across all benchmark iterations.
        if laws_df is None:
            csv_path = getattr(args, "laws_csv_path", "./data/database/laws.csv")
            laws_df = pl.read_csv(csv_path, infer_schema_length=10000)
        self.laws_df = laws_df


def retrieve_top_conflicts(query, context, threashold=0):

    global top_conflicts_lens
    
    top_conflicts = list()
    # laws_df is loaded once in RetrievalContext.__init__ and reused by
    # reference. Lazy-cache here as a safety net for contexts built without
    # it (never re-read per query).
    if context.laws_df is None:
        csv_path = getattr(context.args, "laws_csv_path", "./data/database/laws.csv")
        context.laws_df = pl.read_csv(csv_path, infer_schema_length=10000)
    laws_df = context.laws_df
    batch_size = getattr(context.args, "batch_size", 32)

    if context.args.retrieval_method == "retrieval":

        article_vector, top_conflicts = binary_retriever(biencoder_model, laws_df, context.chroma_collection, query, biencoder_top_k, batch_size=batch_size, tokenizer = tokenizer)


    elif context.args.retrieval_method == "re2":

        if context.args.rex_method == "rocchio":
            from src.methods.ReX.rex_methods import rocchio_binary_retriever
            article_vector, top_conflicts = rocchio_binary_retriever(biencoder_model, laws_df, context.chroma_collection, query, biencoder_top_k, batch_size=batch_size, tokenizer = tokenizer)
            final_conflicts = cross_retriever(
                query, 
                top_conflicts, 
                cross_encoder_model=context.reranker, 
                tokenizer=context.reranker_tokenizer, 
                article_network=context.article_network, 
                index_method=crossencoder_index_method,
                batch_size=batch_size,
                query_vector=article_vector,
            )

            top_conflicts = sorted(final_conflicts, key=lambda x: x[1], reverse=True)
            if threashold == 0:
                top_conflicts = [c[0] for c in top_conflicts]
            else:
                top_conflicts = [c[0] for c in top_conflicts if c[1] > threashold]


        elif context.args.rex_method == "rex2":

            article_vector, top_conflicts = binary_retriever(biencoder_model, laws_df, context.chroma_collection, query, top_k=args.biencoder_top_k, batch_size=batch_size, tokenizer = tokenizer)

            top_conflicts_I = top_conflicts[:args.biencoder_top_k]

            predicts_from_reranker = cross_retriever(
                query, 
                top_conflicts_I, 
                cross_encoder_model=context.reranker, 
                tokenizer=context.reranker_tokenizer, 
                article_network=context.article_network, 
                index_method=crossencoder_index_method,
                batch_size=batch_size,
                query_vector=article_vector,
            )

            p_calib = lambda p, t: 1 / (1 + math.exp(-math.log(p / (1 - p)) / t))
            TEMPERATURE = 1

            minimum_score = min([p_calib(float(c[1]), TEMPERATURE) for c in predicts_from_reranker])

            predicts_from_reranker_filtering = [c[0] for c in predicts_from_reranker if p_calib(float(c[1]), TEMPERATURE) > minimum_score/0.75]

            # 
            prestige_articles = rex2(
                query=article_key_function(query),
                articles=[article_key_function(a) for a in top_conflicts],
                conflicts=context.conflicts,
                predicts_from_reranker={article_key_function(query): [article_key_function(a) for a in predicts_from_reranker_filtering]},
                k0=args.biencoder_top_k,
            )
            
            rex_top_conflicts = gold_retriever(prestige_articles)
            

            final_conflicts = predicts_from_reranker + cross_retriever(
                query, 
                rex_top_conflicts, 
                cross_encoder_model=context.reranker, 
                tokenizer=context.reranker_tokenizer, 
                article_network=context.article_network, 
                index_method=crossencoder_index_method,
                batch_size=batch_size,
                query_vector=article_vector,
            )

            final_conflicts = sorted(final_conflicts, key=lambda x: x[1], reverse=True)

            top_conflicts_lens.append(len(final_conflicts))

            
            if threashold == 0:
                top_conflicts = [c[0] for c in final_conflicts]
            else:
                top_conflicts = [c[0] for c in final_conflicts if c[1] > threashold]

        elif context.args.rex_method == "baseline":
            
            article_vector, top_conflicts = binary_retriever(biencoder_model, laws_df, context.chroma_collection, query, biencoder_top_k, batch_size=batch_size, tokenizer = tokenizer)

            final_conflicts = cross_retriever(
                query, 
                top_conflicts, 
                cross_encoder_model=context.reranker, 
                tokenizer=context.reranker_tokenizer, 
                article_network=context.article_network, 
                index_method=crossencoder_index_method,
                batch_size=batch_size,
                query_vector=article_vector,
            )

            top_conflicts = sorted(final_conflicts, key=lambda x: x[1], reverse=True)
            
            if threashold == 0:
                top_conflicts = [c[0] for c in top_conflicts]
            else:
                top_conflicts = [c[0] for c in top_conflicts if c[1] > threashold]
        else:
            assert(0)
        

    elif context.args.retrieval_method == "tfidf":
        from src.encoders.bi_encoder.retriever.classical_retriever import find_top_conflicts_with_tfidf as tfidf_retriever
        top_conflicts = tfidf_retriever(query, context.chroma_db_name, top_k=biencoder_top_k)

    elif context.args.retrieval_method == "bm25":
        from src.encoders.bi_encoder.retriever.classical_retriever import find_top_conflicts_with_bm25 as bm25_retriever
        top_conflicts = bm25_retriever(query, context.chroma_db_name, top_k=biencoder_top_k)

    else:
        assert(0)

    return top_conflicts


if __name__ == "__main__":

    query = """형법 제201조 (아편흡식 등, 동장소제공) ①아편을 흡식하거나 몰핀을 주사한 자는 5년 이하의 징역에 처한다.
②아편흡식 또는 모르핀주사의 장소를 제공하여 이익을 취할 경우에도 전항의 형과 같다."""


    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message=".*past_key_values.*")
    
    import argparse

    # argparse 설정
    parser = argparse.ArgumentParser(description="legal article conflict detector")

    # 모델 경로 동적으로 입력 받기
    parser.add_argument("--biencoder_model_path", type=str, help="Biencoder model path", default="./data/models/LACD-bi/kbb-baseline-nofinetune")
    parser.add_argument("--crossencoder_model_path", type=str, help="Crossencoder model path", default="./data/models/LACD-cross/roberta-42")
    parser.add_argument("--laws_csv_path", type=str, help="Path to the laws.csv file", default="./data/database/laws.csv")
    parser.add_argument("--chroma_db_name", type=str, help="Name of the Chroma DB where encodings will be stored", default="kbb-baseline-nofinetune")
    parser.add_argument("--biencoder_top_k", type=int, default=100, help="binary_encoder_top_k")

    # retrieval method
    parser.add_argument("--retrieval_method", type=str, choices=["retrieval", "re2", "tfidf", "bm25"], help="retrieval method. retrieval, re2, tfidf, or bm25", default="retrieval")
    parser.add_argument("--crossencoder_index_method", type=str, choices=["none", "vanilla", "gcn", "graphsage", "gat"], help="index usage for cross encoder. none means do not use index. vanilla means use index w/o GNNs.", default="none")
    parser.add_argument("--gnn_edge_way", type=str, choices=["both", "forward", "backward"], help="The way for edges in CAMGraph", default="both")

    parser.add_argument("--output_path", type=str, help="retrieval output paths", default="none")
    parser.add_argument("--mode", type=str, choices=["inference", "test-benchmark", "train-generate"], help="benchmark test or inference once", default="inference")

    # debug mode: limit the number of samples to ensure a fast run
    parser.add_argument("--debug", action="store_true", help="enable debug mode: run on limited samples")
    parser.add_argument("--debug_limit", type=int, default=5, help="number of samples to process in debug mode (only when --debug)")

    # subset sampling: stratified, no need to create a new dataset (alias mini_* deprecated)
    parser.add_argument("--subset_ratio", "--sample_ratio", "--mini_ratio", type=float, default=None, dest="subset_ratio", help="subset sampling: fraction (0,1] stratified sampling for test.jsonl, e.g. 0.15. More representative than --debug head (alias --mini_ratio deprecated)")
    parser.add_argument("--subset_seed", "--sample_seed", "--mini_seed", type=int, default=42, dest="subset_seed", help="seed for subset sampling (alias --mini_seed deprecated)")
    parser.add_argument("--subset_laws", "--mini_laws", "--law_nodes", type=int, default=None, dest="subset_laws", help="subset laws: limit number of articles in LMGraph (graph-aware) (alias --mini_laws deprecated)")

    parser.add_argument("--batch_size", type=int, default=32, help="batch size for encoding and retrieval operations (default 32)")

    # output location for retrieval results (avoid overwriting across experiments)
    parser.add_argument("--output_dir", type=str, default="./outputs/retrieval_results", help="directory to save retrieval results, e.g. ./outputs/my_experiment to avoid overwriting (default: ./outputs/retrieval_results)")
    parser.add_argument("--output_name", type=str, default=None, help="optional custom filename (without extension) for retrieval results; if not set, auto-generated from methods and suffix")

    parser.add_argument("--rex_method", type=str, choices=["baseline", "rex2", "rocchio"], default="baseline")
    parser.add_argument("--rex_conflict", type=str, choices=["train", "train-generate"], default="train")
    parser.add_argument("--multi-fold", type=bool, default=False)
    parser.add_argument("--multi-fold-k", type=int, default=5)
    parser.add_argument("--train_query_path", type=str, default="./data/datasets/LACD-retrieval/queries.jsonl")

    article_dictionary_list()

    args = parser.parse_args()
    # backward compat aliases (deprecated mini_*)
    args.mini_ratio = args.subset_ratio
    args.mini_laws = args.subset_laws
    args.mini_seed = args.subset_seed

    # 사용 예시
    biencoder_model_path = args.biencoder_model_path
    biencoder_top_k = args.biencoder_top_k
    crossencoder_model_path = args.crossencoder_model_path
    laws_csv_path = args.laws_csv_path
    chroma_db_name = args.chroma_db_name
    crossencoder_index_method = args.crossencoder_index_method

    # 결과를 jsonl 파일로 저장
    if args.output_path != "none":
        output_path = args.output_path
    else:
        output_path = "./outputs/retrieval_results/{0}_{1}_{2}".format(
            crossencoder_model_path.split("/")[-1], args.retrieval_method, args.rex_method
        )


    from src.utils.encoder.biencoder_utils import load_chromaDB_byname
    if biencoder_model_path in ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]:
        from src.encoders.bi_encoder.retriever.openai_binary_retriever import openai_biencoder_retriever as binary_retriever
        biencoder_model = openai.OpenAI()
        import tiktoken
        tokenizer = tiktoken.get_encoding('cl100k_base')        
    else:
        from src.encoders.bi_encoder.retriever.binary_retriever import binary_retriever
        biencoder_model = torch.load(biencoder_model_path + "/model.pth", weights_only=False)
        biencoder_model.eval()  # 모델을 평가 모드로 설정
        tokenizer = AutoTokenizer.from_pretrained(biencoder_model_path)




    chroma_collection = load_chromaDB_byname(chroma_db_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    top_conflicts_lens = []


    from src.methods.ReX.rex_methods import rex2
    if (args.rex_method == "rex2")  and args.rex_conflict == "train":
        from src.methods.ReX.hybrid_result import build_conflicts
        conflicts = build_conflicts()
    
    elif (args.rex_method == "rex2") and args.rex_conflict == "train-generate":
        from src.methods.ReX.hybrid_result import build_conflicts_by_generative_conflicts
        conflicts = build_conflicts_by_generative_conflicts(method = args.crossencoder_index_method)
    else:
        conflicts = None
    


    # from src.utils.utils import article_key_function

    article_network = ArticleNetwork(edge_way=args.gnn_edge_way, subset_laws=args.subset_laws, subset_seed=args.subset_seed)
    edge_index_tensor = article_network.create_edge_index()
    edge_index_tensor = edge_index_tensor.to(device)
    if args.subset_laws is not None:
        print(f"[SUBSET] ArticleNetwork nodes {len(article_network.all_article_keys)} subset_laws={args.subset_laws} edges {edge_index_tensor.shape[1]//2}")


    reranker = None
    reranker_tokenizer = None
    if args.retrieval_method in ["re2"]:
        reranker = torch.load(crossencoder_model_path + "/model.pth", weights_only=False)

        reranker_tokenizer = AutoTokenizer.from_pretrained(crossencoder_model_path)
        reranker.eval()
    laws_df = pl.read_csv(laws_csv_path, infer_schema_length=10000)
    context = RetrievalContext(args, reranker, reranker_tokenizer, article_network, chroma_collection, conflicts, chroma_db_name, laws_df=laws_df)

    if args.mode == "inference":
        top_conflicts = retrieve_top_conflicts(query, context)
        print("모순된 법률 article:")
        top_conflicts = top_conflicts[:10]
        for idx, contradiction in enumerate(top_conflicts, 1):
            print(f"{idx}: {contradiction}")
    
    elif args.mode == "test-benchmark" or args.mode == "train-generate":
        import json

        # 결과를 저장할 리스트
        result_list = []

        # queries.jsonl 읽기
        queries = []

        if args.multi_fold:
            with open("./data/datasets/LACD-biclassification/train-test-divide-fivefold/fold_{0}/queries.jsonl".format(args.multi_fold_k), "r", encoding="utf-8") as file:
                for line in file:
                    row = json.loads(line.strip())
                    queries.append(row["query"])
        elif args.mode == "train-generate":
            with open("./data/datasets/LACD-retrieval/train-queries.jsonl", "r", encoding="utf-8") as file:
                for line in file:
                    row = json.loads(line.strip())
                    queries.append(row["query"])
        else:
            with open("./data/datasets/LACD-retrieval/queries.jsonl", "r", encoding="utf-8") as file:
                for line in file:
                    row = json.loads(line.strip())
                    queries.append(row["query"])
        
        if args.subset_ratio is not None:
            import random
            rng = random.Random(args.subset_seed)
            orig_len = len(queries)
            sample_size = max(1, int(orig_len * args.subset_ratio))
            queries = rng.sample(queries, sample_size)
            print(f"[SUBSET] queries sampled {orig_len} -> {len(queries)} (ratio={args.subset_ratio}, seed={args.subset_seed})")

        if args.debug:
            print(f"[DEBUG] limiting test-benchmark queries to {args.debug_limit} samples")
            queries = queries[:args.debug_limit]

        # Query 수행
        for query in tqdm(queries):
            if args.mode == "train-generate":
                top_conflicts = retrieve_top_conflicts(query, context, threashold=0.5)
            else:
                top_conflicts = retrieve_top_conflicts(query, context)
            result_list.append({"article_to_check": query, "articles": top_conflicts})
            top_conflicts_lens.append(len(top_conflicts))

        # 5가지 summary statistics 출력
        if top_conflicts_lens:
            print(
                f"================================================",
                f"min={np.min(top_conflicts_lens)} ",
                f"max={np.max(top_conflicts_lens)} ",
                f"mean={np.mean(top_conflicts_lens):.2f} ",
                f"median={np.median(top_conflicts_lens)} ",
                f"std={np.std(top_conflicts_lens):.2f}",
                f"================================================",
                sep="\n"
            )

        import os
        suffix = "_debug" if args.debug else ""
        if args.subset_ratio is not None:
            suffix += f"_subset{int(args.subset_ratio*100)}"
        if args.subset_laws is not None:
            suffix += f"_laws{args.subset_laws}"
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if args.output_name:
            filename = f"{args.output_name}.jsonl" if not args.output_name.endswith(".jsonl") else args.output_name
            final_output_path = os.path.join(output_dir, filename)
        elif args.output_path != "none":
            final_output_path = args.output_path if args.output_path.endswith(".jsonl") else args.output_path + ".jsonl"
            os.makedirs(os.path.dirname(os.path.abspath(final_output_path)) or ".", exist_ok=True)
        else:
            filename = "{0}_{1}_{2}{3}{4}.jsonl".format(
                biencoder_top_k, crossencoder_index_method, args.rex_method,
                "_noLM" if "noLM" in crossencoder_model_path else "", suffix
            )
            final_output_path = os.path.join(output_dir, filename)

        print(f"[OUTPUT] saving retrieval results to {final_output_path}")
        with open(final_output_path, "w", encoding="utf-8") as outfile:
            for result in result_list:
                json.dump(result, outfile, ensure_ascii=False)
                outfile.write("\n")
