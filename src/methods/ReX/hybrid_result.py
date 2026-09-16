import json
import math
import argparse
import copy
import sys
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.utils import article_key_function
from src.utils.retrieval_methods.gold import article_dictionary_list, gold_retriever
import re

from src.methods.LawGNN.article_network.article_network import ArticleNetwork


def compute_ndcg(relevances, ideal_relevances, k):
    dcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevances[:k]))
    idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_relevances[:k]))
    return dcg / idcg if idcg > 0 else 0.0

def classify_type(article1, article2, article_network):

    criminal = article1.startswith("형법") and article2.startswith("형법")
    mention_others = (article2 in article_network.find_connected_articles(article1))


    if criminal and mention_others:
        return "CRIMINAL_ACT_MENTION"
    elif mention_others:
        return "MENTION"
    elif criminal:
        return "CRIMINAL"
    else:
        return "OTHERS"


def build_conflicts():
    train_data_path = "./data/datasets/LACD-biclassification/train-test-divide-refine/train.jsonl"
    val_data_path = "./data/datasets/LACD-biclassification/train-test-divide-refine/val.jsonl"
    true_dicts_for_filtering_rows = load_jsonl(train_data_path) + load_jsonl(val_data_path)
    true_dicts_for_filtering_rows = [
        {
            "article1": article_key_function(r["article1"]),
            "article2": article_key_function(r["article2"]),
            "answer": r["answer"]
        }
        for r in true_dicts_for_filtering_rows
    ]
    true_dicts_for_filtering_rows = [r for r in true_dicts_for_filtering_rows if r["answer"] is True]

    true_dicts_for_filtering = {}
    for r in true_dicts_for_filtering_rows:
        a1 = r["article1"]
        a2 = r["article2"]
        true_dicts_for_filtering.setdefault(a1, []).append(a2)
        true_dicts_for_filtering.setdefault(a2, []).append(a1)

    return true_dicts_for_filtering


def build_conflicts_by_generative_conflicts(method = "none"):
    """Read conflict relationship data from test.jsonl file and create prefix-based dictionary"""
    if method == "none":
        data_path = "./outputs/retrieval_results/generative-conflicts/re2-0.jsonl"
    elif method == "gat":
        data_path = "./outputs/retrieval_results/generative-conflicts/re2-gat-0.jsonl"
    elif method == "gcn":
        data_path = "./outputs/retrieval_results/generative-conflicts/re2-gcn-0.jsonl"
    elif method == "graphsage":
        data_path = "./outputs/retrieval_results/generative-conflicts/re2-graphsage-0.jsonl"
    else:
        assert(0)


    test_data = load_jsonl(data_path)
    if test_data is None:
        raise ValueError("Failed to load data file")
    processed_rows = []
    for r in test_data:
        if "article_to_check" in r and "articles" in r:
            
            key1 = article_key_function(r["article_to_check"])

            for article in r["articles"]:
                key2 = article_key_function(article)
                processed_rows.append({"article1": key1, "article2": key2, "answer": True})
        else:
            print(f"Warning: Skipping data missing required keys - {r}", file=sys.stderr)
    true_rows = [r for r in processed_rows if r["answer"] is True]
    true_dicts = {}
    for r in true_rows:
        a1, a2 = r["article1"], r["article2"]
        true_dicts.setdefault(a1, []).append(a2)
        true_dicts.setdefault(a2, []).append(a1)
    for key in true_dicts:
        true_dicts[key] = list(set(true_dicts[key]))
    return true_dicts


def build_true_dict_by_test():
    test_data_path  = "./data/datasets/LACD-biclassification/train-test-divide-refine/test.jsonl"
    test_data = load_jsonl(test_data_path)
    if test_data is None:
        raise ValueError("Failed to load data file")
    processed_rows = []
    for r in test_data:
        if "article1" in r and "article2" in r and "answer" in r:
            try:
                key1 = article_key_function(r["article1"])
                key2 = article_key_function(r["article2"])
                processed_rows.append({"article1": key1, "article2": key2, "answer": r["answer"]})
            except Exception as e:
                print(f"Warning: Error applying article_key_function (data: {r}) - {e}", file=sys.stderr)
        else:
            print(f"Warning: Skipping data missing required keys - {r}", file=sys.stderr)
    true_rows = [r for r in processed_rows if r["answer"] is True]
    true_dicts = {}
    for r in true_rows:
        a1, a2 = r["article1"], r["article2"]
        true_dicts.setdefault(a1, []).append(a2)
        true_dicts.setdefault(a2, []).append(a1)
    for key in true_dicts:
        true_dicts[key] = list(set(true_dicts[key]))
    return true_dicts


def build_true_dict():
    """Read conflict relationship data from JSONL file and create prefix-based dictionary"""
    raw_links_pairs = "./data/datasets/LACD-biclassification/checker-generated/raw_links_small.jsonl"
    missed_data_path = "./data/datasets/LACD-retrieval/missed_pairs.jsonl"
    raw_data = load_jsonl(raw_links_pairs)
    missed_data = load_jsonl(missed_data_path)
    if raw_data is None or missed_data is None:
        raise ValueError("Failed to load data file")
    rows = raw_data + missed_data
    processed_rows = []
    for r in rows:
        if "article1" in r and "article2" in r and "answer" in r:
            try:
                key1 = article_key_function(r["article1"])
                key2 = article_key_function(r["article2"])
                processed_rows.append({"article1": key1, "article2": key2, "answer": r["answer"]})
            except Exception as e:
                print(f"Warning: Error applying article_key_function (data: {r}) - {e}", file=sys.stderr)
        else:
            print(f"Warning: Skipping data missing required keys - {r}", file=sys.stderr)
    true_rows = [r for r in processed_rows if r["answer"] is True]
    true_dicts = {}
    for r in true_rows:
        a1, a2 = r["article1"], r["article2"]
        true_dicts.setdefault(a1, []).append(a2)
        true_dicts.setdefault(a2, []).append(a1)
    for key in true_dicts:
        true_dicts[key] = list(set(true_dicts[key]))
    return true_dicts
    




def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data



def build_conflicts_test_generate(index_method = "none", top_k = 5):

    if index_method == "none":
        data_path = "./outputs/retrieval_results/article_key/roberta-0_re2_baseline.jsonl"
    elif index_method == "gat":
        data_path = "./outputs/retrieval_results/article_key/roberta-gat-0_re2_baseline.jsonl"
    else:
        assert(0)
    retrieved_articles = load_jsonl(data_path)

    true_dicts_for_filtering_rows = []

    for row in retrieved_articles:
        for i in range(top_k):
            instance = {"article1": row['article_to_check'], "article2": row["articles"][i]}
            true_dicts_for_filtering_rows.append(instance)

    true_dicts_for_filtering = {}
    for r in true_dicts_for_filtering_rows:
        a1 = r["article1"]
        a2 = r["article2"]
        true_dicts_for_filtering.setdefault(a1, []).append(a2)
        true_dicts_for_filtering.setdefault(a2, []).append(a1)
    return true_dicts_for_filtering