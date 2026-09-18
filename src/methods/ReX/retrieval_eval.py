from src.methods.ReX.hybrid_result import build_true_dict, build_true_dict_by_test, compute_ndcg, classify_type
import json
from tqdm import tqdm
import copy
import os
from src.methods.LawGNN.article_network.article_network import ArticleNetwork
import pandas as pd


def main(result_path, article_network):
    top_k = [1,5,10,50, 100]
    top_k0 = [100]

    true_dicts_with_known_conflicts = build_true_dict()
    true_dicts = build_true_dict_by_test()

    # 결과 파일 읽기
    processed_query = set()

    results = []
    with open(result_path, 'r', encoding='utf-8') as result_file:
        for line in result_file:
            row = json.loads(line.strip())
            results.append(row)
            processed_query.add(row['article_to_check'])

    all_positives = 0
    for q in processed_query:
        all_positives += len(true_dicts[q])

    for k0 in top_k0:
        results_as_binary_markings = []
        all_instances = 0

        # 각 인스턴스(쿼리)에 대해 articles를 rerank하고 binary marking 계산
        for r in tqdm(results):
            query = r["article_to_check"]
            instance = {"article_to_check": query, "articles_as_binary": [], "articles_as_types": []}

            true_articles = set(true_dicts.get(query, []))
            true_articles_with_known_conflicts = set(true_dicts_with_known_conflicts.get(query, []))    
            
            for a in r["articles"]:
                if a in true_articles_with_known_conflicts and a not in true_articles:
                    continue

                instance["articles_as_binary"].append(1 if a in true_articles else 0)
                instance["articles_as_types"].append(
                    classify_type(query, a, article_network) if a in true_articles else "NONE"
                )
            all_instances += len(instance["articles_as_binary"])
            results_as_binary_markings.append(instance)

        print(f"all_positives: {all_positives}")
        print(f"all_instances: {all_instances}")
        
        for k in top_k:
            total_true_positives = 0
            total_retrieved = 0
            recall = 0
            recall_cap = 0
            total_ndcg = 0.0
            possible_positives = 0

            recall = 0
            recall_cap = 0
            
            for r in results_as_binary_markings:
                binary_list = r["articles_as_binary"]
                topk_list = binary_list[:k]
                total_true_positives += sum(topk_list)
                total_retrieved += len(topk_list)
                recall += sum(topk_list) / len(true_dicts[r['article_to_check']])
                recall_cap += sum(topk_list) / min(k, len(true_dicts[r['article_to_check']]))
                total_ndcg += compute_ndcg(binary_list, [1 for i in range(len(true_dicts[r['article_to_check']]))] + [0 for j in range(k-len(true_dicts[r['article_to_check']]))], k)

            recall = recall / len(results_as_binary_markings) * 100
            recall_cap = recall_cap / len(results_as_binary_markings) * 100
            precision = (total_true_positives / total_retrieved * 100) if total_retrieved > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            ndcg = total_ndcg / len(results_as_binary_markings) * 100

            print(f"Top-{k}: true positives = {total_true_positives}, retrievals = {total_retrieved} "
                f"Recall = {recall:.2f}%, RecallCap = {recall_cap:.2f}%, Precision = {precision:.2f}%, F1 = {f1:.2f}%, nDCG = {ndcg:.2f}%")

            # ------------------ CSV 파일에 각 article_to_check 별로 k에서 맞춘 article 수를 저장 ------------------
            per_query_rows = []
            for instance in results_as_binary_markings:
                row = {"article_to_check": instance["article_to_check"]}
                for k_val in top_k:
                    row[f"correct_top_{k_val}"] = sum(instance["articles_as_binary"][:k_val])
                per_query_rows.append(row)

            df_per_query = pd.DataFrame(per_query_rows)
            tag = result_path.split("/")[-1].split(".")[0]
            os.makedirs("./results", exist_ok=True)
            csv_path_per_query = f"./results/stats_{tag}.csv"
            df_per_query.to_csv(csv_path_per_query, index=False, encoding="utf-8")

    return results_as_binary_markings, true_dicts, processed_query

def generate_type_stats(all_results, article_network):
    top_k = [1, 5, 10, 50, 100]
    
    # 모든 메소드의 통계를 저장할 리스트
    all_stats_rows = []
    
    for method_name, (results_as_binary_markings, true_dicts, processed_query) in all_results.items():
        # ------------------ CSV 파일에 top-k 별 retrieved type 정보와 전체 true article type 정보 저장 ------------------
        # retrieved type 통계: 각 top-k 값별로
        types_count = {}

        for instance in results_as_binary_markings:
            query = instance.get("article_to_check")
            true_list = true_dicts[query]
            true_types_query = {}

            for t_article in true_list:
                article_type = classify_type(query, t_article, article_network)
                true_types_query[article_type] = true_types_query.get(article_type, 0) + 1

            for typ in true_types_query.keys():
                types_count[typ] = types_count.get(typ, 0) + 1

        retrieved_stats = {k: {} for k in top_k}
        for instance in results_as_binary_markings:
            query = instance.get("article_to_check")
            true_types_query = {}
            true_list = true_dicts[query]
            for t_article in true_list:
                article_type = classify_type(query, t_article, article_network)
                true_types_query[article_type] = true_types_query.get(article_type, 0) + 1

            types_list = instance.get("articles_as_types", [])
            for k in top_k:
                # Count true positives per type within the top-k slice
                slice_list = types_list[:min(k, len(types_list))]
                type_true_positives = {}
                for t in slice_list:
                    if t != "NONE":
                        type_true_positives[t] = type_true_positives.get(t, 0) + 1

                all_count = 0
                # Accumulate normalized recall per type
                for typ in ["CRIMINAL", "CRIMINAL_ACT_MENTION", "MENTION", "OTHERS", "NONE"]:
                    true_count = true_types_query.get(typ, 0)
                    if true_count == 0:
                        continue
                    retrieved_count = type_true_positives.get(typ, 0)
                    # Increment macro recall accumulator
                    retrieved_stats[k][typ] = retrieved_stats[k].get(typ, 0) \
                        + retrieved_count / (true_count* types_count.get(typ, 0))
                    
                    all_count = all_count + retrieved_count
                
                retrieved_stats[k]["OVERALL"] = retrieved_stats[k].get("OVERALL", 0) + all_count/(len(true_dicts[query])* len(results_as_binary_markings))

        # 전체 true article pair 에 대한 type 통계 (중복 없이)
        true_types_overall = {}
        for query in processed_query:
            true_list = true_dicts[query]
            for t_article in true_list:
                article_type = classify_type(query, t_article, article_network)
                true_types_overall[article_type] = true_types_overall.get(article_type, 0) + 1

        # retrieved 통계와 true 통계를 합쳐, 각 article type에 대해 한 행으로 정리합니다.
        # 행의 컬럼: type, retrieved_top_1, retrieved_top_5, retrieved_top_10, retrieved_top_50, true_count
        all_types = set()
        for k in top_k:
            all_types.update(retrieved_stats[k].keys())
        all_types.update(true_types_overall.keys())

        for typ in sorted(all_types):
            row = {
                "method": method_name,
                "type": typ,
                "true_count": true_types_overall.get(typ, 0)
            }
            for k in top_k:
                row[f"macro_recall_top_{k}"] = retrieved_stats[k].get(typ, 0)
            all_stats_rows.append(row)

    # 모든 메소드의 통계를 하나의 DataFrame으로 만들고 CSV로 저장
    df_type_stats = pd.DataFrame(all_stats_rows)
    os.makedirs("./results", exist_ok=True)
    csv_path = "./results/typestats_all.csv"
    df_type_stats.to_csv(csv_path, index=False, encoding="utf-8")

if __name__ == "__main__":
    import os

    input_dir = "./outputs/retrieval_results/t-test/"
    article_network = ArticleNetwork()

    # 해당 디렉토리 내 모든 파일 경로를 리스트로 수집
    paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
    ]
    paths.sort()
    
    all_results = {}
    for p in paths:
        try:
            print(p)
            method_name = p.split("/")[-1].split(".")[0]
            all_results[method_name] = main(p, article_network)
        except Exception as e:
            print(f"Error processing {p}: {e}")
            continue
    
    # 모든 메소드에 대한 typestats 한번에 생성
    generate_type_stats(all_results, article_network)
