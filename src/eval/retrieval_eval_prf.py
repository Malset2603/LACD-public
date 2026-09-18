from src.methods.ReX.hybrid_result import build_true_dict, build_true_dict_by_test, compute_ndcg, classify_type
import json
from tqdm import tqdm
import copy
import os
from src.methods.LawGNN.article_network.article_network import ArticleNetwork
from tabulate import tabulate
import numpy as np
from scipy import stats

# ANSI escape codes
BLUE_BOLD = '\033[1;34m'
UNDERLINE = '\033[4m'
RESET = '\033[0m'

def highlight_table(table, header):
    # table: List[List[str or float]]
    # header: List[str]
    # 각 column별로 float값만 추출
    num_cols = len(header)
    col_values = [[] for _ in range(num_cols)]
    for row in table:
        for i, val in enumerate(row):
            try:
                # ±로 구분된 평균값만 추출
                if isinstance(val, str) and '±' in val:
                    col_values[i].append(float(val.split('±')[0]))
                else:
                    col_values[i].append(float(val))
            except:
                col_values[i].append(None)
    # 각 col별로 최고, 두번째 값 인덱스 찾기 (0번은 모델명)
    best_idx = {}
    second_idx = {}
    for j in range(1, num_cols):
        vals = [(i, v) for i, v in enumerate(col_values[j]) if v is not None]
        if not vals:
            continue
        vals_sorted = sorted(vals, key=lambda x: x[1], reverse=True)
        if len(vals_sorted) > 0:
            best_idx[j] = vals_sorted[0][0]
        if len(vals_sorted) > 1:
            second_idx[j] = vals_sorted[1][0]
    # 스타일 적용
    styled_table = []
    for i, row in enumerate(table):
        styled_row = []
        for j, val in enumerate(row):
            sval = str(val)
            if j == 0:
                styled_row.append(sval)
            elif j in best_idx and best_idx[j] == i:
                styled_row.append(f"{BLUE_BOLD}{sval}{RESET}")
            elif j in second_idx and second_idx[j] == i:
                styled_row.append(f"{UNDERLINE}{sval}{RESET}")
            else:
                styled_row.append(sval)
        styled_table.append(styled_row)
    return styled_table

def main(result_path, article_network):
    top_k = [5, 10, 50]
    top_k0 = [50]

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

    # metric 결과 저장용 딕셔너리
    table_metrics = {k: {"nDCG": 0.0, "Recall": 0.0, "Retrieval F1": 0.0} for k in top_k}

    for k0 in top_k0:
        results_as_binary_markings = []
        all_instances = 0

        # 각 인스턴스(쿼리)에 대해 articles를 rerank하고 binary marking 계산

        true_articles_count = 0
        for r in tqdm(results):
            query = r["article_to_check"]
            instance = {"article_to_check": query, "articles_as_binary": [], "articles_as_types": []}

            true_articles = set(true_dicts.get(query, []))
            true_articles_with_known_conflicts = set(true_dicts_with_known_conflicts.get(query, []))    

            true_articles_count += len(true_articles)
            for a in r["articles"]:
                if a in true_articles_with_known_conflicts and a not in true_articles:
                    continue

                instance["articles_as_binary"].append(1 if a in true_articles else 0)
                instance["articles_as_types"].append(
                    classify_type(query, a, article_network) if a in true_articles else "NONE"
                )
            all_instances += len(instance["articles_as_binary"])
            results_as_binary_markings.append(instance)

        print(f"True articles count: {true_articles_count}")
        print(f"Average true articles count: {true_articles_count / len(results)}")
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
                total_ndcg += compute_ndcg(
                    binary_list,
                    [1 for i in range(len(true_dicts[r['article_to_check']]))] + [0 for j in range(k-len(true_dicts[r['article_to_check']]))],
                    k
                )

            recall = recall / len(results_as_binary_markings) * 100
            recall_cap = recall_cap / len(results_as_binary_markings) * 100
            precision = (total_true_positives / total_retrieved * 100) if total_retrieved > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            ndcg = total_ndcg / len(results_as_binary_markings) * 100

            # 표용 metric 저장
            table_metrics[k]["nDCG"] = ndcg
            table_metrics[k]["Recall"] = recall
            table_metrics[k]["Retrieval F1"] = f1

    # 모델명 추출
    model_name = os.path.basename(result_path).split(".")[0]
    row = [model_name]
    for metric in ["nDCG", "Recall", "Retrieval F1"]:
        for k in [5, 10, 50]:
            row.append(f"{table_metrics[k][metric]:.2f}")
    return row, table_metrics

if __name__ == "__main__":
    import os
    input_dir = "./outputs/retrieval_results/prf/"
    article_network = ArticleNetwork()

    # 해당 디렉토리 내 모든 파일 경로를 리스트로 수집
    paths = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
    ]
    paths.sort()

    # 모델별 결과 저장
    model_results = {
        'rocchioprf_gat': [],
        'rocchioprf_re2': [],
    }

    # 각 모델별 결과 수집
    for p in paths:
        try:
            model_name = os.path.basename(p).split(".")[0]
            for prefix in model_results.keys():
                if model_name.startswith(prefix):
                    print(f"Processing: {p}")
                    _, metrics = main(p, article_network)
                    model_results[prefix].append(metrics)
                    break
        except Exception as e:
            print(f"Error processing {p}: {e}")
            continue

    # 표 헤더
    header = ["Model"] + [f"nDCG@{k}" for k in [5,10,50]] + [f"Recall@{k}" for k in [5,10,50]] + [f"F1@{k}" for k in [5,10,50]]
    table = []

    # 각 모델 그룹별 평균과 표준편차 계산
    for model_name, results in model_results.items():
        if not results:
            continue
            
        row = [model_name]
        for metric in ["nDCG", "Recall", "Retrieval F1"]:
            for k in [5, 10, 50]:
                values = [r[k][metric] for r in results]
                mean = np.mean(values)
                std = np.std(values)
                row.append(f"{mean:.2f}±{std:.2f}")
        table.append(row)

    # 모든 모델 쌍에 대해 모든 지표에 대한 쌍체 t-검정 수행
    print("\n모델 간 쌍체 t-검정 결과:")
    print("-" * 80)
    
    # 각 지표와 k 값에 대해 모델 간 비교
    for metric in ["nDCG", "Recall", "Retrieval F1"]:
        print(f"\n[{metric} 지표에 대한 t-검정 결과]")
        for k in [5, 10, 50]:
            print(f"\n{metric}@{k}:")
            
            # 모든 모델 쌍에 대해 비교
            for i, model1 in enumerate(model_results.keys()):
                if not model_results[model1]:
                    continue
                    
                for model2 in list(model_results.keys())[i+1:]:
                    if not model_results[model2]:
                        continue
                        
                    # 각 모델의 해당 지표 값 추출
                    values1 = [r[k][metric] for r in model_results[model1]]
                    values2 = [r[k][metric] for r in model_results[model2]]
                    
                    if values1 and values2:
                        # 쌍체 t-검정 수행 (paired=True)
                        # 두 모델이 동일한 토픽에 대해 평가되었으므로 쌍체 검정이 적합
                        if len(values1) == len(values2):
                            t_stat, p_value = stats.ttest_rel(values1, values2)
                            test_type = "쌍체 t-검정"
                        else:
                            # 길이가 다른 경우 독립 표본 t-검정 수행
                            t_stat, p_value = stats.ttest_ind(values1, values2)
                            test_type = "독립 표본 t-검정"
                        
                        mean1 = np.mean(values1)
                        mean2 = np.mean(values2)
                        
                        # 결과 출력
                        if p_value < 0.05:  # 유의수준 0.05
                            better_model = model1 if mean1 > mean2 else model2
                            worse_model = model2 if better_model == model1 else model1
                            print(f"  {better_model}({mean1:.2f}) > {worse_model}({mean2:.2f}): {test_type}, t={t_stat:.4f}, p={p_value:.4f} *")
                        else:
                            print(f"  {model1}({mean1:.2f}) vs {model2}({mean2:.2f}): {test_type}, t={t_stat:.4f}, p={p_value:.4f}")

    # 결과 테이블 출력
    print("\n결과 테이블:")
    print("-" * (len(header) * 15))
    styled_table = highlight_table(table, header)
    print(tabulate(styled_table, headers=header, tablefmt="github"))
    print("-" * (len(header) * 15))
